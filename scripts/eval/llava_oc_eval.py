from operator import attrgetter
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu

import jsonlines
import json


import random
import os
import copy
from tqdm import tqdm
import jieba
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction

warnings.filterwarnings("ignore")




def read_jsonl(file):
    results = []
    with open(file, "r", encoding='utf-8') as f:
        for item in jsonlines.Reader(f):
            results.append(item)
    return results


def read_json(file):
    with open(file, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data


def write_json(file, data):
    with open(file, "w") as f:
        json.dump(data, f, indent=2)

# Function to extract frames from video
def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)

    
# Load the OneVision model
pretrained = "/share/henry/LLaVA-NeXT/checkpoints/llava-onevision-qwen2-7b-ov-robovqa-v3" #"lmms-lab/llava-onevision-qwen2-7b-ov"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa")

model.eval()

# Load data
robovqa_dataset = read_json("/share/world_model/EVA/train_data/robot_vqa_val.json")
val = []


res_eval = []

for data in tqdm(robovqa_dataset):
    question = data['conversations'][0]['value']
    gt_ans = data['conversations'][1]['value']
    gt_ans = gt_ans.split(":")[1].replace(':',' ').replace('(',' ').replace(')',' ')
    video_path = data['video']
    video_path = os.path.join("/share/world_model/robovqa/videos", video_path)
    
    # Load and process video
    video_frames = load_video(video_path, 16)  #论文里max_frames_num最大32

    image_tensors = []
    frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
    image_tensors.append(frames)

    # Prepare conversation input
    conv_template = "qwen_1_5"


    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [frame.size for frame in video_frames]
    modalities = ["video"] * len(video_frames)

    # Generate response
    cont = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
        modalities=modalities,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    #print(text_outputs[0])
    
    
    uid = data['id']
    
    
    gt =''.join(jieba.cut(gt_ans)).strip()

    pred_ans = text_outputs[0].split(":")[-1].replace(':',' ').replace('(',' ').replace(')',' ')
    ans = "".join(jieba.cut(pred_ans)).strip()
    
    gt = gt.lower()
    ans = ans.lower()
    
    BLEU = [
        max(sentence_bleu([ans[:len(gt)-lenth].split()], gt.split(), smoothing_function=SmoothingFunction().method4, weights=(1,0,0,0)) for lenth in range(-5,5)),
        max(sentence_bleu([ans[:len(gt)-lenth].split()], gt.split(), smoothing_function=SmoothingFunction().method4, weights=(0,1,0,0)) for lenth in range(-5,5)),
        max(sentence_bleu([ans[:len(gt)-lenth].split()], gt.split(), smoothing_function=SmoothingFunction().method4, weights=(0,0,1,0)) for lenth in range(-5,5)),
        max(sentence_bleu([ans[:len(gt)-lenth].split()], gt.split(), smoothing_function=SmoothingFunction().method4, weights=(0,0,0,1)) for lenth in range(-5,5)),
    ]
    
    val.append(BLEU)
        
        
    
    res_eval.append({
        "id": uid,
        "question": question,
        "video_path": video_path,
        "gt_ans": gt,
        "pred_ans": ans,
        "BLEU": BLEU
        
    })
 
BLEU_1 = sum([i[0] for i in val])/len(val)
BLEU_2 = sum([i[1] for i in val])/len(val)
BLEU_3 = sum([i[2] for i in val])/len(val)
BLEU_4 = sum([i[3] for i in val])/len(val)

print(f"BLEU-1: {BLEU_1:.2f},  BLEU-2: {BLEU_2:.2f},  BLEU-3: {BLEU_3:.2f},  BLEU-4: {BLEU_4:.2f}")

write_json("/share/henry/LLaVA-NeXT/scripts/eval/llavaoc-robovqa_scores.json", res_eval)
    

