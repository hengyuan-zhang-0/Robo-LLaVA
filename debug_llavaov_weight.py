
# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

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

from transformers import AutoConfig, AutoModel, AutoTokenizer


from llava.model import *
# from llava.model.language_model.llava_llama import LlavaConfig
from llava.model.language_model.llava_qwen import LlavaQwenConfig

import safetensors.torch 

warnings.filterwarnings("ignore")
# Load the OneVision model
pretrained =  "/share/henry/LLaVA-NeXT/checkpoints/llava-onevision-qwen2-7b-ov-robovqa-v3" #"/share/henry/LLaVA-NeXT/models--lmms-lab--llava-onevision-qwen2-7b-ov/snapshots/f45d0e455ff2ad051a1783c3d9abec41dd68134c" #"/share/henry/LLaVA-NeXT/checkpoints/llava-onevision-qwen2-7b-ov-robovqa-v3" #"lmms-lab/llava-onevision-qwen2-7b-ov"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"

llava_cfg = LlavaQwenConfig.from_pretrained(pretrained)
# overwrite_config = {'tie_word_embeddings': False, 'use_cache': True, "vocab_size": 152064} #152064
# for k, v in overwrite_config.items():
#     setattr(llava_cfg, k, v)
print(llava_cfg)
model = LlavaQwenForCausalLM.from_pretrained(pretrained, low_cpu_mem_usage=True, attn_implementation="sdpa", config=llava_cfg)



# state_dict_path = "/share/henry/LLaVA-NeXT/checkpoints/llava-onevision-qwen2-7b-ov-robovqa-v3/model.safetensors.index.json"


# index_file_path = state_dict_path

# state_dict = safetensors.torch.load_file(index_file_path, device="cpu")

# # model = LlavaQwenForCausalLM.from_pretrained(pretrained, low_cpu_mem_usage=True, attn_implementation="sdpa", config=llava_cfg)
# # Step 3: 修改 state_dict 中的权重
# for name, param in state_dict.items():
#     if param.shape == torch.Size([152064, 3584]):
#         print(f"Original shape of {name}: {param.shape}")
#         # 修正维度为 [151936, 3584]
#         state_dict[name] = param[:151936, :]  # 裁剪多余的部分
#         print(f"Modified shape of {name}: {state_dict[name].shape}")

# # Step 4: 使用修改后的 state_dict 加载模型
# model = LlavaQwenForCausalLM(config=llava_cfg)
# model.load_state_dict(state_dict, strict=False)  # strict=False 允许跳过某些不匹配的参数

# # Step 5: 模型设置为评估模式
# model.eval()

# tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa")

# model.eval()
print("load successfully")