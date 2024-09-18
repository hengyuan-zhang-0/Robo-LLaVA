from collections import defaultdict
import os
import sacrebleu
# from sacrebleu.metrics import BLEU
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import logging
import sys
import jieba

eval_logger = logging.getLogger("lmms-eval")

dir_name = os.path.dirname(os.path.abspath(__file__))

# # 19 classes
# eval_type_dict = {
#     "Sensation": ["count","color", "scene", "poster", "attribute_recognition", "ocr", "position"],
#     "Cognition": ["calculation", "code", "translation", "math", "cross_instance_reason", "attribute_reason"],
#     "Knowledge": ["celebrity", "chemistry", "physics", "biology", "landmark", "artwork"]
# }

def egothink_doc_to_images(doc):
    images_path = "/home/pd/Dataset/EgoThink/parsed_test"
    image_path = os.path.join(images_path, doc["image_name"] + ".jpg")
    if os.path.exists(image_path):
        image_path = image_path
    else:
        sys.exit(f"image {image_path} does not exist.")
    
    return [image_path]

def egothink_doc_to_text(doc, lmms_eval_specific_kwargs=None):#这里的第二个参数是什么意思？这个原版功能是去看那个instructions吗？
                                                              #是需要也改成for循环吗，然后answer和question要怎么处理一下
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]#这个第二个参数是啥意思？
    question = doc["question"].strip()#这里的question是question的path吗？
    return f"{pre_prompt}{question}{post_prompt}"#这里是返回了一些啥？为啥不是返回一个list


def egothink_doc_to_target(doc):#这个doc咋搞的？怎么一下子就能找到路径
    answer = doc["answer"].split(":")[1].replace('(',' ').replace(')',' ').strip()
    return answer#这个是返回了path？



def parse_pred_ans_NY(pred_ans):
    pred_label = None
    if pred_ans in ["yes", "no"]:
        pred_label = pred_ans
    else:
        prefix_pred_ans = pred_ans[:4]

        if "yes" in prefix_pred_ans:
            pred_label = "yes"
        elif "no" in prefix_pred_ans:
            pred_label = "no"
        else:
            pred_label = "other"
    return pred_label


def parse_pred_ans_choice(pred_ans):
    return pred_ans.replace(" ", "")[0]


def egothink_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mme score), value: metric value
    """
    pred = results[0].replace('A:',' ').replace(':',' ').replace('(',' ').replace(')',' ').strip()
    pred = pred.replace('\n', "").lower()
    # parser
    if doc["question_field"] == "N/Y":
        pred_ans = parse_pred_ans_NY(pred)
    elif doc["question_field"] == "Choices":
        pred_ans = parse_pred_ans_choice(pred)
    else:
        pred_ans = pred
    
    #gt_ans = doc["answer"].lower()
    pred_ans = pred_ans.lower()
    gt_ans = doc["answer"].lower().split(":")[1].replace('(',' ').replace(')',' ').strip()
    # ans =  pred_ans
    # gt = gt_ans
    ans = "".join(jieba.cut(pred_ans)).strip()
    gt = ''.join(jieba.cut(gt_ans)).strip()
        


    score =[
        max(sentence_bleu([ans[:len(gt)-lenth].split()], gt.split(), smoothing_function=SmoothingFunction().method4, weights=(1,0,0,0)) for lenth in range(-5,5)),
        max(sentence_bleu([ans[:len(gt)-lenth].split()], gt.split(), smoothing_function=SmoothingFunction().method4, weights=(0,1,0,0)) for lenth in range(-5,5)),
        max(sentence_bleu([ans[:len(gt)-lenth].split()], gt.split(), smoothing_function=SmoothingFunction().method4, weights=(0,0,1,0)) for lenth in range(-5,5)),
        max(sentence_bleu([ans[:len(gt)-lenth].split()], gt.split(), smoothing_function=SmoothingFunction().method4, weights=(0,0,0,1)) for lenth in range(-5,5)),
    ]

    # score
    # score = 1 if (doc["question_field"] == "Q/A" and anls_score(prediction=pred_ans, gold_labels=[gt_ans], threshold=0.95) >= 0.4) \
    #                 or (gt_ans == pred_ans) \
            # else 0

    



    return {"egothink":{"video_id": doc["video_name"], "ans": ans , "gt": gt,
                       "BLEU_1": score[0], "BLEU_2": score[1], "BLEU_3": score[2], "BLEU_4": score[3]}}

def egothink_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    summary = defaultdict(dict)
    BLEU_1 = 0
    BLEU_2 = 0
    BLEU_3 = 0
    BLEU_4 = 0
    
    for result in results:
        video_id = result["video_id"]
        BLEU_1 += result["BLEU_1"]
        BLEU_2 += result["BLEU_2"]
        BLEU_3 += result["BLEU_3"]
        BLEU_4 += result["BLEU_4"]

    BLEU_1 = BLEU_1 / (len(results))
    BLEU_2 = BLEU_2 / (len(results))
    BLEU_3 = BLEU_3 / (len(results))
    BLEU_4 = BLEU_4 / (len(results))
        

    print(f"BLEU-1: {BLEU_1:.2f},  BLEU-2: {BLEU_2:.2f},  BLEU-3: {BLEU_3:.2f},  BLEU-4: {BLEU_4:.2f}")
    # cnt_con = cnt_con / (len(results) / 3)
    eval_logger.info(f"BLEU-1: {BLEU_1:.2f},  BLEU-2: {BLEU_2:.2f},  BLEU-3: {BLEU_3:.2f},  BLEU-4: {BLEU_4:.2f}")
    return BLEU_1, BLEU_2, BLEU_3, BLEU_4


def egothink_aggregate_res_bleu_1(results):
    BLEU_1, _, _ , _ = egothink_aggregate_results(results)
    return BLEU_1

def egothink_aggregate_res_bleu_2(results):
    _, BLEU_2, _ , _ = egothink_aggregate_results(results)
    return BLEU_2
def egothink_aggregate_res_bleu_3(results):
    _, _, BLEU_3 , _ = egothink_aggregate_results(results)
    return BLEU_3
def egothink_aggregate_res_bleu_4(results):
    _, _, _ , BLEU_4 = egothink_aggregate_results(results)
    return BLEU_4

    

# def egothink_aggregate_results(results):
#     """
#     Args:
#         results: a list of values returned by process_results
#     Returns:
#         A score
#     """
#     summary = defaultdict(dict)
#     for result in results:
#         video_id = result["video_id"]
#         score = result["score"]
#         if video_id not in summary.keys():
#             summary[video_id] = 0
#         summary[video_id] += score

#     cnt_con = 0
#     for video_id, score in summary.items():
#         if score == 3:
#             cnt_con += 1
    
#     print("Consistency Cases are ", cnt_con)
#     cnt_con = cnt_con / (len(results) / 3)
#     eval_logger.info(f"ConScore_D: {cnt_con:.2f}")
#     return cnt_con