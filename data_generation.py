import json
import os
from tqdm import tqdm
import random
import jsonlines

def get_conversations_video(question, answer):
    conversations = [
        {
            "from": "human",
            "value": f"<video>\n{question}",
        },
        {
            "from": "gpt",
            "value": answer,
        }
    ]
    
    return conversations

def read_json(file):
    with open(file, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data

def read_jsonl(file):
    results = []
    with open(file, "r", encoding='utf-8') as f:
        for item in jsonlines.Reader(f):
            results.append(item)
    return results

def write_json(file, data):
    with open(file, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False) # 确保保存为UTF-8编码

#input_path = '<robovqa_path>'   #TODO: here
input_path =  "/share/henry/json/val/data-00000-of-00021.json"
dataset = read_jsonl(input_path)
new_dataset = []
for data in tqdm(dataset[:2]):
    print(data['text'])
    # data['conversations'] = get_conversations_video(data['Question'], data['Answer'])
    # new_dataset.append(data)


