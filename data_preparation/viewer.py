import jsonlines
import json
import random
import os
from pathlib import Path

import debugpy
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass


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



llava_pretrain_558k =   "/share/lmz/data/llava-v1.5-instruct/llava_v1_5_lrv_mix1008k.json"
llava_pretrain_558k = Path(llava_pretrain_558k)

dataset = read_json(llava_pretrain_558k)

data_len = len(dataset)

for data in dataset:
    
    if 'image' in data and isinstance(data['image'], list):
        place_holder  = None