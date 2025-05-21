import os
import re
import sys
sys.path.append("/home/zchu/codes/train_2412")

import json
import random
import argparse
import threading
import concurrent.futures
import numpy as np

from collections import defaultdict
from functools import partial
from copy import deepcopy
from tqdm import tqdm

from data_generation.utils import chat_completion_call_deepseek_msg
from data_generation.utils import em_score, pm_score, f1_score

def extract_answer(text):
    pattern = r"@@(.*?)@@"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return "Unknown"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["utility", "groundness", "relevance"], default=None)
    parser.add_argument("--task_input", type=str, default=None)
    #parser.add_argument("--task_output", type=str, default=None)
    parser.add_argument("--n", type=int, default=1)
    args = parser.parse_args()
    
    assert args.task_input is not None 
    return args

def process_func(item):
    input_prompt = item["input_prompt"]
    try:
        response = chat_completion_call_deepseek_msg(
            messages=[
                {"role": "user", "content": input_prompt}
            ],
            temperature=0.0,
            model="deepseek-chat"
        )
        item["response"] = response[0]
        with writing_lock:
            fcache_obj.write(json.dumps(item, ensure_ascii=False) + "\n")
            fcache_obj.flush()
        return item
            
    except Exception as e:
        print(e)
        item["response"] = "FAILED"
        with writing_lock:
            fcache_obj.write(json.dumps(item, ensure_ascii=False) + "\n")
            fcache_obj.flush()
        return item

if __name__ == "__main__":
    args = parse_args()
    print("Args:", json.dumps(args.__dict__, ensure_ascii=False, indent=4))
    writing_lock = threading.Lock()
    #output_dir = f"./outputs/{args.task}/{args.task_input}"
    output_dir = f"./data_generation/ours_T2/critic/outputs/{args.task}/{args.task_input}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    datas = json.load(open(f"./data_generation/ours_T2/critic/{args.task}/outputs/{args.task_input}.json"))
    fcache_obj = open(f"{output_dir}/cache.jsonl", "a")
    print(f"Number of datas: {len(datas)}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=400) as executor:
        results = list(tqdm(executor.map(process_func, datas), total=len(datas)))
    
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    fcache_obj.close()
    
## Running command:
## Utility
# python -m data_generation.ours_T2.critic.main --task utility --task_input zeroshot_v1
# python -m data_generation.ours_T2.critic.main --task utility --task_input zeroshot_v2 (we use this one)
## Groundness
# python -m data_generation.ours_T2.critic.main --task groundness --task_input fewshot_v2 (we use this one)
# python -m data_generation.ours_T2.critic.main --task groundness --task_input zeroshot_v2
## Relevance
# python -m data_generation.ours_T2.critic.main --task relevance --task_input fewshot_v2 (we use this one)
# python -m data_generation.ours_T2.critic.main --task relevance --task_input zeroshot_v2