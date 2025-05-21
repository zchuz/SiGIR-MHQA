# conda environment: vllm (or any else)
# create date: 2024/12/17
# convert the t2_inference results into relevance input format.
# target format: {
#     "question": str,
#     "evidence": str,
# }

import os
import json
import argparse
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fin", type=str, default="/home/zchu/codes/train_2412/inference/outputs/t2_train/0_None/results.json")
    parser.add_argument("--mode", type=str, choices=["zeroshot", "fewshot"], default="fewshot")
    args = parser.parse_args()
    return args

def parse_one_line(item):
    reasoning, answer, query_list, completion_list, documents = item["reasoning"], item["answer"], item["query_list"], item["completion_list"], item["documents"]
    return_examples = []
    for query, doc in zip(query_list, documents):
        query = query.strip()
        doc_text = f'{doc["title"]}\n{doc["paragraph_text"]}'
        input_prompt = prompt_template.format(query, doc_text)
        tmp = {
            "input_prompt": input_prompt,
            "qid": item["qid"],
            "type": item["type"],
            "dataset": item["dataset"],
            "task": "relevance"
        }
        return_examples.append(tmp)
    return return_examples
        
def load_prompt():
    if args.mode == "zeroshot":
        prompt = open("./data_generation/ours_T2/critic/relevance/prompts/v2/relevance_zeroshot.txt").read()
    elif args.mode == "fewshot":
        prompt = open("./data_generation/ours_T2/critic/relevance/prompts/v2/relevance.txt").read()
    return prompt
    
if __name__ == "__main__":

    args = parse_args()
    print(f"Parsing Arguments: {json.dumps(args.__dict__, indent=4)}")
    
    datas = json.load(open(args.fin))
    datas = [item for item in datas if item["flag"]]
    datas = [item for item in datas if (2 * len(item["query_list"]) + 1 == len(item["completion_list"]))]
    prompt_template = load_prompt()
    
    fout = f"./data_generation/ours_T2/critic/relevance/outputs"
    if not os.path.exists(fout):
        os.makedirs(fout)
    
    processed_examples = []
    for item in tqdm(datas):
        processed_examples.extend(parse_one_line(item))
    
    print(f"Num of Datas: {len(datas)}")
    print(f"Num of Instances: {len(processed_examples)}")
    print(f"Average Num of Instances per Data: {len(processed_examples) / len(datas):.2f}")
    
    with open(f"{fout}/{args.mode}_v2.json", "w") as f:
        json.dump(processed_examples, fp=f, ensure_ascii=False, indent=4)
    
## Example command
# python -m data_generation.ours_T2.critic.relevance.preprocess --mode fewshot
    
    
    
    
    
    