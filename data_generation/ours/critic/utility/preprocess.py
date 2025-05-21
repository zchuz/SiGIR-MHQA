# conda environment: vllm (or any else)
# create date: 2024/12/17
# convert the t2_inference results into utility input format.
# target format: {
#     "question": str,
#     "reasoning": str,
# }

import os
import re
import json
import argparse
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
from data_generation.utils import em_score, pm_score, f1_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fin", type=str, default="/home/zchu/codes/train_2412/inference/outputs/t2_train/0_None/results.json")
    parser.add_argument("--mode", type=str, choices=["zeroshot"], default="zeroshot")
    parser.add_argument("--prompt", type=str, choices=["v1", "v2"], default="v1")
    args = parser.parse_args()
    return args


def extract_answer(text):
    pattern = r"@@(.*?)@@"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return "Unknown"

def parse_one_line(item):
    reasoning = item["reasoning"]
    question = re.findall(r"\[Question\](.*?)\[\/INST\]", reasoning)[0]
    response = reasoning.split("[/INST]")[1].strip()
    input_prompt = prompt_template.format(question, response)
    prediction = extract_answer(reasoning)
    ground_truth = item["answer"]
    em, pm, f1 = em_score(prediction, ground_truth), pm_score(prediction, ground_truth), f1_score(prediction, ground_truth)[0]
    return {
        "input_prompt": input_prompt,
        "qid": item["qid"],
        "type": item["type"],
        "dataset": item["dataset"],
        "answer": item["answer"],
        "prediction": prediction,
        "task": "utility",
        "metrics": {
            "em": em,
            "pm": pm,
            "f1": f1
        }
    }
    
        
def load_prompt():
    if args.prompt == "v1":
        prompt = open("./data_generation/ours_T2/critic/utility/prompts/utility_v1.txt").read()
    elif args.prompt == "v2":
        prompt = open("./data_generation/ours_T2/critic/utility/prompts/utility_v2.txt").read()
    return prompt
    
if __name__ == "__main__":

    args = parse_args()
    print(f"Parsing Arguments: {json.dumps(args.__dict__, indent=4)}")
    
    datas = json.load(open(args.fin))
    datas = [item for item in datas if item["flag"]]
    datas = [item for item in datas if (2 * len(item["query_list"]) + 1 == len(item["completion_list"]))]
    prompt_template = load_prompt()
    
    fout = f"./data_generation/ours_T2/critic/utility/outputs"
    if not os.path.exists(fout):
        os.makedirs(fout)
    
    processed_examples = []
    for item in tqdm(datas):
        processed_examples.append(parse_one_line(item))
    
    print(f"Num of Datas: {len(datas)}")
    print(f"Num of Instances: {len(processed_examples)}")
    print(f"Average Num of Instances per Data: {len(processed_examples) / len(datas):.2f}")
    
    with open(f"{fout}/{args.mode}_{args.prompt}.json", "w") as f:
        json.dump(processed_examples, fp=f, ensure_ascii=False, indent=4)
    
## Example command
# python -m data_generation.ours_T2.critic.utility.preprocess --mode zeroshot --prompt v2