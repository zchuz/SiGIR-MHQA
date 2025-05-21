# create date: 2024-12-23
# conda environment: vllm
# Test the critic model

import os 
import re
import sys
import json
import torch
import random
import argparse
import threading
import numpy as np
import concurrent.futures
from tqdm import tqdm
from copy import deepcopy
from itertools import chain
from functools import partial
from collections import defaultdict
from typing import List
from transformers import AutoTokenizer
from inference.utils import get_logger
from vllm import LLM, SamplingParams

logger = get_logger(__file__)

CHAT_TEMPLATE = "[INST] {} [/INST]"

def load_datas():import os 
import re
import sys
import json
import torch
import random
import argparse
import threading
import numpy as np
import concurrent.futures
from tqdm import tqdm, trange
from transformers import AutoTokenizer
INSTRUCTION_TEMPLATE = {
    "groundness": "You will be provided with a question, an evidence document, along with a response.\nYour job is to determine whether the response is supported by the evidence, and provide explanation for your decision.\nUse the following scale to rate the response:\n[Fully supported] Most of the information in the response is supported by the evidence.\n[Partially supported] Some of the information in the response is supported by the evidence, but there are some parts that are speculative.\n[Not supported] The response is not supported by the evidence / The response does not provide useful reasoning.",
    "relevance": "You will be provided with a question, along with an evidence document.\nYour job is to determine whether the evidence is relevant to the question, and provide explanation for your decision.\nIf the evidence is relevant to the question, you should response with [Relevant]; otherwise, response with [Irrelevant].",
    "utility": "You will be provided with a question, along with a reasoning trajectory.\nYour job is to determine whether the reasoning trajectory is useful for answering the question, and provide explanation for your decision.\nUse the following scale to rate the reasoning:\n[4]: The reasoning process is clear, logically structured, and well-supported by the evidence.\n[3]: The reasoning process is mostly clear, partially logically structured, and supported by evidence, but may contains minor logical flaws.\n[2]: The reasoning is somewhat unclear with noticeable flaws and uses limited or weak evidence.\n[1]: The reasoning is flawed, lacking supporting evidence, which results in an incorrect conclusion.\n[0]: The reasoning fails to provide a definitive answer."
}
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/home/share/models/Mistral-7B-v0.2")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--task", type=str, choices=["groundness", "relevance", "utility"], nargs="+")
    parser.add_argument("--fin", type=str, default=None, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_datas", type=int, default=None)
    parser.add_argument("--inference_mode", type=str, choices=["direct", "pre_cot", "post_cot"], default="pre_cot")
    args = parser.parse_args()
    assert args.adapter_path is None, "Please first merge the adapter."
    return args

def parse_critic_output(text, mode):
    if mode == "pre_cot":
        explanation, rating = text.split("\n")
        explanation = explanation.strip()
        rating = rating.split("Rating: ")[-1].strip()
    elif mode == "post_cot": 
        rating, explanation = text.split("\n")
        explanation = explanation.strip()
        rating = rating.split("Rating: ")[-1].strip()
    elif mode == "direct":
        rating = text.strip()
        explanation = ""
    else:
        raise ValueError(f"Invalid inference mode: {mode}")
    
    return explanation, rating

def load_datas():
    all_datas = []
    #datas = json.load(open("./data_generation/ours_T2/critic/outputs/test.json"))
    datas = json.load(open(args.fin))
    for task in args.task:
        task_datas = [item for item in datas if item["meta"]["task"] == task][:args.num_datas]
        all_datas.extend(task_datas)
        logger.info(f"Load {len(task_datas)} {task} datas")
    
    logger.info(f"Total Testing datas: {len(all_datas)}")
    return all_datas

if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Args: {json.dumps(args.__dict__, indent=4)}")
    output_dir = f"./inference/outputs/critic/{'_'.join(args.task)}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    writing_lock = threading.Lock()
    fout = os.path.join(output_dir, "results.json")
    fcache_obj = open(os.path.join(output_dir, "cache.jsonl"), "w")

    datas = load_datas()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = LLM(model=args.model_name_or_path)
    sampling_params = SamplingParams(max_tokens=768, temperature=0.0, stop_token_ids=tokenizer.eos_token)
    accuracies = []
    for i in trange(0, len(datas), args.batch_size):
        
        batch_datas = datas[i:i+args.batch_size]
        batch_instructions = [item["instruction"] for item in batch_datas]
        batch_prompts = [CHAT_TEMPLATE.format(instruction) for instruction in batch_instructions]
        results = model.generate(
            batch_prompts,
            sampling_params=sampling_params,
        )
        outputs = [item.outputs[0].text for item in results]
        explanations, ratings = [], []
        for item in outputs:
            explanation, rating = parse_critic_output(item, mode=args.inference_mode)
            explanations.append(explanation)
            ratings.append(rating)
            
            # explanation, rating = item.split("\n")
            # explanations.append(explanation.strip())
            # ratings.append(rating.split("Rating: ")[-1].strip())
        
        for i in range(len(batch_datas)):
            label = batch_datas[i]["output"]["rating"]
            batch_datas[i]["completion"] = outputs[i]
            batch_datas[i]["explanation"] = explanations[i]
            batch_datas[i]["prediction"] = ratings[i]
            batch_datas[i]["accuracy"] = float(ratings[i] == label)
            accuracies.append(batch_datas[i]["accuracy"])
        
        pass
    
    with open(fout, "w") as fout:
        json.dump(datas, fout, indent=4, ensure_ascii=False)
    
    metrics = {}
    for task in args.task:
        task_datas = [item for item in datas if item["meta"]["task"] == task]
        acc = np.mean([item["accuracy"] for item in task_datas])
        metrics[task] = acc
        logger.info(f"{task}: {acc:.3f}")
    
    with open(os.path.join(output_dir, "metrics.json"), "w") as fout:
        json.dump(metrics, fout, indent=4, ensure_ascii=False)
    

    
