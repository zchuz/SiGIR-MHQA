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

from data_generation.utils import chat_completion_call_openai_msg, chat_completion_call_deepseek_msg
from data_generation.utils import load_datas, em_score, pm_score, f1_score

def extract_answer(text):
    pattern = r"@@(.*?)@@"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return "Unknown"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["gpt-4o-mini", "deepseek"], default="deepseek")
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()
    return args

def prepare_prompts(data, prompt_template):
    question = data["question_text"]
    contexts = data["context"]
    supporting_contexts = [
        item for item in contexts if item["is_supporting"]
    ]
    context_str = ""
    for i, sf in enumerate(supporting_contexts):
        title = sf["title"]
        text = sf["text"]
        context_str += f"### Document #{i+1}: {title}\n{text}\n\n"
    return prompt_template.format(context_str.strip(), question)

PROMPT_TEMPLATE_BRIDGE_2HOP = open("prompts_v2.5/bridge_2hop.txt").read()
PROMPT_TEMPLATE_BRIDGE_3HOP = open("prompts_v2.5/bridge_3hop.txt").read()
PROMPT_TEMPLATE_BRIDGE_4HOP = open("prompts_v2.5/bridge_4hop.txt").read()
PROMPT_TEMPLATE_COMPARISON = open("prompts_v2.5/comparison.txt").read()
PROMPT_TEMPLATE_BRIDGE_COMPARISON = open("prompts_v2.5/bridge_comparison.txt").read()

if __name__ == "__main__":
    args = parse_args()
    save_base = f"outputs/{args.model}_1215_v2.5"
    if not os.path.exists(save_base):
        os.makedirs(save_base)
    fout = os.path.join(save_base, "results.json")
    fcache_obj = open(os.path.join(save_base, "cache.jsonl"), "w")
    
    writing_lock = threading.Lock()
    datas_2wiki = json.load(open("/home/zchu/datasets/mhqa/datas_for_acl25_mhqa/2wiki/train.20k.json"))
    datas_hotpot = json.load(open("/home/zchu/datasets/mhqa/datas_for_acl25_mhqa/hotpotqa/train.10k.json"))
    datas_music = json.load(open("/home/zchu/datasets/mhqa/datas_for_acl25_mhqa/musique/train.20k.json"))
    datas = datas_2wiki + datas_hotpot + datas_music
    datas = datas[:]
    print(f"Dataset Info:\n\tTotal: {len(datas)}\n\t2Wiki: {len(datas_2wiki)}\n\tHotpotQA: {len(datas_hotpot)}\n\tMusique: {len(datas_music)}")
    
    def process_func(item, args):
        qtype = item["type"]
        if qtype in ["comparison"]:
            prompt = prepare_prompts(item, PROMPT_TEMPLATE_COMPARISON)
            item["prompt"] = "PROMPT_TEMPLATE_COMPARISON"
        elif qtype in ["bridge_comparison"]:
            prompt = prepare_prompts(item, PROMPT_TEMPLATE_BRIDGE_COMPARISON)
            item["prompt"] = "PROMPT_TEMPLATE_BRIDGE_COMPARISON"
        elif qtype in ["inference", "bridge", "compositional", "2hop"]:
            prompt = prepare_prompts(item, PROMPT_TEMPLATE_BRIDGE_2HOP)
            item["prompt"] = "PROMPT_TEMPLATE_BRIDGE_2HOP"
        elif qtype in ["3hop1", "3hop2"]:
            prompt = prepare_prompts(item, PROMPT_TEMPLATE_BRIDGE_3HOP)
            item["prompt"] = "PROMPT_TEMPLATE_BRIDGE_3HOP"
        elif qtype in ["4hop1", "4hop2", "4hop3"]:
            prompt = prepare_prompts(item, PROMPT_TEMPLATE_BRIDGE_4HOP)
            item["prompt"] = "PROMPT_TEMPLATE_BRIDGE_4HOP"
        else:
            prompt = prepare_prompts(item, PROMPT_TEMPLATE_BRIDGE_2HOP)
            item["prompt"] = "PROMPT_TEMPLATE_BRIDGE_2HOP"
        completion_fn = chat_completion_call_deepseek_msg if args.model == "deepseek" else chat_completion_call_openai_msg
        try:
            if args.n == 1:
                responses = completion_fn(
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    model=args.model,
                    temperature=0.0
                )
                item["completions"] = responses
                with writing_lock:
                    fcache_obj.write(json.dumps(item, ensure_ascii=False) + "\n")
                    fcache_obj.flush()
            else:
                if completion_fn is chat_completion_call_openai_msg:
                    responses = completion_fn(
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        model=args.model,
                        temperature=1.15,
                        top_p=1.0,
                        n=args.n
                    )
                    item["completions"] = responses
                elif completion_fn is chat_completion_call_deepseek_msg:
                    responses = []
                    response_greedy = completion_fn(
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        model=args.model,
                        temperature=0.0
                    )[0]
                    responses.append(response_greedy)
                    for _ in range(args.n - 1):
                        response_sampling = completion_fn(
                            messages=[
                                {"role": "user", "content": prompt}
                            ],
                            model=args.model,
                            temperature=1.15,
                            top_p=1.0
                        )[0]
                        responses.append(response_sampling)
                    item["completions"] = responses
                with writing_lock:
                    fcache_obj.write(json.dumps(item, ensure_ascii=False) + "\n")
                    fcache_obj.flush()
        except Exception as e:
            print(e)
            item["completions"] = ["FAILED"]
            with writing_lock:
                fcache_obj.write(json.dumps(item, ensure_ascii=False) + "\n")
                fcache_obj.flush()
        return responses

    with concurrent.futures.ThreadPoolExecutor(max_workers=500) as executor:
        process_func_partial = partial(process_func, args=args)
        results = list(tqdm(executor.map(process_func_partial, datas), total=len(datas)))
    
    for item in datas:
        answer = item["answer"]
        completions = item["completions"]
        extracted_answers = [extract_answer(completion) for completion in completions]
        item["predictions"] = extracted_answers
        item["metrics"] = {
            "em" : [em_score(p, answer) for p in extracted_answers],
            "pm" : [pm_score(p, answer) for p in extracted_answers],
            "f1" : [f1_score(p, answer) for p in extracted_answers],
        }
        
    
    overall_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    datasets = set([item["dataset"] for item in datas[:]])
    for dataset in datasets:
        print(f"Evaluating {dataset}...")
        dataset_datas = [item for item in datas[:] if item["dataset"] == dataset]
        avg_metrics = {
            "em" : np.mean([np.mean(item["metrics"]["em"]) for item in dataset_datas]),
            "pm" : np.mean([np.mean(item["metrics"]["pm"]) for item in dataset_datas]),
            "f1" : np.mean([np.mean(item["metrics"]["f1"]) for item in dataset_datas])
        }
        overall_metrics[dataset]["avg_metrics"] = avg_metrics
        print(f"\tAverage Metrics over Qtype: {avg_metrics}")
        
        qtypes = set([item["type"] for item in dataset_datas])
        for qtype in qtypes:
            print(f"\tEvaluating {qtype}...")
            qtype_datas = [item for item in dataset_datas if item["type"] == qtype]
            avg_metrics = {
                "em" : np.mean([np.mean(item["metrics"]["em"]) for item in qtype_datas]),
                "pm" : np.mean([np.mean(item["metrics"]["pm"]) for item in qtype_datas]),
                "f1" : np.mean([np.mean(item["metrics"]["f1"]) for item in qtype_datas])
            }
            overall_metrics[dataset][qtype] = {
                "avg_metrics" : avg_metrics,
            }
            print(f"\t\tAverage metrics: {avg_metrics}")
            for k, v in avg_metrics.items():
                print(f"\t\t{k} : {v:.4f}")
                
    with open(f"{save_base}/overall_metrics.json", "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, indent=4, ensure_ascii=False)
    # Save processed data
    with open(fout, 'w') as f:
        json.dump(datas, f, indent=2, ensure_ascii=False)
    
    fcache_obj.close()