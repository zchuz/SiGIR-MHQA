# create date: 2025-01-02
# conda environment: vllm
# This file is used to add critic tags with trained critic model to the vanilla output.

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
from concurrent.futures import ProcessPoolExecutor

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from peft import PeftModel
from vllm import LLM, SamplingParams

from inference.utils import get_logger
from data_generation.utils import completion_call_vllm
from inference.t2_parallel import extract_query


logger = get_logger(__file__)
CHAT_TEMPLATE = "[INST] {} [/INST]"
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="./train/outputs/t2_critic/groundness_relevance_utility/mistral_critic_all_v3.1_bs64_lr5e-5_ep2_1231-1843-Fmb/merged_checkpoint", help="The path of the critic model used to rating the vanilla data")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Wether to overwrite the existing merged model")
    parser.add_argument("--fin", type=str, default=None, nargs="+")
    parser.add_argument("--start", type=int, default=0, help="Start data index for inference")
    parser.add_argument("--end", type=int, default=None, help="End data index for inference")
    parser.add_argument("--max_workers", type=int, default=500)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    return args

def load_train_datas():
    all_datas = []
    for fi in args.fin:
        datas = json.load(open(fi))
        all_datas.extend(datas)
        logger.info(f"Loading Training datas for Inference from {fi}")
    logger.info(f"Totally {len(all_datas)} datas for Inference.")
    return all_datas

def vanilla_inference_v2_parse_only(data):
    try:
        instruction = data["instruction"]
        output = data["output"]
        query_list, completion_list, reasoning_list, document_list = [], [], [], []
        prediction = "FAILED"
        current_index = 0
        sentences = []
        while True:
            eol_index = output.find("\n\n", current_index)
            paragraph_index = output.find("<paragraph>", current_index)
            eop_index = output.find("</paragraph>", current_index)
            if eol_index == -1:
                eol_index = 99999999999
            else:
                eol_index = eol_index + 2
            
            if paragraph_index == -1:
                paragraph_index = 99999999999
            else:
                paragraph_index = paragraph_index + 11
                #+11
            
            if eop_index == -1:
                eop_index = 99999999999
            else:
                eop_index = eop_index + 12
                
            min_break_index = min(eol_index, paragraph_index, eop_index)
            if min_break_index == paragraph_index: # question
                sentence = output[current_index:min_break_index-11]
            elif min_break_index == eop_index: # document:
                sentence = output[current_index-11:min_break_index]
            else:
                sentence = output[current_index:min_break_index]
            if sentence == "":
                break
            sentences.append(sentence)
            current_index = min_break_index
        
        try:
            qtype = data["meta"]["type"]
            if qtype in ["comparison", "bridge_comparison"]:
                assert len(sentences) % 3 == 2
            else:
                assert len(sentences) % 3 == 1
        except Exception as e:
            print(f"qid: {data['meta']['id']}, qtype: {qtype}, len(sentences): {len(sentences)}")
            return None
        
        num_of_groups = len(sentences) // 3
        groups = []
        completion_not_in_groups = []
        for i in range(num_of_groups):
            reasoning_group = sentences[i*3:(i+1)*3]
            question_text, document_text, reasoning_text = reasoning_group
            query = extract_query(question_text)
            document = document_text.replace("<paragraph>", "").replace("</paragraph>", "").strip()
            reasoning = reasoning_text.replace("</sub_question>", "").strip()
            query_list.append(query)
            document_list.append(document)
            reasoning_list.append(reasoning)
            completion_list.extend([question_text, reasoning_text])
            groups.append((query, document, reasoning))
        completion_not_in_groups.extend(sentences[-(len(sentences) - 3 * num_of_groups):])
        if len(sentences) % 3 == 1:
            completion_list.append(sentences[-1])
            prediction = sentences[-1]
        elif len(sentences) % 3 == 2:
            completion_list.extend(sentences[-2:])
            query_list.append(extract_query(sentences[-2]))
            prediction = sentences[-1]
        
        def prepare_document(document_text):
            title = document_text.split(":")[0].strip()
            text = ":".join(document_text.split(":")[1:]).strip()
            return f"{title}\n{text}"

        def extract_rating(completion):
            completion = completion.strip("</s>")
            explanation, rating = completion.split("\n")
            rating = rating.split("Rating: ")[-1].strip()
            return rating
        
        group_results = []
        inst_gnd = open("data_generation/ours_T2/critic/groundness/prompts/v2/instruction.txt").read()
        #inst_rel = open("data_generation/ours_T2/critic/relevance/prompts/v2/instruction.txt").read()
        inst_rel = open("data_generation/ours_T2/critic/relevance/prompts/v2.1/instruction.txt").read()
        inst_util = open("data_generation/ours_T2/critic/utility/prompts/instruction.txt").read() 
        for group in groups:
            query, document, reasoning = group
            formatted_document = prepare_document(document)
            input_prompt_rel = inst_rel.format(query, formatted_document)
            completion_rel = completion_call_vllm(prompt=CHAT_TEMPLATE.format(input_prompt_rel), model=args.model_name_or_path).choices[0].text
            
            input_prompt_gnd = inst_gnd.format(query, formatted_document, reasoning)
            completion_gnd = completion_call_vllm(prompt=CHAT_TEMPLATE.format(input_prompt_gnd), model=args.model_name_or_path).choices[0].text
            
            group_results.append({
                "relevance" : extract_rating(completion_rel),
                "groundness" : extract_rating(completion_gnd),
            })
            
        mhq = instruction.replace("[Question]", "")
        input_prompt_util = inst_util.format(mhq, output)
        completion_util = completion_call_vllm(prompt=CHAT_TEMPLATE.format(input_prompt_util), model=args.model_name_or_path).choices[0].text
        rating_utility = extract_rating(completion_util)
        
        ## merge labels into text
        sentence_list_with_tags = []
        for i, (group, group_result) in enumerate(zip(groups, group_results)):
            query, document, reasoning = sentences[i*3:(i+1)*3]
            rating_relevance, rating_groundness = group_result["relevance"], group_result["groundness"]
            document = document + rating_relevance
            if reasoning.endswith("\n\n"):
                reasoning = reasoning.replace("\n\n", f"{rating_groundness}\n\n")
            else:
                reasoning = reasoning + "\n\n"
            sentence_list_with_tags.extend([query, document, reasoning])
        sentence_list_with_tags.extend(completion_not_in_groups)
        output_with_tags = "".join(sentence_list_with_tags) + rating_utility
        data.update({"output_with_tags": output_with_tags})
        with writing_lock:
            fcache_obj.write(json.dumps(data, ensure_ascii=False) + "\n")
            fcache_obj.flush()
        return data
    except Exception as e:
        print(e)
        return None

if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Args: {json.dumps(vars(args), indent=4, ensure_ascii=False)}")
    output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    writing_lock = threading.Lock()
    
    fout = os.path.join(output_dir, "results.json")
    fcache_obj = open(os.path.join(output_dir, "cache.jsonl"), "w")
    
    datas = load_train_datas()[args.start:args.end]
    print(f"Data length: {len(datas)}, start: {args.start}, end: {args.end}")

    logger.info(f"Inference function: {vanilla_inference_v2_parse_only}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        #process_func_partial = partial(inference_on_train_set)
        process_func_partial = partial(vanilla_inference_v2_parse_only)
        results = list(tqdm(executor.map(process_func_partial, datas), total=len(datas)))

    results = [{
        "instruction" : item["instruction"],
        "output" : item["output_with_tags"],
        "output_without_tags" : item["output"],
        "meta" : item["meta"]
        } for item in results if item is not None]
    print(f"Original results length: {len(datas)}, filtered results length: {len(results)}")
    with open(fout, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    fcache_obj.close()