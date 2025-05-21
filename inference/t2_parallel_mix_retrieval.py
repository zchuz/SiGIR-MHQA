# conda environment: vllm
# create date: 2025/02/04. Support mix retrieval.
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
sys.path.append("/home/zchu/codes/train_2412")

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
from data_generation.utils import completion_call_vllm, completion_call_vllm_with_dynamic_port, chat_completion_call_openai_msg
from data_generation.utils import em_score, f1_score, pm_score
from data_generation.utils import BM25_retrieval, dense_retrieval, mix_retrieval
from train.utils import PS, PE, resize_token_embeddings_with_init, is_bf16_supported
from safetensors.torch import load_file


logger = get_logger(__file__)

CHAT_TEMPLATE = "[INST] {} [/INST]"

RETRIEVER_FUNC = {
    "sparse" : BM25_retrieval,
    "dense" : dense_retrieval,
    "mixed" : mix_retrieval
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/home/share/models/Mistral-7B-v0.2")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, choices=["2wikimqa", "hotpotqa", "musique", "bamboogle"], nargs="+", default=["2wikimqa", "hotpotqa", "musique", "bamboogle"])
    parser.add_argument("--num_datas", type=int, default=100, help="Number of test datas for each dataset.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Wether to overwrite the existing merged model")
    parser.add_argument("--train_set", action="store_true", default=False, help="Use train set for inference")
    parser.add_argument("--test_set", action="store_true", default=False, help="Use test set for inference")
    parser.add_argument("--fin", type=str, default=None, nargs="+")
    parser.add_argument("--start", type=int, default=0, help="Start data index for inference")
    parser.add_argument("--end", type=int, default=None, help="End data index for inference")
    parser.add_argument("--max_workers", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.15)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--inference_func", type=str, choices=["guided", "greedy", "vanilla", "parse_only", "greedy_with_critic"], default=None, required=True)
    parser.add_argument("--retriever", type=str, choices=["sparse", "dense", "mixed"], default="sparse")
    args = parser.parse_args()
    return args

def extract_answer(text):
    pattern = r"@@(.*?)@@"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return "Unknown"

def load_datas():
    if args.fin is None:
        fin = "dataset/test/test.jsonl"
    else:
        raise ValueError("fin must be a list of files")
    logger.info(f"Loading Test data from {fin}")
    all_test_datas = [json.loads(line) for line in open(fin)]
    loaded_datas = []
    datasets = args.dataset
    for dataset in datasets:
        dataset_datas = [item for item in all_test_datas if item["dataset"] == dataset]
        loaded_datas.extend(dataset_datas[:args.num_datas])
    return loaded_datas
        
def load_train_datas():
    if args.fin is None:
        datas_2wiki = json.load(open("./datas_for_acl25_mhqa/2wiki/train.20k.json"))[args.start:args.end]
        datas_hotpot = json.load(open("./datas_for_acl25_mhqa/hotpotqa/train.10k.json"))[args.start:args.end]
        datas_music = json.load(open("./datas_for_acl25_mhqa/musique/train.19k.json"))[args.start:args.end]
        datas = datas_2wiki + datas_hotpot + datas_music
        logger.info(f"Loading Training datas for Inference from ./datas_for_acl25_mhqa")
        return datas
    else:
        all_datas = []
        for fi in args.fin:
            datas = json.load(open(fi))
            all_datas.extend(datas)
            logger.info(f"Loading Training datas for Inference from {fi}")
        logger.info(f"Totally {len(all_datas)} datas for Inference.")
        return all_datas
    

def load_lora_model(model_name_or_path, adapter_path=None):
    if adapter_path is None: # Only Load Base Model
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return LLM(
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            dtype=torch.bfloat16 if is_bf16_supported() else torch.float16,
            max_model_len=3192
        ), tokenizer
        
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    merge_model_save_path = os.path.join(".cache", os.path.split(model_name_or_path)[-1], "-".join(adapter_path.split(os.sep)[-2:]))
    if os.path.exists(merge_model_save_path) and not args.overwrite:
        #raise FileExistsError(f"Merged model already exists at {merge_model_save_path}")
        logger.warning(f"Merged model already exists at {merge_model_save_path}. Use the existing model.")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        resize_token_embeddings_with_init(model, tokenizer)
        ## Loading embeddings
        embeddings_tensors = load_file(os.path.join(adapter_path, "embeddings.safetensors"))
        model.load_state_dict(embeddings_tensors, strict=False)
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        model.save_pretrained(merge_model_save_path)
        logger.info(f"Merged model saved to {merge_model_save_path}")
    
    llm = LLM(
        model=merge_model_save_path,
        tokenizer=adapter_path if adapter_path is not None else model_name_or_path,
        dtype=torch.bfloat16 if is_bf16_supported() else torch.float16,
        max_model_len=3192
    )
    return llm, tokenizer
    

def extract_query(completion: str):
    if "[Non-Atomic Question]" in completion:
        query_string = completion.split("[Non-Atomic Question]")[1]
    else:
        query_string = completion
    query = query_string.replace("<paragraph>", "").replace("[Non-Atomic Question]", "").replace("[Atomic Question]", "").replace("</paragraph>", "").replace("<sub-question>", "").replace("/<sub-question>", "").replace("[Remaining Question]", "").strip()
    return query


QUERY_REWRITE_INSTRUCTION = """Please rewrite the following search query without altering its original meaning, ensure that the revised query is diverse, clear, and consistent with the original query.\nWhen rewriting, consider using synonyms, adjusting word order, or changing the query structures. Do not change the core entity of the query.\n\n### Question: {}\n### Rewritten:"""

def extract_rating(text):
    text = text.strip("</s>").strip("<|im_end|>")
    if text.endswith("[4]"):
        return 4
    elif text.endswith("[3]"):
        return 3
    elif text.endswith("[2]"):
        return 2
    elif text.endswith("[1]"):
        return 1
    elif text.endswith("[0]"):
        return 0
    else:
        return 0

def convert_to_score(text):
    if text.endswith("[Fully supported]"):
        return 0.5
    if text.endswith("[Partially supported]"):
        return 0.25
    if text.endswith("[Not supported]"):
        return -0.5
    
    if text == "[Relevant]":
        return 1.0
    if text == "[Irrelevant]":
        return -1.0
    if text == "[Partially Relevant]":
        return 0.5
    return 0.0
def deduplicate(texts):
    return list(set(texts))

def get_prefix_text(prefix_dict: dict, pid: str):
    text = ""
    for i in range(len(pid)):
        pid_prefix = pid[:i+1]
        text += prefix_dict[pid_prefix]
    return text

from dataclasses import dataclass


def guided_inference_v2(data):
    # if data["type"] != "bridge_comparison":
    #     return []
    CANDIDATE_SIZE = 2  # Number of retained candidates each round
    # Generally, only one of (SUBQUESTION_SIZE, REWRITE_QUERY_SIZE) is set more than 1.
    SUBQUESTION_SIZE = 2 # Number of decomposed subquestions per decomposition step
    REWRITE_QUERY_SIZE = 1 # Number of rewritten queries per decomposition step.
    EXPLORATION_SIZE = 6 # Number of retrieval documents per retrieval step
    REASONING_SIZE = 2 # Number of knowledge reasoning per round.
    question = data["question_text"]
    answer = data["answer"]
    prediction = "FAILED"
    input_prompt = CHAT_TEMPLATE.format(f"[Question]{question}")
    candidates = [input_prompt]
    candidates_scores = [0.0] * CANDIDATE_SIZE
    query_list, completion_list, reasoning_list, document_list, document_relevance_list = [], [], [], [], []
    
    completion_call_rewrite = partial(chat_completion_call_openai_msg, model="gpt-4o-mini", temperature=args.temperature, n=REWRITE_QUERY_SIZE)
    completion_call_decompose = partial(completion_call_vllm, model=args.model_name_or_path, stop=["\n\n"] + ["<paragraph>"] + ["</s>", "<|im_end|>"])
    completion_call_relevance = partial(completion_call_vllm, model=args.model_name_or_path, stop=["\n\n"] + ["[Relevant]", "[Irrelevant]", "[Partially Relevant]"] + ["</s>", "<|im_end|>"])    
    completion_call_groundness = partial(completion_call_vllm, model=args.model_name_or_path, stop=["\n\n"] + ["[Fully supported]", "[Partially supported]", "[Not supported]"] + ["</s>", "<|im_end|>"])
    completion_call_final = partial(completion_call_vllm, model=args.model_name_or_path, stop=["\n\n"] + ["[4]", "[3]", "[2]", "[1]", "[0]"] + ["</s>", "<|im_end|>"])
    try:
        for round_index in range(9):
            prefix_dict = {f"{i}" : item for i, item in enumerate(candidates)}
            prefix_dict_scores = {f"{i}" : 0.0 for i, item in enumerate(candidates)}
            
            # Generally, this is decomposition round.
            completions_list = [completion_call_decompose(prompt=item, temperature=args.temperature, n=SUBQUESTION_SIZE) for item in candidates]
            completions_list = [[item.text for item in completions.choices] for completions in completions_list]
            completions_flatten = list(chain(*completions_list)) # Shape: CANDIDATE_SIZE * SUB_QUESTION_SIZE
            
            for i in range(len(candidates)):
                for j in range(SUBQUESTION_SIZE):
                    completion = completions_list[i][j]
                    prefix_dict[f"{i}{j}"] = completion
            
            # In the case the last round of comparison type question. We add additional reasoning step.
            if any(["[Atomic Question]" in completion and "<paragraph>" not in completion for completion in completions_flatten]):
                final_input_prompts = []
                for i in range(CANDIDATE_SIZE):
                    for j in range(SUBQUESTION_SIZE):
                        final_input_prompts.append(candidates[i] + completions_list[i][j])
                
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(completions_flatten)) as executor:
                    final_completions = list(executor.map(completion_call_final, final_input_prompts))
                # Shape: CANDIDATE_SIZE * SUBQUESTION_SIZE
                final_completions = [item.choices[0].text.strip() for item in final_completions]
                ret_list = []
                for i in range(CANDIDATE_SIZE):
                    for j in range(SUBQUESTION_SIZE):
                        ret_list.append({
                            "reasoning": candidates[i] + completions_list[i][j] + final_completions[i*SUBQUESTION_SIZE+j],
                            "answer": final_completions[i*SUBQUESTION_SIZE+j],
                            "scores": candidates_scores[i],
                            "metrics" : {
                                "em": em_score(extract_answer(final_completions[i*SUBQUESTION_SIZE+j]), answer),
                                "f1": f1_score(extract_answer(final_completions[i*SUBQUESTION_SIZE+j]), answer)[0],
                                "pm": pm_score(extract_answer(final_completions[i*SUBQUESTION_SIZE+j]), answer),
                            }
                        })
                        
                return_dict = {
                    "qid": data["qid"],
                    "question": question,
                    "answer": answer,
                    "reasoning_list": ret_list,
                    "flag": True,
                    "dataset": data["dataset"],
                    "type": data["type"],
                }
                with writing_lock:
                    fcache_obj.write(json.dumps(return_dict, ensure_ascii=False) + "\n")
                    fcache_obj.flush()
                return return_dict
            
            # The last step generates the final answer.
            if any(["[Final Answer]" in completion for completion in completions_flatten]):
                ret_list = []
                for i in range(CANDIDATE_SIZE):
                    for j in range(SUBQUESTION_SIZE):
                        ret_list.append({
                            "reasoning": candidates[i] + completions_list[i][j],
                            "answer": completions_list[i][j],
                            "scores": candidates_scores[i],
                            "metrics" : {
                                "em": em_score(extract_answer(completions_list[i][j]), answer),
                                "f1": f1_score(extract_answer(completions_list[i][j]), answer)[0],
                                "pm": pm_score(extract_answer(completions_list[i][j]), answer),
                            }
                        })
                return_dict = {
                    "qid": data["qid"],
                    "question": question,
                    "answer": answer,
                    "reasoning_list": ret_list,
                    "flag": True,
                    "dataset": data["dataset"],
                    "type": data["type"],
                }
                with writing_lock:
                    fcache_obj.write(json.dumps(return_dict, ensure_ascii=False) + "\n")
                    fcache_obj.flush()
                return return_dict
                
            
            queries = [extract_query(item) for item in completions_flatten] # Shape: CANDIDATE_SIZE * SUBQUESTION_SIZE

            # Retrieval round. For each query, retrieval EXPLORATION_SIZE documents.
            documents = [retrieval_func(query, dataset=data["dataset"], topk=EXPLORATION_SIZE) for query in queries]
            # Shape: CANDIDATE_SIZE * SUBQUESTION_SIZE * EXPLORATION_SIZE
            documents_text = [f"{item['title']}: {item['paragraph_text']}</paragraph>" for retrieval_contents in documents for item in retrieval_contents] 
            
            # Shape: CANDIDATE_SIZE * SUBQUESTION_SIZE * EXPLORATION_SIZE
            prefix_ids = [] 
            for i in range(len(candidates)):
                for j in range(SUBQUESTION_SIZE):
                    for k in range(EXPLORATION_SIZE):
                        prefix_dict[f"{i}{j}{k}"] = documents_text[i*SUBQUESTION_SIZE*EXPLORATION_SIZE+j*EXPLORATION_SIZE+k]
                        prefix_ids.append(f"{i}{j}{k}")
            
            # Shape: CANDIDATE_SIZE * SUBQUESTION_SIZE * EXPLORATION_SIZE
            all_prefix_texts = [get_prefix_text(prefix_dict, pid) for pid in prefix_ids]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=EXPLORATION_SIZE*SUBQUESTION_SIZE) as executor:
                relevance_completions = list(executor.map(completion_call_relevance, all_prefix_texts))
                relevance_completions = [item.choices[0].text.strip() for item in relevance_completions]
                relevance_scores = [convert_to_score(item) for item in relevance_completions]
            
            for i, pid in enumerate(prefix_ids):
                prefix_dict[pid] = prefix_dict[pid] + relevance_completions[i]
                if pid not in prefix_dict_scores:
                    prefix_dict_scores[pid] = relevance_scores[i]
                else:
                    prefix_dict_scores[pid] += relevance_scores[i]
            
            all_prefix_texts = [get_prefix_text(prefix_dict, pid) for pid in prefix_ids]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=EXPLORATION_SIZE*SUBQUESTION_SIZE) as executor:
                reasoning_completions = list(executor.map(completion_call_groundness, all_prefix_texts))
                reasoning_completions = [item.choices[0].text.strip() for item in reasoning_completions]
                reasoning_scores = [convert_to_score(item) for item in reasoning_completions]
            for i, pid in enumerate(prefix_ids):
                prefix_dict[pid] = prefix_dict[pid] + reasoning_completions[i]
                if pid not in prefix_dict_scores:
                    prefix_dict_scores[pid] = reasoning_scores[i]
                else:
                    prefix_dict_scores[pid] += reasoning_scores[i]        
            
            # Shape: CANDIDATE_SIZE * SUBQUESTION_SIZE * EXPLORATION_SIZE
            all_prefix_scores = {pid: prefix_dict_scores[pid] for pid in prefix_ids}
            
            sorted_prefix = sorted(all_prefix_scores.items(), key=lambda x: x[1], reverse=True)
            candidates_pid = [pid for pid, score in sorted_prefix[:CANDIDATE_SIZE]]
            
            candidates = [get_prefix_text(prefix_dict, pid) + "\n\n" for pid in candidates_pid]
            
            new_candidates_scores = []
            for pid, score in sorted_prefix[:CANDIDATE_SIZE]:
                prev_score = candidates_scores[int(pid[0])]
                new_score = prev_score + score
                new_candidates_scores.append(new_score)
            candidates_scores = new_candidates_scores
    except Exception as e:
        print(e)
        return {
            "qid": data["qid"],
            "question": question,
            "answer": answer,
            "reasoning_list": [],
            "flag": False,
            "dataset": data["dataset"],
            "type": data["type"],
        }

        # for candidate, score in zip(candidates, candidates_scores):
        #     print(f"Score: {score}, Candidate: {candidate}")

        #print(1)
        


# Greedy infernce for critic trained generator model
def greedy_inference(data):
    question = data["question_text"]
    answer = data["answer"]
    prediction = "FAILED"
    input_prompt = CHAT_TEMPLATE.format(f"[Question]{question}")
    query_list, completion_list, reasoning_list, document_list, document_relevance_list = [], [], [], [], []
    try:
        for i in range(15):
            completion = completion_call_vllm(
                prompt=input_prompt,
                model=args.model_name_or_path,
                stop=["\n\n", "<paragraph>"]
            ).choices[0].text.replace(" \n\n", "\n\n").replace("  ", " ").replace("[Relevant] ", "[Relevant]").replace("[Irrelevant] ", "[Irrelevant]").replace("[Partially Relevant] ", "[Partially Relevant]")
            if "[Final Answer]" in completion:
                prediction = extract_answer(completion)
                input_prompt = input_prompt + completion
                break

            # i % 2 == 0: Decomposition Round. It includes twice completion call.
            # 1. Generate a new query
            # 2. Retrieve multiple documents, and update input_prompts.
            # 3. Let model to judge if the retrieval content is relevant to the query.
            # 4. Keep the most relevant retrieval content and update input_prompt.
            if i % 2 == 0:
                if "<paragraph>" in completion: # Do retrieval
                    query = extract_query(completion)
                    retrieval_contents = retrieval_func(query, dataset=data["dataset"], topk=5)
                    documents = [f"{item['title']}: {item['paragraph_text']}" for item in retrieval_contents]
                    relevance_input_prompts = [
                        f"{input_prompt}{completion}{doc}</paragraph>"
                        for doc in documents
                    ]
                    relevance_completions = []
                    for rip in relevance_input_prompts:
                        relevance_completion = completion_call_vllm(
                            prompt=rip,
                            model=args.model_name_or_path,
                            stop=["\n\n"] + ["[Relevant]", "[Irrelevant]", "[Partially Relevant]"]
                        ).choices[0].text
                        relevance_completions.append(relevance_completion.strip())
                    if all([item == "[Irrelevant]" for item in relevance_completions]): # When the model cannot make a judgment, we trust the retriever's scoring.
                        select_index = 0
                    else:
                        sorted_with_indices = sorted(enumerate(relevance_completions), key=lambda x: x[1], reverse=True)    
                        select_index = sorted_with_indices[0][0]
                    input_prompt = relevance_input_prompts[select_index] + relevance_completions[select_index]
                    query_list.append(query)
                    completion_list.append(completion)
                    document_list.append(retrieval_contents[select_index])
                    document_relevance_list.append(relevance_completions[select_index])
                else: # Last step of comparison question
                    input_prompt = input_prompt + completion
                    completion_list.append(completion)
                    reasoning_list.append(completion)
            # i % 2 == 1: Reasoning Round.
            # 1. Generate a new completion.
            # 2. Use supportive tags to guided search. (This is not applicable for greedy inference)
            else:
                input_prompt = input_prompt + completion
                completion_list.append(completion)
                reasoning_list.append(completion)
        return_dict = {
            "qid": data["qid"],
            "question": question,
            "reasoning": input_prompt,
            "prediction": prediction,
            "answer": answer,
            "query_list": query_list,
            "completion_list": completion_list,
            "reasoning_list": reasoning_list,
            "documents": document_list,
            "documents_relevance": document_relevance_list,
            "flag": True,
            "dataset": data["dataset"],
            "type": data["type"],
            "metrics": {
                "em": em_score(prediction, answer),
                "f1": f1_score(prediction, answer)[0],
                "pm": pm_score(prediction, answer),
            }
        }
        with writing_lock:
            fcache_obj.write(json.dumps(return_dict, ensure_ascii=False) + "\n")
            fcache_obj.flush()
        return return_dict
            
    except Exception as e:
        print(e)
        return_dict = {
            "qid": data["qid"],
            "question": question,
            "reasoning": input_prompt,
            "prediction": prediction,
            "answer": answer,
            "query_list": query_list,
            "completion_list": completion_list,
            "reasoning_list": reasoning_list,
            "documents": document_list,
            "documents_relevance": document_relevance_list,
            "flag": False,
            "dataset": data["dataset"],
            "type": data["type"],
            "metrics": {
                "em": 0,
                "f1": 0,
                "pm": 0,
            }
        }
        with writing_lock:
            fcache_obj.write(json.dumps(return_dict, ensure_ascii=False) + "\n")
            fcache_obj.flush()
        return return_dict



# Use the seperated critic model to assign reward labels to the reasoning process.
def greedy_inference_with_critic(data):
    raise NotImplementedError("Please refer to inference.t2_with_critic to run inference with seperated critic model.")

# Only Parse, do not inference.
# Use the trained critic model to assign reward labels to the reasoning process generated by the Teacher model (generally deepseek-chat-v2.5).
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
            return
        
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
            completion = completion.strip("</s>").strip("<|im_end|>")
            explanation, rating = completion.split("\n")
            rating = rating.split("Rating: ")[-1].strip()
            return rating
        
        group_results = []
        inst_gnd = open("data_generation/ours_T2/critic/groundness/prompts/v2/instruction.txt").read()
        inst_rel = open("data_generation/ours_T2/critic/relevance/prompts/v2/instruction.txt").read()
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
    

def vanilla_inference_v2(data):
    
    question = data["question_text"]
    answer = data["answer"]
    prediction = "FAILED"
    input_prompt = CHAT_TEMPLATE.format(f"[Question]{question}")
    query_list, completion_list, reasoning_list, document_list = [], [], [], []
    try:
        for i in range(15):
            completion = completion_call_vllm(
                prompt=input_prompt,
                model=args.model_name_or_path,
                stop=["\n\n", "<paragraph>"]
            ).choices[0].text.replace(" \n\n", "\n\n").replace("  ", " ").replace("[Relevant] ", "[Relevant]").replace("[Irrelevant] ", "[Irrelevant]").replace("[Partially Relevant] ", "[Partially Relevant]")
            if "[Final Answer]" in completion:
                prediction = extract_answer(completion)
                input_prompt = input_prompt + completion
                completion_list.append(completion)
                break
            
            if i % 2 == 0:
                if "<paragraph>" in completion: # Do retrieval
                    query = extract_query(completion)
                    retrieval_content = retrieval_func(query, dataset=data["dataset"], topk=1)[0]
                    document = f"{retrieval_content['title']}: {retrieval_content['paragraph_text']}"
                    input_prompt = input_prompt + completion + document + "</paragraph>"
                    query_list.append(query)
                    completion_list.append(completion)
                    document_list.append(retrieval_content)
                else: # Last step of comparison type question, no need for retrieval
                    input_prompt = input_prompt + completion
                    completion_list.append(completion)
                    reasoning_list.append(completion)
            else:
                input_prompt = input_prompt + completion
                completion_list.append(completion)
                reasoning_list.append(completion)
        return_dict = {
            "qid": data["qid"],
            "question": question,
            "reasoning": input_prompt,
            "prediction": prediction,
            "answer": answer,
            "query_list": query_list,
            "completion_list": completion_list,
            "reasoning_list": reasoning_list,
            "documents": document_list,
            "flag": True,
            "dataset": data["dataset"],
            "type": data["type"],
            "metrics": {
                "em": em_score(prediction, answer),
                "f1": f1_score(prediction, answer)[0],
                "pm": pm_score(prediction, answer),
            }
        }   
        with writing_lock:
            fcache_obj.write(json.dumps(return_dict, ensure_ascii=False) + "\n")
            fcache_obj.flush()
        return return_dict 
                
    except Exception as e:
        print(e)
        return_dict = {
            "qid": data["qid"],
            "question": question,
            "reasoning": input_prompt,
            "prediction": prediction,
            "answer": answer,
            "query_list": query_list,
            "completion_list": completion_list,
            "reasoning_list": reasoning_list,
            "documents": document_list,
            "flag": False,
            "dataset": data["dataset"],
            "type": data["type"],
            "metrics": {
                "em": 0,
                "f1": 0,
                "pm": 0,
            }
        }
        with writing_lock:
            fcache_obj.write(json.dumps(return_dict, ensure_ascii=False) + "\n")
            fcache_obj.flush()
        return return_dict


def inference_on_train_set(data):
    input_prompt, documents, completion_list, query_list, answer = "", [], [], [], data["answer"]
    try:
        stop_token_ids=[tokenizer.encode("<paragraph>")[-1]]
        stop=["\n\n"]
        
        question = f"[Question]{data['question_text']}"
        dataset = data["dataset"]
        answer = data["answer"]
        input_prompt = CHAT_TEMPLATE.format(question)
        completion = completion_call_vllm(prompt=input_prompt, model=args.model_name_or_path, n=1, temperature=0.0, top_p=1.0, max_tokens=150, stop=stop, stop_token_ids=stop_token_ids).choices[0].text
        query = completion.replace("<paragraph>", "").replace("[Non-Atomic Question]", "").replace("[Atomic Question]", "").replace("</paragraph>", "").replace("<sub-question>", "").replace("/<sub-question>", "").strip()

        retrieval_content = retrieval_func(query, dataset=dataset, topk=3)[0]
        retrieval_context = f"{retrieval_content['title']}: {retrieval_content['paragraph_text']}"
        documents = [retrieval_content]
        input_prompt = input_prompt + "" + completion + retrieval_context + "</paragraph>"
        completion_list = [completion]
        query_list = [query]
        for i in range(10):
            completion = completion_call_vllm(prompt=input_prompt, model=args.model_name_or_path, n=1, temperature=0.0, top_p=1.0, max_tokens=150, stop=stop, stop_token_ids=stop_token_ids).choices[0].text
            completion_list.append(completion)
            if "[Final Answer]" in completion:
                input_prompt = input_prompt + "" + completion
                break
            if i % 2 == 0: # Reasoning Round
                input_prompt = input_prompt + "" + completion
            else: # Retrieval
                if "[Non-Atomic Question]" in completion:
                    query_string = completion.split("[Non-Atomic Question]")[1]
                else:
                    query_string = completion
                query = query_string.replace("<paragraph>", "").replace("[Non-Atomic Question]", "").replace("[Atomic Question]", "").replace("</paragraph>", "").replace("<sub-question>", "").replace("/<sub-question>", "").replace("[Remaining Question]", "").strip()
                query_list.append(query)
                retrieval_content = retrieval_func(query, dataset=dataset, topk=3)[0]
                retrieval_context = f"{retrieval_content['title']}: {retrieval_content['paragraph_text']}"
                documents.append(retrieval_content)
                input_prompt = input_prompt + "" + completion + retrieval_context + "</paragraph>"
        return_dict = {
            "reasoning" : input_prompt,
            "answer" : answer,
            "query_list" : query_list,
            "completion_list" : completion_list,
            "documents" : documents,
            "flag" : True,
            "dataset" : data["dataset"],
            "type" : data["type"]
        }
        with writing_lock:
            fcache_obj.write(json.dumps(return_dict, ensure_ascii=False) + "\n")
            fcache_obj.flush()
        return return_dict
    except Exception as e:
        print(e)
        return {
            "reasoning" : input_prompt,
            "answer" : answer,
            "query_list" : query_list,
            "completion_list" : completion_list,
            "documents" : documents,
            "flag" : False,
            "dataset" : data["dataset"],
            "type" : data["type"]
        }
INFERENCE_FUNC_DICT = {
    "greedy": greedy_inference,
    "vanilla": vanilla_inference_v2,
    "guided": guided_inference_v2,
    "parse_only": vanilla_inference_v2_parse_only,
    "greedy_with_critic": greedy_inference_with_critic
}    

if __name__ == "__main__":
    args = parse_args()
    if args.output is None:
        output_dir = f"./inference/outputs/t2_train/{args.start}_{args.end}"
    else:
        output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    writing_lock = threading.Lock()
    fout = os.path.join(output_dir, "results.json")
    fcache_obj = open(os.path.join(output_dir, "cache.jsonl"), "w")
    
    if args.train_set:
        datas = load_train_datas()
    elif args.test_set:
        datas = load_datas()
    else:
        datas = [json.loads(line) for line in open("./dataset/test/test.preview.jsonl")]
    
    print(f"Data length: {len(datas)}, start: {args.start}, end: {args.end}")
        

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    logger.info(f"Sampling params: {vars(args)}")
    
    # for data in tqdm(datas):
    #     greedy_inference(data)
    
    inference_func = INFERENCE_FUNC_DICT[args.inference_func]
    logger.info(f"Inference function: {inference_func}")
    
    retrieval_func = RETRIEVER_FUNC[args.retriever]
    logger.info(f"Retriever: {args.retriever} --- {retrieval_func}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        process_func_partial = partial(inference_func)
        results = list(tqdm(executor.map(process_func_partial, datas), total=len(datas)))
    
    if args.inference_func not in ["guided"]:
        metrics = {}
        for dataset in args.dataset:
            dataset_results = [item for item in results if item["dataset"] == dataset]
            avg_f1 = np.mean([item["metrics"]["f1"] for item in dataset_results])
            avg_em = np.mean([item["metrics"]["em"] for item in dataset_results])
            avg_pm = np.mean([item["metrics"]["pm"] for item in dataset_results])
            metrics[dataset] = {
                "avg_f1": avg_f1,
                "avg_em": avg_em,
                "avg_pm": avg_pm
            }
            print(f"Dataset: {dataset}, Avg F1: {avg_f1}, Avg EM: {avg_em}, Avg PM: {avg_pm}")
    else:
        metrics = {
            "first" : {},
            "max" : {},
            "mean" : {},
            "sorted_by_score": {},
            "sorted_by_rating": {},
            
        }
        for dataset in args.dataset:
            dataset_results = [item for item in results if item["dataset"] == dataset]
            dataset_results = [item for item in dataset_results if item["flag"]]
            # First results
            avg_f1 = np.mean([item["reasoning_list"][0]["metrics"]["f1"] for item in dataset_results])
            avg_em = np.mean([item["reasoning_list"][0]["metrics"]["em"] for item in dataset_results])
            avg_pm = np.mean([item["reasoning_list"][0]["metrics"]["pm"] for item in dataset_results])
            metrics["first"][dataset] = {
                "avg_f1": avg_f1,
                "avg_em": avg_em,
                "avg_pm": avg_pm
            }
            print(f"Eval Mode: First, Dataset: {dataset}, Avg F1: {avg_f1}, Avg EM: {avg_em}, Avg PM: {avg_pm}")
    
            # Max results
            max_f1 = np.mean([np.max([jtem["metrics"]["f1"] for jtem in item["reasoning_list"]]) for item in dataset_results])
            max_em = np.mean([np.max([jtem["metrics"]["em"] for jtem in item["reasoning_list"]]) for item in dataset_results])
            max_pm = np.mean([np.max([jtem["metrics"]["pm"] for jtem in item["reasoning_list"]]) for item in dataset_results])
            metrics["max"][dataset] = {
                "avg_f1": max_f1,
                "avg_em": max_em,
                "avg_pm": max_pm
            }
            print(f"Eval Mode: Max, Dataset: {dataset}, Avg F1: {max_f1}, Avg EM: {max_em}, Avg PM: {max_pm}")

            # Mean results
            mean_f1 = np.mean([np.mean([jtem["metrics"]["f1"] for jtem in item["reasoning_list"]]) for item in dataset_results])
            mean_em = np.mean([np.mean([jtem["metrics"]["em"] for jtem in item["reasoning_list"]]) for item in dataset_results])
            mean_pm = np.mean([np.mean([jtem["metrics"]["pm"] for jtem in item["reasoning_list"]]) for item in dataset_results])
            metrics["mean"][dataset] = {
                "avg_f1": mean_f1,
                "avg_em": mean_em,
                "avg_pm": mean_pm
            }
            print(f"Eval Mode: Mean, Dataset: {dataset}, Avg F1: {mean_f1}, Avg EM: {mean_em}, Avg PM: {mean_pm}")
            
            # Sorted by score
            sorted_by_score_f1 = np.mean([item[0]["metrics"]["f1"] for item in [sorted(item["reasoning_list"], key=lambda x: x["scores"], reverse=True) for item in dataset_results]])
            sorted_by_score_em = np.mean([item[0]["metrics"]["em"] for item in [sorted(item["reasoning_list"], key=lambda x: x["scores"], reverse=True) for item in dataset_results]])
            sorted_by_score_pm = np.mean([item[0]["metrics"]["pm"] for item in [sorted(item["reasoning_list"], key=lambda x: x["scores"], reverse=True) for item in dataset_results]])
            metrics["sorted_by_score"][dataset] = {
                "avg_f1": sorted_by_score_f1,
                "avg_em": sorted_by_score_em,
                "avg_pm": sorted_by_score_pm
            }
            print(f"Eval Mode: Sorted by score, Dataset: {dataset}, Avg F1: {sorted_by_score_f1}, Avg EM: {sorted_by_score_em}, Avg PM: {sorted_by_score_pm}")
            
            # Sorted by rating
            sorted_by_rating_f1 = np.mean([item[0]["metrics"]["f1"] for item in [sorted(item["reasoning_list"], key=lambda x: extract_rating(x["reasoning"]), reverse=True) for item in dataset_results]])
            sorted_by_rating_em = np.mean([item[0]["metrics"]["em"] for item in [sorted(item["reasoning_list"], key=lambda x: extract_rating(x["reasoning"]), reverse=True) for item in dataset_results]])
            sorted_by_rating_pm = np.mean([item[0]["metrics"]["pm"] for item in [sorted(item["reasoning_list"], key=lambda x: extract_rating(x["reasoning"]), reverse=True) for item in dataset_results]])
            metrics["sorted_by_rating"][dataset] = {
                "avg_f1": sorted_by_rating_f1,
                "avg_em": sorted_by_rating_em,
                "avg_pm": sorted_by_rating_pm
            }
            print(f"Eval Mode: Sorted by rating, Dataset: {dataset}, Avg F1: {sorted_by_rating_f1}, Avg EM: {sorted_by_rating_em}, Avg PM: {sorted_by_rating_pm}")
            
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    with open(fout, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    fcache_obj.close()


# First should launch vllm server:
# python -m vllm.entrypoints.openai.api_server --model /home/zchu/codes/train_2412/.cache/Mistral-7B-v0.2/mistral_with_critic_ep2/merged_checkpoint/ --dtype auto --api-key abc --disable-log-requests --port 8090 --gpu-memory-utilization 0.9

# Then start reasoning
# python -m inference.t2_parallel --model_name_or_path ./train/outputs/t2_naive/mistral_2wikimqa_t2_naive_v3_bs64_lr1e-4_ep2_1226-2200-NjW/merged_checkpoint --dataset 2wikimqa hotpotqa musique --num_datas 500 --test_set --max_workers 300 --output train/scripts/generator_1226/naive_2wiki 
