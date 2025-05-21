## Naive parse the results

import re
import os
import json
import argparse

import numpy as np

from tqdm import tqdm
from collections import defaultdict, Counter

SQS, SQE, PS, PE = "<sub-question>", "</sub-question>", "<paragraph>", "</paragraph>"
ATOM, NATOM = "[Atomic Question]", "[Non-Atomic Question]"
REL, IRREL = "[Relevant]", "[Irrelevant]"
Q, RQ, A = "[Question]", "[Remaining Question]", "[Final Answer]"

def naive_parsing_bridge(data):
    results = []
    for i in range(len(data["parsed_results"])):
        try:
            results.append(naive_parsing_bridge_single(data, i))
        except Exception as e:
            pass
    return results

def naive_parsing_comparison(data):
    results = []
    for i in range(len(data["parsed_results"])):
        try:
            results.append(naive_parsing_comparison_single(data, i))
        except Exception as e:
            pass
    return results

def naive_parsing_bridge_single(data, idx):
    meta = {
        "dataset" : data["dataset"],
        "type" : data["type"],
        "id" : data["qid"]
    }
    question = data["question_text"]
    metric = {k:v[idx] for k, v in data["metrics"].items()}
    f1 = metric["f1"][0]
    if f1 < 0.8:
        raise ValueError(f"f1 score is too low: {f1}")
    parsed_result = data["parsed_results"][idx]
    supporting_contexts = [item for item in data["context"] if item["is_supporting"]]
    instruction = f"{Q}{question}"
    full_text = f"{NATOM}"
    qa_pairs, final_answer, sub_questions = parsed_result[:-2], parsed_result[-2], parsed_result[-1]
    for i, qa_pair in enumerate(qa_pairs):
        subq = qa_pair["sub_question"]
        subq_type = qa_pair["type"]
        doc_ids = [item["doc_id"] for item in qa_pair["reasoning"]]
        reasonings = [item["sentence"] for item in qa_pair["reasoning"]]
        reasoning_text = " ".join(reasonings)
        reasoning_text =  re.sub(r'Document #\d+', 'Document', reasoning_text)
        if subq_type == "remaining_question":
            if i != len(qa_pairs) - 1: # There are following sub-questions
                full_text += f"{RQ}{subq}{NATOM}"
                assert len(reasonings) == 0
            else:
                doc = supporting_contexts[int(doc_ids[0][0].split("#")[-1]) - 1]
                doc_text = f"{doc['title']}:{doc['text']}"
                full_text += f"{RQ}{subq}{ATOM}{PS}{doc_text}{PE}{reasoning_text}\n\n"
        else: # Sub-question
            doc = supporting_contexts[int(doc_ids[0][0].split("#")[-1]) - 1]
            doc_text = f"{doc['title']}: {doc['text']}"
            full_text += f"{SQS}{subq}{PS}{doc_text}{PE}{reasoning_text}{SQE}\n\n"    
    full_text += f"{A}{final_answer['final_answer']}"
    return {"instruction": instruction, "output": full_text, "meta": meta}

def naive_parsing_comparison_single(data, idx):
    meta = {
        "dataset" : data["dataset"],
        "type" : data["type"],
        "id" : data["qid"]
    }
    question = data["question_text"]
    metric = {k:v[idx] for k, v in data["metrics"].items()}
    f1 = metric["f1"][0]
    if f1 < 0.7:
        raise ValueError(f"f1 score is too low: {f1}")
    parsed_result = data["parsed_results"][idx]
    supporting_contexts = [item for item in data["context"] if item["is_supporting"]]
    instruction = f"{Q}{question}"
    full_text = f"{NATOM}"
    qa_pairs, final_answer, sub_questions = parsed_result[:-2], parsed_result[-2], parsed_result[-1]
    for i, qa_pair in enumerate(qa_pairs):
        subq = qa_pair["sub_question"]
        subq_type = qa_pair["type"]
        doc_ids = [item["doc_id"] for item in qa_pair["reasoning"]]
        reasonings = [item["sentence"] for item in qa_pair["reasoning"]]
        reasoning_text = " ".join(reasonings)
        reasoning_text =  re.sub(r'Document #\d+', 'Document', reasoning_text)
        if subq_type == "remaining_question":
            if reasoning_text == "": #Remaining question in the middle.
                full_text += f"{RQ}{subq}{NATOM}"
            else:
                assert doc_ids[0] == "Unknown" #Generally the final ATOM remaining question.
                full_text += f"{RQ}{subq}{ATOM}{reasoning_text}"
        else:
            doc = supporting_contexts[int(doc_ids[0][0].split("#")[-1]) - 1]
            doc_text = f"{doc['title']}: {doc['text']}"
            full_text += f"{SQS}{subq}{PS}{doc_text}{PE}{reasoning_text}{SQE}\n\n"      
    if not full_text.endswith("\n\n"):
        full_text += "\n\n"
    full_text += f"{A}{final_answer['final_answer']}"
    return {"instruction": instruction, "output": full_text, "meta": meta}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./data_generation/ours_T2/base_data/outputs/deepseek_1215_v2/results.json")
    parser.add_argument("--output", type=str, default="./dataset/ours_naive_v2")
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print(f"Args: {json.dumps(args.__dict__, indent=4)}")
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    else:
        if args.overwrite:
           print(f"Warninig: Overwrite the existing output directory: {args.output}") 
        else:
            raise ValueError(f"Output directory already exists: {args.output}")
    
    # datas = json.load(open("/home/zchu/codes/train_2412/data_generation/ours_T2/base_data/outputs/deepseek/results.json"))
    # datas_wo_bdgcmp = [item for item in datas if item["type"] not in ["bridge_comparison"]]
    # datas_cmp = json.load(open("/home/zchu/codes/train_2412/data_generation/ours_T2/base_data/outputs/deepseek_bridge_comparison/results.json"))
    # datas = datas_wo_bdgcmp + datas_cmp
    datas = json.load(open(args.input))
    ## Parse Completions into structured results
    for data in tqdm(datas):
        parsed_results = []
        for completion in data["completions"]:
            sentences = completion.split("\n")
            sub_questions = []
            qa_pairs = []
            tmp = []
            state = "init"
            for sentence in sentences:
                if "### Decompose" in sentence:
                    state = "decomposition"
                    continue
                elif "### Sub-question" in sentence:
                    state = "subquestion"
                elif "### Final Answer" in sentence:
                    state = "final_answer"
                    continue
                    
                if sentence == "":
                    continue
                if state == "init":
                    pass
                elif state == "decomposition":
                    sub_questions.append(sentence)
                elif state == "subquestion":
                    
                    if sentence.startswith("### Sub-question") or sentence.startswith("### Remaining"):
                        qa_pairs.append(tmp)
                        tmp = []
                    tmp.append(sentence)
                elif state == "final_answer":
                    final_ans = sentence
                else:
                    pass
            qa_pairs.append(tmp)

            sub_questions_parsed = []
            for sub_question in sub_questions:
                match = re.match(pattern=r"\d+\.\s*(.*)", string=sub_question)
                if match:
                    sub_questions_parsed.append(match.groups(1)[0])
                else:
                    continue
                
            qa_pairs_parsed = []
            for qa_pair in qa_pairs:
                if len(qa_pair) == 0:
                    continue
                sentence_question = qa_pair[0]
                sub_question = sentence_question
                subq_type = "subquestion"
                if "Sub-question:" in sentence_question:
                    sub_question = sentence_question.split("Sub-question:")[-1].strip()
                elif "Remaining Question" in sentence_question:
                    subq_type = "remaining_question"
                    sub_question = sentence_question.split("Remaining Question:")[-1].strip()
                else:
                    sub_question = sentence_question
                
                tmp = []
                reasoning_sentences = qa_pair[1:]
                for rs in reasoning_sentences:
                    rs = rs.strip()
                    search_res = re.findall(pattern=r".*?Document\s*(#\d+).*?", string=rs)
                    if len(search_res) == 0:
                        doc_id = "Unknown"
                    else:
                        doc_id = search_res
                    tmp.append({"doc_id": doc_id, "sentence": rs})
                qa_pairs_parsed.append({"sub_question": sub_question, "type": subq_type, "reasoning": tmp})
            qa_pairs_parsed.append({"final_answer": final_ans})
            qa_pairs_parsed.append({"sub_questions": sub_questions_parsed})
            parsed_results.append(qa_pairs_parsed)
        data["parsed_results"] = parsed_results
    
    
    import random
    parsed_train_format_results = []
    S, F = 0, 0
    for data in tqdm(datas):
        qtype = data["type"]
        if qtype in ["comparison", "bridge_comparison"]:
            results = naive_parsing_comparison(data)
        else:
            results = naive_parsing_bridge(data)
        if len(results) >= 1:
            parsed_train_format_results.extend(random.choices(results, k=1))
            S += 1
        else:
            F += 1
        
        
    print(f"Num: {len(parsed_train_format_results)}, Coverage: {(S/(S+F))*100:.2f}%, Diversity: {(len(parsed_train_format_results)/len(datas)):.2f}", end="")
    
    
    with open(f"{args.output}/train.json", "w") as f:
        json.dump(parsed_train_format_results, f, indent=4, ensure_ascii=False)
    with open(f"{args.output}/train.jsonl", "w") as f:
        for line in parsed_train_format_results:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    with open(f"{args.output}/train.preview.json", "w") as f:
        from collections import defaultdict
        preview_datas = defaultdict(list)
        for data in parsed_train_format_results:
            preview_datas[data["meta"]["type"]].append(data)
        preview_datas = {k:v[:100] for k, v in preview_datas.items()}
        json.dump(preview_datas, f, indent=4, ensure_ascii=False)

# Example running commands
# python -m data_generation.ours_T2.base_data.naive_parse --output "./dataset/ours_naive_v3_1226" --overwrite