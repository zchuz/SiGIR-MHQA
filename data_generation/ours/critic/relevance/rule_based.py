import json
import random
import argparse
import concurrent.futures
from tqdm import tqdm
from data_generation.utils import chat_completion_call_deepseek_msg
from itertools import chain
FPS = {
    "2wikimqa" : "/home/zchu/datasets/mhqa/raw_data/2wikimqa/train.json",
    
}

triplet_to_query_prompt = "I will give you the subject and predicate of a triple, and you need to transform it into a natural language question. Here is an example:\nSubject: Stuart Rosenberg\nPredicate: country of citizenship\nQuestion: Where did Stuart Rosenberg born?\n\nSubject: {}\nPredicate: {}\nQuestion:"

def process_musique(data):
    def replace_placeholders(sub_question, sub_ans_list):
        if "#1" in sub_question:
            sub_question = sub_question.replace("#1", sub_ans_list[0])
        if "#2" in sub_question:
            sub_question = sub_question.replace("#2", sub_ans_list[1])
        if "#3" in sub_question:
            sub_question = sub_question.replace("#3", sub_ans_list[2])
        return sub_question
    
    def rewrite_subquestion(sub_question):
        if ">>" in sub_question:
            subject, predicate = [item.strip() for item in sub_question.split(">>")]
            prompt = triplet_to_query_prompt.format(subject, predicate)
            messages = [{"role" : "user", "content" : prompt}]
            response = chat_completion_call_deepseek_msg(messages, max_tokens=100, temperature=1.3)[0].strip("\"\'").replace("*","")
            return response
        else:
            return sub_question
    
    try:
        qid = data["id"]
        context = data["paragraphs"]
        question_decomposition = data["question_decomposition"]
        query_doc_list = []
        sub_ans_list = [item["answer"] for item in question_decomposition]
        for i, qd in enumerate(question_decomposition):
            subq = qd["question"]
            suba = qd["answer"]
            doc_idx = qd["paragraph_support_idx"]
            supporting_doc = context[doc_idx]
            supporting_doc = {supporting_doc["title"] : supporting_doc["paragraph_text"]}
            rewritten_subq = rewrite_subquestion(replace_placeholders(subq, sub_ans_list))
            query_doc_list.append({
                "triplet" : subq,
                "query" : rewritten_subq, 
                "doc" : supporting_doc,
                "dataset" : "musique",
                "qid" : qid
            })
        return query_doc_list
    except Exception as e:
        print(e)
        return []

def process_2wiki(data):
    try:
        context = data["context"]
        context = {item[0] : " ".join(item[1]) for item in context}
        supporting_fact_keys = [item[0] for item in data["supporting_facts"]]
        evidence_triplets = data["evidences"]
        assert len(supporting_fact_keys) == len(evidence_triplets), "The number of supporting facts and evidence triplets is not the same"
        query_doc_list = []
        for sfk, triplet in zip(supporting_fact_keys, evidence_triplets):
            subject, predicate, object = triplet
            prompt = triplet_to_query_prompt.format(subject, predicate)
            messages = [{"role" : "user", "content" : prompt}]
            response = chat_completion_call_deepseek_msg(messages, max_tokens=100, temperature=1.3)[0].strip("\"\'").replace("*","")
            query_doc_list.append({
                "triplet" : triplet,
                "query" : response, 
                "doc" : {sfk : context[sfk]},
                "dataset" : "2wikimqa",
                "qid" : data["_id"]
            })
        return query_doc_list
    except Exception as e:
       print(e)
       return []
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["2wikimqa", "musique"])
    parser.add_argument("--merge", action="store_true", default=False)
    parser.add_argument("--formatting", action="store_true", default=False)
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--test_size", type=int, default=2000)
    args = parser.parse_args()
    return args
    


if __name__ == "__main__":
    args = parse_args()
    
    if args.dataset == "2wikimqa":
        datas_2wiki = json.load(open("/home/zchu/datasets/mhqa/raw_data/2wikimqa/train.json"))
        selected_datas = random.choices(datas_2wiki, k=5000)

        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            results = list(tqdm(executor.map(process_2wiki, selected_datas), total=len(selected_datas)))
        
        flatten_results = list(chain(*results))
        print(len(flatten_results))
        with open("/home/zchu/codes/train_2412/data_generation/ours_T2/critic/relevance/outputs/rule_based/2wiki_results.json", "w") as f:
            json.dump(flatten_results, f, ensure_ascii=False, indent=4)
    elif args.dataset == "musique":
        datas_musique = [json.loads(line) for line in open("/home/zchu/datasets/mhqa/raw_data/musique/musique_ans_v1.0_train.jsonl")]
        selected_datas = random.choices(datas_musique, k=5000)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            results = list(tqdm(executor.map(process_musique, selected_datas), total=len(selected_datas)))
        
        flatten_results = list(chain(*results))
        print(len(flatten_results))
        with open("/home/zchu/codes/train_2412/data_generation/ours_T2/critic/relevance/outputs/rule_based/musique_results.json", "w") as f:
            json.dump(flatten_results, f, ensure_ascii=False, indent=4)
    else:
        print(f"Dataset {args.dataset} not supported")
        
    if args.merge:
        def is_doc_same(doc1, doc2):
            if list(doc1.keys())[0] == list(doc2.keys())[0]:
                return True
            else:
                return False
        datas_2wiki = json.load(open("/home/zchu/codes/train_2412/data_generation/ours_T2/critic/relevance/outputs/rule_based/2wiki_results.json"))
        datas_musique = json.load(open("/home/zchu/codes/train_2412/data_generation/ours_T2/critic/relevance/outputs/rule_based/musique_results.json"))
        datas = datas_2wiki + datas_musique
        all_docs = [item["doc"] for item in datas]
        for data in datas:
            cur_doc = data["doc"]
            rand_doc = random.choices(all_docs, k=3)
            rand_doc = [item for item in rand_doc if not is_doc_same(cur_doc, item)]
            data.update({"irrelevant_docs" : rand_doc})
        
        with open("/home/zchu/codes/train_2412/data_generation/ours_T2/critic/relevance/outputs/rule_based/merged_results.json", "w") as f:
            json.dump(datas, f, ensure_ascii=False, indent=4)
    
    RELEVANCE_INSTRUCTION = "You will be provided with a question, along with an evidence document.\nYour job is to determine whether the evidence is relevant to the question, and provide explanation for your decision.\nIf the evidence is relevant to the question, you should response with [Relevant]; otherwise, response with [Irrelevant].\n\n### Question: {}\n\n### Evidence: {}"
    if args.formatting:
        new_datas = []
        datas = json.load(open("/home/zchu/codes/train_2412/data_generation/ours_T2/critic/relevance/outputs/rule_based/merged_results.json"))
        for data in datas:
            query = data["query"]
            doc_text = [f"{k}\n{v}" for k, v in data["doc"].items()][0]
            instruction = RELEVANCE_INSTRUCTION.format(query, doc_text)
            output = {
                "rating" : "[Relevant]",
                "explanation" : None
            }
            new_datas.append({
                "instruction" : instruction,
                "output" : output,
                "meta" : {
                    "task" : "relevance",
                    "dataset" : data["dataset"],    
                    "type" : None,
                    "qid" : data["qid"]
                }
            })
            
            # negative samples
            irrelevant_doc = random.choice(data["irrelevant_docs"])
            irrelevant_doc_text = [f"{k}\n{v}" for k, v in irrelevant_doc.items()][0]
            instruction_irrelevant = RELEVANCE_INSTRUCTION.format(query, irrelevant_doc_text)
            output_irrelevant = {
                "rating" : "[Irrelevant]",
                "explanation" : None
            }
            new_datas.append({
                "instruction" : instruction_irrelevant,
                "output" : output_irrelevant,
                "meta" : {
                    "task" : "relevance",
                    "dataset" : data["dataset"],
                    "type" : None,
                    "qid" : data["qid"]
                }
            })
        random.shuffle(new_datas)
        if args.train_size is None:
            train_datas = new_datas[:-args.test_size]
        else:
            assert args.train_size + args.test_size <= len(new_datas), "Train size + Test size should be less than the number of parsed datas"
            train_datas = new_datas[:args.train_size]
        test_datas = new_datas[-args.test_size:]
        with open("/home/zchu/codes/train_2412/data_generation/ours_T2/critic/relevance/outputs/rule_based/train.json", "w") as f:
            json.dump(train_datas, f, ensure_ascii=False, indent=4)
        with open("/home/zchu/codes/train_2412/data_generation/ours_T2/critic/relevance/outputs/rule_based/test.json", "w") as f:
            json.dump(test_datas, f, ensure_ascii=False, indent=4)

## Example command
# python -m data_generation.ours_T2.critic.relevance.rule_based --dataset 2wikimqa
# python -m data_generation.ours_T2.critic.relevance.rule_based --dataset musique
# python -m data_generation.ours_T2.critic.relevance.rule_based --merge
# python -m data_generation.ours_T2.critic.relevance.rule_based --formatting --train_size 20000 --test_size 3000