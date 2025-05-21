# create date: 2024/12/18
# conda environment: vllm
# This file is used for parsing the critic datas

import os
import re
import json
import random
import argparse
from tqdm import tqdm
from collections import defaultdict


#{"input_prompt": "...", "qid": "4786401208db11ebbd9cac1f6bf848b6", "type": "bridge_comparison", "dataset": "2wikimqa", "task": "groundness", "response": "Rating: [Fully supported]\nExplanation: The response is fully supported by the evidence. The evidence clearly states that \"To Walk with Lions\" is a 1999 film directed by Carl Schultz. The response accurately extracts this information and concludes that Carl Schultz is the director of the film, which is directly supported by the evidence provided."}

# - [4] The reasoning process is clear and the answer is correct. (partial match = 1)
# - [3] The reasoning process is clear and the answer is partially correct. (f1 > 0.7)
# - [2] The reasoning process is flawed, with the answer partially correct. (f1 < 0.7)
# - [1] The reasoning process is flawed, with the answer incorrect. (f1 = 0)
# - [0] The reasoning doesn't provide the final answer. (answer = unknown)

INSTRUCTION = {
    "groundness": """You will be provided with a question, an evidence document, along with a response.
Your job is to determine whether the response is supported by the evidence, and provide explanation for your decision.
Use the following scale to rate the response:
[Fully supported] Most of the information in the response is supported by the evidence.
[Partially supported] Some of the information in the response is supported by the evidence, but there are some parts that are speculative.
[Not supported] The response is not supported by the evidence / The response does not provide useful reasoning.""",
    "relevance": """You will be provided with a question, along with an evidence document.
Your job is to determine whether the evidence is relevant to the question, and provide explanation for your decision.
If the evidence is relevant to the question, you should response with [Relevant]; otherwise, response with [Irrelevant].""",
    "utility": """You will be provided with a question, along with a reasoning trajectory.
Your job is to determine whether the reasoning trajectory is useful for answering the question, and provide explanation for your decision.
Use the following scale to rate the reasoning:
[4]: The reasoning process is clear, logically structured, and well-supported by the evidence.
[3]: The reasoning process is mostly clear, partially logically structured, and supported by evidence, but may contains minor logical flaws.
[2]: The reasoning is somewhat unclear with noticeable flaws and uses limited or weak evidence.
[1]: The reasoning is flawed, lacking supporting evidence, which results in an incorrect conclusion.
[0]: The reasoning fails to provide a definitive answer."""
}

INSTRUCTION_WITH_IN_CONTEXT = {
    "groundness": """You will be provided with a question, an evidence document, along with a response.
Your job is to determine whether the response is supported by the evidence, and provide explanation for your decision.
Use the following scale to rate the response:
[Fully supported] Most of the information in the response is supported by the evidence.
[Partially supported] Some of the information in the response is supported by the evidence, but there are some parts that are speculative.
[Not supported] The response is not supported by the evidence / The response does not provide useful reasoning.

### Question: In which country is the Financial Conduct Authority located?

### Evidence: Martin Wheatley
Martin Wheatley is a British financier, formerly managing director of the Consumer and Markets Business Unit of the Financial Services Authority in the UK, and is the former CEO of the Financial Conduct.

### Response: From Document, we know that Martin Wheatley is a British financier who formerly managed the Consumer and Markets Business Unit of the Financial Services Authority in the UK. This indicates that the Financial Conduct Authority is located in the United Kingdom.

Rating: [Fully supported]
Explanation: The response is fully supported by the evidence. The evidence clearly states that Martin Wheatley is a British financier who formerly managed the Consumer and Markets Business Unit of the Financial Services Authority in the UK and was the former CEO of the Financial Conduct Authority. The response accurately extracts this information and concludes that the Financial Conduct Authority is located in the United Kingdom, which is directly supported by the evidence provided.

### Question: What is the birthplace of Jane Siberry?

### Evidence: Gavin Bradley 
Gavin Bradley is an award-winning Canadian singer-songwriter and producer based in Toronto who has worked with artists like Nelly Furtado, Tori Amos and Jane Siberry. Fusing acoustic and electronic elements, his work is identifiable for its signature "warm" piano sound and live strings mixed with filtered synthesizers and other electronic manipulations . Besides production, Bradley is a solo recording artist. His debut album 'Deep Freeze' was released on UMI Records in 2006.

### Response: The response is partially supported because the evidence mentions Gavin Bradley's Toronto connection but does not directly state Jane Siberry's birthplace. The inference about her possible connection to Toronto is speculative.

Rating: [Partially supported]
Explanation: The response correctly identifies that the evidence mentions Gavin Bradley's connection to Toronto but does not directly state Jane Siberry's birthplace. The inference that Jane Siberry might be connected to Toronto based on her collaboration with Gavin Bradley is speculative, as the evidence does not provide explicit information about her birthplace. Therefore, the response is partially supported.

### Question: What is the city where WXPN is located?

### Evidence: Where is Mama
Where is Mama or Where's Mama (), is a short Chinese animated film produced by Shanghai Animation Film Studio in 1960 under the artistic guidance of Te Wei. The narrated film describes the adventures and misadventures of a group of tadpoles in search of their mother. It is one of Te Wei's first attempts to break away from Western style animation and aim for a painterly style influenced by Qi Baishi and more in keeping with native Chinese aesthetic sensibilities. Because of its simple story line and repetitive script it is ideal for children who are beginning their study of the Chinese language.

### Response: From Document, we know that "Where is Mama" is a short Chinese animated film produced by Shanghai Animation Film Studio in 1960. Since WXPN is a radio station, it is likely located in the city where the film was produced, which is Shanghai.

Rating: [Not supported]
Explanation: The response is not supported by the evidence provided. The evidence document discusses a short Chinese animated film called "Where is Mama," produced by Shanghai Animation Film Studio in 1960, and provides information about its artistic style and target audience. However, there is no mention of WXPN, a radio station, in the evidence document. The response speculates that WXPN is likely located in Shanghai because of the connection to the film's production, but this is an assumption that is not grounded in the evidence. Therefore, the response is not supported by the evidence.""",
    "relevance": """You will be provided with a question, along with an evidence document.
Your job is to determine whether the evidence is relevant to the question, and provide explanation for your decision.
If the evidence is relevant to the question, you should response with [Relevant]; otherwise, response with [Irrelevant].

### Question: In what country was the first Pan-African conference held?

### Evidence: First Pan-African Conference
The First Pan-African Conference was held in London from 23 to 25 July 1900 (just prior to the Paris Exhibition of 1900 ``in order to allow tourists of African descent to attend both events ''). Organized primarily by the Trinidadian barrister Henry Sylvester Williams, it took place in Westminster Town Hall (now Caxton Hall) and was attended by 37 delegates and about 10 other participants and observers from Africa, the West Indies, the US and the UK, including Samuel Coleridge Taylor (the youngest delegate), John Alcindor, Dadabhai Naoroji, John Archer, Henry Francis Downing, and W.E.B. Du Bois, with Bishop Alexander Walters of the AME Zion Church taking the chair.

Rating: [Relevant]
Explanation: The evidence directly answers the question by stating that the first Pan-African conference was held in London, which is in the United Kingdom. The details provided, such as the dates and location of the conference, further confirm the relevance of the evidence to the question.

### Question: What is the administrative territorial entity for Malta Township?

### Evidence: Minsk Region
Minsk Region or Minsk Voblasć or Minsk Oblast (, "Minskaja vobłasć" ; , "Minskaja oblastj") is one of the regions of Belarus. Its administrative center is Minsk, although it is a separate administrative territorial entity of Belarus. As of 2011, the region's population is 1,411,500.

Rating: [Irrelevant]
Explanation: The evidence provided discusses the Minsk Region, which is a region in Belarus. The question, however, asks about the administrative territorial entity for Malta Township. Since the evidence pertains to Belarus and not Malta, it is not relevant to the question.""",
    "utility": """You will be provided with a question, along with a reasoning trajectory.
Your job is to determine whether the reasoning trajectory is useful for answering the question, and provide explanation for your decision.
Use the following scale to rate the reasoning:
[4]: The reasoning process is clear, logically structured, and well-supported by the evidence.
[3]: The reasoning process is mostly clear, partially logically structured, and supported by evidence, but may contains minor logical flaws.
[2]: The reasoning is somewhat unclear with noticeable flaws and uses limited or weak evidence.
[1]: The reasoning is flawed, lacking supporting evidence, which results in an incorrect conclusion.
[0]: The reasoning fails to provide a definitive answer.

### Question: Are Karel Dujardin and Víctor Latou both from the same country?"""
}

def rule_based_utility(item: dict):
    metrics = item["metrics"]
    prediction = item["prediction"]
    em, pm, f1 = metrics["em"], metrics["pm"], metrics["f1"]
    if pm == 1.0 or f1 >= 0.85:
        return "[4]"

    if f1 >= 0.6:
        return "[3]"
    
    if f1 == 0.0 and (
        "not" in prediction.lower() or
        "unknown" in prediction.lower()        
    ):
        return "[0]"
    
    if f1 == 0.0:
        return "[1]"

    return "[2]"

# Scale: [Fully supported] [Partially supported] [Not supported]
def parse_groundness_data(item):
    instruction_with_context = item["input_prompt"]
    instruction = instruction_with_context.replace(INSTRUCTION_WITH_IN_CONTEXT["groundness"], "").strip()
    instruction = f'{INSTRUCTION["groundness"]}\n\n{instruction}'
    
    response = item["response"]
    sentences = response.split("\n")
    sent_rating = sentences[0].strip()
    sent_explanation = " ".join([item.strip() for item in sentences[1:]]).strip()
    
    if "fully support" in sent_rating.lower():
        rating = "[Fully supported]"
    elif "partially" in sent_rating.lower():
        rating = "[Partially supported]"
    elif "not" in sent_rating.lower():
        rating = "[Not supported]"
    else:
        raise ValueError(f"Unknown rating: {sent_rating}")
    
    explanation = sent_explanation.replace("Explanation", "").strip(": ").strip()
    return {
        "instruction": instruction,
        "output": {
            "rating": rating,
            "explanation": explanation
        },
        "meta": {
            "task": "groundness",
            "dataset": item["dataset"],
            "type": item["type"],
            "qid": item["qid"]
        }
    }

# Scale: [Relevant] [Irrelevant]
def parse_relevance_data(item):
    instruction_with_context = item["input_prompt"]
    instruction = instruction_with_context.replace(INSTRUCTION_WITH_IN_CONTEXT["relevance"], "").strip()
    instruction = f'{INSTRUCTION["relevance"]}\n\n{instruction}'
    
    response = item["response"]
    sentences = response.split("\n")
    sent_rating = sentences[0].strip()
    sent_explanation = " ".join([item.strip() for item in sentences[1:]]).strip()
    
    if "irrelevant" in sent_rating.lower():
        rating = "[Irrelevant]"
    elif "relevant" in sent_rating.lower():
        rating = "[Relevant]"
    else:
        raise ValueError(f"Unknown rating: {sent_rating}")
    
    explanation = sent_explanation.replace("Explanation", "").strip(": ").strip()
    return {
        "instruction": instruction,
        "output": {
            "rating": rating,
            "explanation": explanation
        },
        "meta": {
            "task": "relevance",
            "dataset": item["dataset"],
            "type": item["type"],
            "qid": item["qid"]
        }
    }
# Scale: [4] [3] [2] [1] [0]
def parse_utility_data(item):
    instruction_with_context = item["input_prompt"]
#     instruction = instruction_with_context.replace(INSTRUCTION_WITH_IN_CONTEXT["utility"], "").replace("""Rating: 
# Explanation:""", "").strip()
    instruction = instruction_with_context.replace("Rating: \nExplanation:", "").strip()
    #instruction = f'{INSTRUCTION["utility"]}\n\n{instruction}'

    response = item["response"]
    sentences = response.split("\n")
    sent_rating = sentences[0].strip()
    sent_explanation = " ".join([item.strip() for item in sentences[1:]]).strip()
    
    rating = re.findall(r"\d", sent_rating)[0]
    explanation = sent_explanation.replace("Explanation", "").strip(": ").strip()
    
    rule_based_rating = rule_based_utility(item)
    return {
        "instruction": instruction,
        "output": {
            "rating": f"[{rating}]",
            "explanation": explanation
        },
        "rule_based_rating": rule_based_rating,
        "meta": {
            "task": "utility",
            "dataset": item["dataset"],
            "type": item["type"],
            "qid": item["qid"]
        }
    }

PARSE_FUNC = {
    "groundness": parse_groundness_data,
    "relevance": parse_relevance_data,
    "utility": parse_utility_data
}

def parse_datas(datas, task):
    print(f"Parsing {task} datas...")
    print(f"\tNumber of datas: {len(datas)}")
    parsed_datas = []
    for item in tqdm(datas):
        try:
            parsed = PARSE_FUNC[task](item)
            parsed_datas.append(parsed)
        except Exception as e:
            #print(f"Error: {e}")
            #print(f'Item: {item["response"]}')
            pass
    print(f"Number of parsed datas: {len(parsed_datas)}")
    return parsed_datas

def load_jsonl(fin):
    datas = []
    with open(fin, "r") as f:
        for line in f:
            datas.append(json.loads(line))
    return datas

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fin_groundness", type=str, default=None)
    parser.add_argument("--fin_relevance", type=str, default=None)
    parser.add_argument("--fin_utility", type=str, default=None)
    parser.add_argument("--fout", type=str, default=None)
    parser.add_argument("--test_size", type=int, default=2000)
    parser.add_argument("--train_size", type=int, default=None)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    merged_datas = []
    train_datas = []
    test_datas = []
    if args.fin_groundness is not None:
        datas = load_jsonl(args.fin_groundness)
        parsed_datas = parse_datas(datas, "groundness")
        merged_datas.extend(parsed_datas)
        random.shuffle(parsed_datas)
        if args.train_size is None:
            train_datas.extend(parsed_datas[:-args.test_size])
        else:
            assert args.train_size + args.test_size <= len(parsed_datas), "Train size + Test size should be less than the number of parsed datas"
            train_datas.extend(parsed_datas[:args.train_size])
        test_datas.extend(parsed_datas[-args.test_size:])
    
    if args.fin_relevance is not None:
        datas = load_jsonl(args.fin_relevance)
        parsed_datas = parse_datas(datas, "relevance")
        merged_datas.extend(parsed_datas)
        random.shuffle(parsed_datas)
        if args.train_size is None:
            train_datas.extend(parsed_datas[:-args.test_size])
        else:
            assert args.train_size + args.test_size <= len(parsed_datas), "Train size + Test size should be less than the number of parsed datas"
            train_datas.extend(parsed_datas[:args.train_size])
        test_datas.extend(parsed_datas[-args.test_size:])
        
    if args.fin_utility is not None:
        datas = load_jsonl(args.fin_utility)
        parsed_datas = parse_datas(datas, "utility")
        merged_datas.extend(parsed_datas)
        random.shuffle(parsed_datas)
        if args.train_size is None:
            train_datas.extend(parsed_datas[:-args.test_size])
        else:
            assert args.train_size + args.test_size <= len(parsed_datas), "Train size + Test size should be less than the number of parsed datas"
            train_datas.extend(parsed_datas[:args.train_size])
        test_datas.extend(parsed_datas[-args.test_size:])
    
    print(f"Number of merged parsed datas: {len(merged_datas)}")
    print(f"\tNumber of Train set: {len(train_datas)}")
    print(f"\tNumber of Test set: {len(test_datas)}")
    
    if not os.path.exists(args.fout):
        os.makedirs(args.fout, exist_ok=True)
    with open(os.path.join(args.fout, "merged_results.json"), "w") as fout:
        json.dump(merged_datas, fout, indent=4, ensure_ascii=False)
    with open(os.path.join(args.fout, "train.json"), "w") as fout:
        json.dump(train_datas, fout, indent=4, ensure_ascii=False)
    with open(os.path.join(args.fout, "test.json"), "w") as fout:
        json.dump(test_datas, fout, indent=4, ensure_ascii=False)
## Example command
# python -m data_generation.ours_T2.critic.parse_critic_datas --fin_groundness ./outputs/groundness/fewshot_v2/cache.jsonl --fin_relevance ./outputs/relevance/fewshot_v2/cache.jsonl --fin_utility ./outputs/utility/zeroshot_v2/cache.jsonl --fout ./outputs

# python -m data_generation.ours_T2.critic.parse_critic_datas --fin_groundness ./data_generation/ours_T2/critic/outputs/groundness/fewshot_v2/cache.jsonl --fin_relevance ./data_generation/ours_T2/critic/outputs/relevance/fewshot_v2/cache.jsonl --fin_utility ./data_generation/ours_T2/critic/outputs/utility/zeroshot_v2/cache.jsonl --fout ./data_generation/ours_T2/critic/outputs