import re
import os
import sys
import json
import string
import requests
from datetime import datetime
from collections import Counter, defaultdict

# PROXY = "http://127.0.0.1:17890"
# os.environ["http_proxy"] = PROXY
# os.environ["https_proxy"] = PROXY
# os.environ["no_proxy"] = "localhost,127.0.0.1,127.0.0.0,0.0.0.0"

def normalize_answer(s):
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s.strip()))))

def f1_score(prediction, ground_truth):
    """return F1, P, R
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def em_score(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def pm_score(prediction, ground_truth):
    """PM is abbreviation for prediction match.
    PM = 1 if
        f1 >= 0.8 
    or
        f1 >= 0.6 and (prediction contains ground truth) or (ground truth contains prediction)
    Args:
        prediction (str): prediction answer
        ground_truth (str): ground truth answer
    """
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    f1, _, _ = f1_score(prediction, ground_truth)
    if f1 >= 0.8:
        return 1
    elif f1 >= 0.6 and (prediction in ground_truth or ground_truth in prediction):
        return 1
    return 0

def extract_answer(text):
    pattern = r"\*\*(.*?)\*\*"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return "Unknown"


from openai import AzureOpenAI, OpenAI
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import os
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential, 
    wait_exponential   
)
from typing import List, Dict

### Completion
deepseek_client = OpenAI(api_key="sk-7d030b21a2f84dcfb91a00f4c8f10bde", base_url="https://api.deepseek.com")
openai_client = OpenAI(api_key="sk-BXP6K1mC4WzgOCG3xH76kwL4m8ezfrQXfpwwI4xtgXSvbSA5", base_url="https://api.openai-proxy.org/v1")
if os.environ.get("MY_PORT", None) is None:
    print("MY_PORT is not set, using default port 8000")
    vllm_client = OpenAI(api_key="abc", base_url="http://localhost:8000/v1")
else:
    print(f"MY_PORT is set to {os.environ.get('MY_PORT')}")
    vllm_client = OpenAI(api_key="abc", base_url=f"http://localhost:{os.environ.get('MY_PORT')}/v1")

vllm_client_responser = OpenAI(api_key="abc", base_url="http://localhost:8200/v1")
vllm_client_decomposer = OpenAI(api_key="abc", base_url="http://localhost:8190/v1")

def completion_call_vllm_base(prompt, model, temperature=0.0, n=1, top_p=1.0, max_tokens=1024, stop=None, stop_token_ids=None, client="default"):
    if client == "default":
        client = vllm_client
    elif client == "responser":
        client = vllm_client_responser
    elif client == "decomposer":
        client = vllm_client_decomposer
    else:
        raise ValueError(f"Invalid client: {client}")
    return client.completions.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        n=n,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop,
        extra_body={
            "stop_token_ids" : stop_token_ids,
            "skip_special_tokens" : False,
            "include_stop_str_in_output" : True,
            "spaces_between_special_tokens" : False,
            "repetition_penalty" : 1.2
        }
    )

def completion_call_vllm_responser(prompt, model, temperature=0.0, n=1, top_p=1.0, max_tokens=256, stop=None, stop_token_ids=None):
    return completion_call_vllm_base(prompt, model, temperature, n, top_p, max_tokens, stop, stop_token_ids, client="responser")

def completion_call_vllm_decomposer(prompt, model, temperature=0.0, n=1, top_p=1.0, max_tokens=256, stop=None, stop_token_ids=None):
    return completion_call_vllm_base(prompt, model, temperature, n, top_p, max_tokens, stop, stop_token_ids, client="decomposer")

def completion_call_vllm(prompt, model, temperature=0.0, n=1, top_p=1.0, max_tokens=1024, stop=None, stop_token_ids=None):
    response = vllm_client.completions.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        n=n,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop,
        extra_body={
            "stop_token_ids" : stop_token_ids,
            "skip_special_tokens" : False,
            "include_stop_str_in_output" : True,
            "spaces_between_special_tokens" : False,
            "repetition_penalty" : 1.2
        }
    )
    return response

def completion_call_vllm_with_dynamic_port(prompt, model, temperature=0.0, port=None, n=1, top_p=1.0, max_tokens=512, stop=None, stop_token_ids=None):
    assert port is not None, "Vllm Access Port is not set"
    client = OpenAI(api_key="abc", base_url=f"http://localhost:{port}/v1")
    return client.completions.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        n=n,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop,
        extra_body={
            "stop_token_ids" : stop_token_ids,
            "skip_special_tokens" : False,
            "include_stop_str_in_output" : True,
            "spaces_between_special_tokens" : False,
            "repetition_penalty" : 1.2
        }
    )
@retry(wait=wait_random_exponential(min=1, max=4), stop=stop_after_attempt(2)) 
def chat_completion_call_deepseek_msg(messages, temperature=0.0, n=1, top_p=1.0,stop=None, max_tokens=1024, logprobs=None, model="deepseek-chat")-> List[str]:
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
        stream=False,
        top_p=top_p
    )
    return [item.message.content for item in response.choices]

@retry(wait=wait_random_exponential(min=1, max=4), stop=stop_after_attempt(2)) 
def chat_completion_call_openai_msg(messages, temperature=0.0, n=1, top_p=1.0, stop=None, max_tokens=1024, logprobs=None, model="gpt-4o-mini")-> List[str]:
    assert model in ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    if model == "gpt-4o-mini":
        model = "gpt-4o-mini-2024-07-18"
    elif model == "gpt-4o":
        model = "gpt-4o-2024-08-06"
    elif model == "gpt-3.5-turbo":
        model = "gpt-3.5-turbo-0125"
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        n=n,
        temperature=temperature,
        stop=stop,
        stream=False,
        top_p=top_p
    )
    return [item.message.content for item in response.choices]

## Retrieval
DATASET_PORT_DICT = {
    "2wikimqa" : 1440,
    "hotpotqa" : 1450,
    "musique" : 1460,
    "bamboogle": 1440
}

@retry(wait=wait_random_exponential(min=1, max=2), stop=stop_after_attempt(2)) 
def BM25_retrieval(query, dataset, topk=100):
    port = DATASET_PORT_DICT[dataset]
    #url = f'http://localhost:{port}/' # This is Local launched BM25
    url = f'http://172.16.1.76:{port}/' # This is remote launched BM25 (on cs1 Jump server)
    post =  {
        "query": query,
        "k": topk
    }   
    res = requests.post(url, json=post)
    if res.status_code != 200:
        raise Exception(f"Failed to retrieve from {dataset} with status code {res.status_code}")
    return res.json()[:topk]


# Dense retrieval using contriever-msmarco
@retry(wait=wait_random_exponential(min=1, max=2), stop=stop_after_attempt(2))
def dense_retrieval(query:str, dataset:str, topk:int=10):
    dense_retr_port = os.environ.get("DENSE_RETR_PORT", 2440)
    url = f"http://gpu06:{dense_retr_port}/search"
    post = {
        "query" : query,
        "dataset" : dataset,
        "n" : topk
    }
    res = requests.post(url, json=post)
    if res.status_code != 200:
        raise Exception(f"Failed to retrieve from {dataset} with status code {res.status_code}")
    return res.json()["search_results"][:topk]

# Mixed Retrieval
@retry(wait=wait_random_exponential(min=1, max=2), stop=stop_after_attempt(2))
def mix_retrieval(query:str, dataset: str, topk=3):
    dense_retr_port = os.environ.get("DENSE_RETR_PORT", 2440)
    url_sparse = f'http://172.17.3.203:{DATASET_PORT_DICT[dataset]}/'
    url_dense = f'http://gpu06:{dense_retr_port}/search'
    
    post_sparse = {
        "query" : query, "k":topk
    }
    post_dense = {
        "query" : query, "n":topk, "dataset":dataset
    }
    res_sparse = requests.post(url_sparse, json=post_sparse)
    res_dense = requests.post(url_dense, json=post_dense)
    if res_sparse.status_code != 200 or res_dense.status_code != 200:
        raise Exception(f"Failed to retrieve from {dataset} with status.")
    
    return res_dense.json()["search_results"][:topk] + res_sparse.json()[:topk]
    

def format_retrieval_results(results: List[Dict], topk=3):
    results = results[:topk]
    doc_text_str = ""
    for i, result in enumerate(results):
        doc_text_str += f"#{i+1} Wikipedia Title: {result['title']}\nText: {result['paragraph_text']}\n"
    return doc_text_str.strip()


## Data utils
def load_datas(dataset):
    if dataset == "2wikimqa":
        return json.load(open("./datas_for_acl25_mhqa/2wiki/train.20k.json"))
    elif dataset == "hotpotqa":
        return json.load(open("./datas_for_acl25_mhqa/hotpotqa/train.10k.json"))
    elif dataset == "musique":
        return json.load(open("./datas_for_acl25_mhqa/musique/train.19k.json"))
    