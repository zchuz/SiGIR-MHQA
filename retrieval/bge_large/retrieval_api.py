import os
import sys
import json
import torch
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
from copy import deepcopy
from transformers import AutoTokenizer, AutoModel
from bge_large.utils import pooling, format_data

DATA_PATH_DICT = {
    "2wikimqa" : "/PATH/TO/CORPUS/2wikimqa.json",
    "hotpotqa" : "/PATH/TO/CORPUS/hotpotqa.json",
    "musique" : "/PATH/TO/CORPUS/musique.json"
}

PREBUILD_INDEX_DICT = {
    "2wikimqa" : "/share/home/zchu/codes/bm25/service/bge_large/prebuild_index/2wikimqa_embeddings.pt",
    "hotpotqa" : "/share/home/zchu/codes/bm25/service/bge_large/prebuild_index/hotpotqa_embeddings.pt",
    "musique" : "/share/home/zchu/codes/bm25/service/bge_large/prebuild_index/musique_embeddings.pt"
}

# PORT_DICT = {
#     "2wikimqa" : 1440,
#     "hotpotqa" : 1450,
#     "musique" : 1460,
# }

@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--dataset", type=str, default="2wikimqa", choices=["2wikimqa", "hotpotqa", "musique"])
    return parser.parse_args()  

class SearchRequest(BaseModel):
    query: str = None,
    n: int = 10,
    dataset: str = None
    
class SearchResponse(BaseModel):
    search_results: List[dict] = None

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    if not isinstance(request.query, str):
        raise HTTPException(status_code=400, detail="Query must be a string")
    dataset = request.dataset
    if dataset not in ["2wikimqa", "hotpotqa", "musique"]:
        raise HTTPException(status_code=400, detail="Dataset must be one of 2wikimqa, hotpotqa, musique")
    
    prebuild_index = dataset_to_prebuild_index[dataset]
    documents = dataset_to_datas[dataset]
    
    batch_encodings = tokenizer([request.query], padding=True, truncation=True, return_tensors="pt")
    batch_encodings = {k: v.cuda() for k, v in batch_encodings.items()}
    with torch.no_grad():
        outputs = model(**batch_encodings)
        query_embeddings = pooling(outputs)
        #query_embeddings = mean_pooling(outputs.last_hidden_state, batch_encodings["attention_mask"])
        sim_scores = query_embeddings @ prebuild_index.t() #[1, num_docs]
        sim_scores = sim_scores.squeeze(0) #[num_docs]
        sim_sorted = sim_scores.argsort(descending=True)
        topk_indices = sim_sorted[:request.n] # [num_docs]
        topk_scores = sim_scores[topk_indices].cpu().tolist()
        search_results = [documents[i] for i in topk_indices]
        search_results = deepcopy(search_results)
        for i, item in enumerate(search_results):

            item["score"] = topk_scores[i]
            item["paragraph_text"] = item["text"]
            del item["text"]
        
    return SearchResponse(search_results=search_results)
    
    
if __name__ == "__main__":
    args = parse_args()
    datas_2wikimqa = json.load(open(DATA_PATH_DICT["2wikimqa"]))
    datas_hotpotqa = json.load(open(DATA_PATH_DICT["hotpotqa"]))
    datas_musique = json.load(open(DATA_PATH_DICT["musique"]))
    prebuild_index_2wikimqa = torch.load(PREBUILD_INDEX_DICT["2wikimqa"]).cuda()
    prebuild_index_hotpotqa = torch.load(PREBUILD_INDEX_DICT["hotpotqa"]).cuda()
    prebuild_index_musique = torch.load(PREBUILD_INDEX_DICT["musique"]).cuda()
    dataset_to_datas = {
        "2wikimqa" : datas_2wikimqa,
        "hotpotqa" : datas_hotpotqa,
        "musique" : datas_musique
    }
    dataset_to_prebuild_index = {
        "2wikimqa" : prebuild_index_2wikimqa,
        "hotpotqa" : prebuild_index_hotpotqa,
        "musique" : prebuild_index_musique
    }
    
    model = AutoModel.from_pretrained("/share/home/zchu/pretrained/bge-large-en-v1.5").cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained("/share/home/zchu/pretrained/bge-large-en-v1.5")
    
    uvicorn.run(app, host="0.0.0.0", port=2450)
    print("BGE Large server started on port 2450")
    
## How to run the server:
# 1. conda activate vllm
# 2. cd /home/zchu/codes/ProbTree/src/service/contriever
# 3. python retrieval_api.py

## How to call the server:
# import requests
# url = "http://localhost:2440/search"
# data = {
#     "query": "What is the capital of France?",
#     "n" : 3,
#     "dataset" : "2wikimqa"
# }
# response = requests.post(url, json=data).json()
