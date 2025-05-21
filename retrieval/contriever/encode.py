# conda env: vllm
import os
import sys
import json
import torch
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModel
from contriever.utils import mean_pooling, format_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="2wikimqa", choices=["2wikimqa", "hotpotqa", "musique"])
    parser.add_argument("--batch_size", type=int, default=512)
    return parser.parse_args()

FILE_PATH_DICT = {
    "2wikimqa" : "/PATH/TO/CORPUS/2wikimqa.json",
    "hotpotqa" : "/PATH/TO/CORPUS/hotpotqa.json",
    "musique" : "/PATH/TO/CORPUS/musique.json"
}

if __name__ == "__main__":
    args = parse_args()
    model = AutoModel.from_pretrained("/home/zchu/pretrained/embedding_models/contriever-msmarco").cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained("/home/zchu/pretrained/embedding_models/contriever-msmarco")
    datas = json.load(open(FILE_PATH_DICT[args.dataset]))
    batch_size = args.batch_size
    for idx in trange(0, len(datas), batch_size):
        batch_datas = [format_data(data) for data in datas[idx:idx+batch_size]]
        with torch.no_grad():
            batch_encodings = tokenizer(batch_datas, padding=True, truncation=True, return_tensors="pt")
            batch_encodings = {k: v.cuda() for k, v in batch_encodings.items()}
            batch_outputs = model(**batch_encodings)
            batch_embeddings = mean_pooling(batch_outputs.last_hidden_state, batch_encodings["attention_mask"])
            
            if idx == 0:
                embeddings_list = [batch_embeddings.cpu()]
            else:
                embeddings_list.append(batch_embeddings.cpu())

    # Concatenate all embeddings at once after processing all batches
    all_embeddings = torch.cat(embeddings_list, dim=0)
    
    # Save embeddings
    save_path = f"/home/zchu/codes/ProbTree/src/service/contriever/prebuild_index/{args.dataset}_embeddings.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(all_embeddings, save_path)

    # Example code to load embeddings:
    # embeddings = torch.load(f"/home/zchu/codes/ProbTree/src/service/embeddings/{args.dataset}_embeddings.pt")
