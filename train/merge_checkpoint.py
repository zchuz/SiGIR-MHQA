# create date: 2024-12-23
# conda env: llama_factory_810
# This file is used to merge the lora adapter with the base model.

import os
import json
import time
import torch
import argparse
import numpy as np

from tqdm import tqdm
from copy import deepcopy
from safetensors.torch import load_file
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
from train.utils import get_logger, resize_token_embeddings_with_init

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/home/share/models/Mistral-7B-v0.2")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resize_vocab", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()
    return args

def get_base_name(model_name_or_path):
    directories = model_name_or_path.split(os.sep)
    if "checkpoint" in directories[-1]:
        return directories[-2]
    else:
        return directories[-1]

logger = get_logger(__name__)
if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Args: {json.dumps(vars(args), indent=4, ensure_ascii=False)}")
    
    output_dir = f"./.cache/{get_base_name(args.model_name_or_path)}/" + get_base_name(args.adapter_path) if args.output_dir is None else args.output_dir
    logger.info(f"Output dir: {output_dir}")
    if os.path.exists(output_dir):
        if not args.overwrite:
            raise FileExistsError(f"Merged model already exists at {output_dir}")
        else:
            logger.warning(f"Merged model already exists at {output_dir}. Overwriting...")
    else:
        os.makedirs(output_dir, exist_ok=False)
    # ## merge checkpoints
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    
    if args.resize_vocab:
        logger.info("Resizing tokenize and input/output embeddings...")
        resize_token_embeddings_with_init(model=base_model, tokenizer=tokenizer, init=False)

        embeddings_tensors = load_file(os.path.join(args.adapter_path, "embeddings.safetensors"))
        base_model.load_state_dict(embeddings_tensors, strict=False)
    
    merged_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    merged_model = merged_model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Merged model and tokenizer saved to {output_dir}")
        
    
# Example:
# python -m train.merge_checkpoint --adapter_path /home/zchu/codes/train_2412/train/outputs/dr_distillation/mistral_2wikimqa_decomp_bs32_lr4e-5_ep2_1204-1501-Tds/checkpoint-3033-ep2 --overwrite
# python -m train.merge_checkpoint --adapter_path /home/zchu/codes/train_2412/train/outputs/dr_distillation/mistral_2wikimqa_responser_bs32_lr4e-5_ep2_1204-1606-K3i/checkpoint-1065-ep2 --overwrite
