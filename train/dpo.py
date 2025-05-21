# conda environment: llama_factory_810
# create date: 2024/12/24
# This file is used for preference tuning DPO.
import os
import json
import time
import torch
import wandb
import string
import random
import datetime
import argparse
import numpy as np
from collections import defaultdict
from itertools import chain
from tqdm import tqdm
from copy import deepcopy
from contextlib import contextmanager
from typing import Dict, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import set_seed
from accelerate import Accelerator
from peft import PeftModel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn import functional as F

from train.utils import (
    load_model, load_datas, 
    calculate_lr_scheduler, save_checkpoint, save_merged_checkpoint,
    get_logger, get_random_suffix, infer_model_name,
    load_model_with_resize_vocab, is_bf16_supported,
    PreferenceDataset
)

logger = get_logger(__name__)
os.environ["WANDB_MODE"] = "offline"

def parse_args():
    parser = argparse.ArgumentParser()
    ## Model Args
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--adapter_name_or_path", type=str, default=None, help="Only set if loaded from a lora tuned checkpoint.")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--finetuning_type", type=str, choices=["lora"], default="lora")
    parser.add_argument("--lora_target", type=str, nargs="+", default="all")
    parser.add_argument("--lora_rank", type=int, help="LoRA Adapter rank.")
    parser.add_argument("--lora_alpha", type=int, help="LoRA Adapter alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA Adapter dropout.")
    parser.add_argument("--dtype", type=str, default="auto", choices=["fp16", "bf16", "auto"], help="Mix precision.")
    parser.add_argument("--resize_vocab", action="store_true", help="Add special tokens and resize tokenizer, embeddings and lm_head.")
    
    ## Train Args
    parser.add_argument("--stage", type=str, default="dpo", choices=["dpo"])
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=999999999, help="Max training samples, set to a small number when debugging.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=999999)
    parser.add_argument("--save_steps", type=int, default=999999)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=0.02, help="By default, we use 0.02*steps for warmup.")
    parser.add_argument("--optim", type=str, default="adamw_torch", choices=["adamw_torch"])
    parser.add_argument("--eval_on_epoch_end", action="store_true", help="Eval on epoch ends. Independent from eval_steps, default is False.")
    parser.add_argument("--save_on_epoch_end", action="store_true", help="Save ckpt on epoch end. Independent from save_steps, default is False.")
    parser.add_argument("--eval_on_training_end", action="store_true", help="Eval on training procedure done. Default is False.")
    parser.add_argument("--save_on_training_end", action="store_true", help="Save on training procedure done. Default is False.")
    parser.add_argument("--train_batch_size", type=int, help="Per device batch size in training.")
    parser.add_argument("--eval_batch_size", type=int, help="Per device batch size in evaluating.", default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Update parameters after every x steps.")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing for DPO.")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--ftx_gamma", type=float, default=0.0)
    
    ## Data Args
    parser.add_argument("--train_data_path", type=str, default=None)
    
    ## Other Args
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--project_name", type=str, help="Project name of wandb.", default=None)
    parser.add_argument("--group_name", type=str, help="Group name of wandb.", default=None)
    parser.add_argument("--run_name", type=str, help="Run name of wandb.", default=None)

    args = parser.parse_args()
    args.model = infer_model_name(args.model_name_or_path.replace("/merged_checkpoint", ""))
    args.n_gpu = torch.cuda.device_count()    
    if args.tokenizer is None:
        args.tokenizer = args.model_name_or_path
    
    args.output_dir = "./train/outputs/t2_dpo/" + args.output_dir + "_" + get_random_suffix(n=3)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    with open(f"{args.output_dir}/args.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(vars(args), indent=4))
    
    return args


def get_batch_logps(logits: "torch.Tensor", labels: "torch.Tensor", label_pad_token_id: int = -100):
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batchsize x seqlen) and labels must have the same shape.")
    
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id
    labels[labels == label_pad_token_id] = 0  # dummy token
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)

def dpo_loss(
    accelerator: "Accelerator",
    policy_chosen_logps: "torch.Tensor",
    policy_rejected_logps: "torch.Tensor",
    reference_chosen_logps: "torch.Tensor",
    reference_rejected_logps: "torch.Tensor"
):
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    pi_logratios = pi_logratios.to(accelerator.device)
    ref_logratios = ref_logratios.to(accelerator.device)
    logits = pi_logratios - ref_logratios

    ## Compute sigmoid loss
    losses = (
        -F.logsigmoid(args.beta * logits) * (1 - args.label_smoothing)
        - F.logsigmoid(-args.beta * logits) * args.label_smoothing
    )

    ## Compute chosen/rejected rewards (This is equal when starting trianing)
    chosen_rewards = (
        args.beta
        * (
            policy_chosen_logps.to(accelerator.device) - reference_chosen_logps.to(accelerator.device)
        ).detach()
    )
    rejected_rewards = (
        args.beta
        * (
            policy_rejected_logps.to(accelerator.device)
            - reference_rejected_logps.to(accelerator.device)
        ).detach()
    )
    
    return losses, chosen_rewards, rejected_rewards

@contextmanager
def get_ref_context(accelerator: "Accelerator", model: "PreTrainedModel"):
    with accelerator.unwrap_model(model).disable_adapter():
        model.eval()
        yield
        model.train()

def dpo(args):
    ## Init accelerator
    accelerator = Accelerator(
        cpu=False, 
        mixed_precision=args.dtype if args.dtype != "auto" else (
            "bf16" if is_bf16_supported() else "fp16"
        )
    )
    if accelerator.is_main_process:
        logger.info(json.dumps(args.__dict__, indent=4))
    
    ## Init wandb
    project_name, group_name, run_name = args.project_name, args.group_name, args.run_name
    if all([project_name, group_name, run_name]):
        args.log_to_wandb = True
        if accelerator.is_main_process: #only init wandb run in main-process
            wandb.init(
                project=project_name,
                group=group_name,
                name=run_name
            )
    else:
        args.log_to_wandb = False
    
    ## Init tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning(f"Tokenizer {tokenizer} does not have pad token. We set tokenizer.pad_token = tokenizer.eos_token.")
    tokenizer.padding_side = "right"
    
    llm, tokenizer = load_model_with_resize_vocab(args, tokenizer)
      
    ## collate function
    def collate_fn_train(examples):
        input_ids_chosen = [_[0] for _ in examples]
        input_ids_rejected = [_[2] for _ in examples]
        input_ids = input_ids_chosen + input_ids_rejected
        labels_chosen = [_[1] for _ in examples] 
        labels_rejected = [_[3] for _ in examples] 

        max_length = max([
            max([len(_) for _ in input_ids_chosen]),
            max([len(_) for _ in input_ids_rejected])
        ])

        if tokenizer.padding_side == "right":
            labels_chosen = [_ + [-100] * (max_length - len(_)) for _ in labels_chosen]
            labels_rejected = [_ + [-100] * (max_length - len(_)) for _ in labels_rejected]
            labels = labels_chosen + labels_rejected
        elif tokenizer.padding_side == "left":
            labels_chosen = [[-100] * (max_length - len(_)) + _ for _ in labels_chosen]
            labels_rejected = [[-100] * (max_length - len(_)) + _ for _ in labels_rejected]
            labels = labels_chosen + labels_rejected
        else:
            raise NotImplementedError()

        
        batch_encoding = tokenizer.pad(
            {"input_ids" : input_ids},
            padding="longest",
            return_tensors="pt",
            max_length=2568 # This needs a hyperpameter to control
        )
        batch_encoding["labels"] = torch.LongTensor(labels)
        
        return batch_encoding
    
    ## Load dataset
    datas = json.load(open(args.train_data_path))[:args.max_samples]
    logger.info(f"Loaded {len(datas)} training datas from {args.train_data_path}")
    train_dataset = PreferenceDataset(datas, tokenizer, args)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn_train, batch_size=args.train_batch_size, drop_last=True
    )
    
    optimizer = AdamW(params=llm.parameters(), lr=args.learning_rate)
    lr_scheduler_info = calculate_lr_scheduler(args=args, per_epoch_data_num=len(train_dataset)) #[TO BE ADD]
    if args.lr_scheduler_type == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, \
            num_warmup_steps=lr_scheduler_info["num_warmup_steps"], num_training_steps=lr_scheduler_info["num_training_steps"])
    elif args.lr_scheduler_type == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, \
            num_warmup_steps=lr_scheduler_info["num_warmup_steps"], num_training_steps=lr_scheduler_info["num_training_steps"])
    
    if args.seed > 0:
        set_seed(args.seed)
    
    llm, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        llm, optimizer, train_dataloader, lr_scheduler
    )

    ## Training Loops
    if accelerator.is_main_process:
        pbar = tqdm(total=lr_scheduler_info["num_training_steps"]//args.n_gpu, desc="Training", leave=False)
        
    overall_step = 0
    total_loss = []
    total_grad_norm = []
    total_logs = []
    total_attn_loss = []
    for epoch in range(args.num_train_epochs):
        llm.train()
        accelerator.free_memory()
        # Inner training loops
        for step, batch in enumerate(train_dataloader):
            def concatenated_forward(batch):
                batch = {k: v.detach() for k, v in batch.items()}
                all_logits: "torch.Tensor" = llm(**batch, return_dict=True, use_cache=False).logits.to(torch.float32) #[2*bs, seq_len, vocab_size]
                all_logps, valid_length = get_batch_logps(all_logits, batch["labels"])
                # if args.loss_type in ["ipo", "orpo", "simpo"]:
                #     all_logps = all_logps / valid_length
                    
                batch_size = batch["input_ids"].shape[0] // 2
                chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
                chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
                chosen_length, rejected_length = valid_length.split(batch_size, dim=0)
                average_chosen_logps = chosen_logps / chosen_length
                return chosen_logps, rejected_logps, chosen_logits, rejected_logits, average_chosen_logps

            ## Compute logits and logprobs for policy model
            policy_chosen_logps, policy_rejected_logps, policy_chosen_logits, policy_rejected_logits, policy_chosen_logps_avg = concatenated_forward(batch)

            ## Compute logits and logprobs for reference model
            with torch.no_grad(), get_ref_context(accelerator, llm):
                reference_chosen_logps, reference_rejected_logps, *_ = concatenated_forward(batch)
            
            dpo_losses, chosen_rewards, rejected_rewards = dpo_loss(
                accelerator,
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps
            )
            sft_loss = - policy_chosen_logps_avg
            if args.ftx_gamma > 1e-6:
                dpo_losses += args.ftx_gamma * sft_loss
            
            reward_accuracies = (chosen_rewards > rejected_rewards).float()
            loss = dpo_losses.mean()
            
            
            total_loss.append(loss.detach().item())
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
        
            # Step end
            if (args.gradient_accumulation_steps == 1) \
                or (step != 0 and (step + 1) % args.gradient_accumulation_steps == 0) \
                or (step == len(train_dataloader) - 1):
                _grad_norm = accelerator.clip_grad_norm_(llm.parameters(), args.max_grad_norm)
                total_grad_norm.append(_grad_norm.detach().item() if isinstance(_grad_norm, torch.Tensor) else _grad_norm)
                overall_step += 1
                
                if accelerator.is_main_process:
                    pbar.update(1)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            
            # logging_step
            if (overall_step != 0) and ((overall_step + 1) % args.logging_steps == 0) \
                and (step + 1) % args.gradient_accumulation_steps == 0:
                log_lr = lr_scheduler.get_lr()[0]
                log_loss = round(np.mean(total_loss[-args.logging_steps * args.gradient_accumulation_steps:]), 4)
                #log_epoch = round(float(overall_step / lr_scheduler_info["num_training_steps"]), 4)
                log_epoch = round(epoch + float((step+1) / len(train_dataloader)), 4)
                log_step = overall_step
                log_grad_norm = round(np.mean(total_grad_norm[-args.logging_steps:]), 4)
                log_dict = {
                    "train/epoch" : log_epoch, "train/loss" : log_loss, "train/grad_norm" : log_grad_norm,
                    "train/learning_rate" : log_lr, "train/global_step" : log_step
                }
                if len(total_attn_loss) > 0:
                    log_dict.update({"train/attn_loss" : round(np.mean(total_attn_loss[-args.logging_steps:]), 4)})
                total_logs.append(log_dict)

                if args.log_to_wandb is True and accelerator.is_main_process:
                    logger.info(log_dict)
                    wandb.log(log_dict)
                pass
            
            # saving_step
            if (overall_step != 0) and ((overall_step + 1) % args.save_steps == 0) \
                and (step + 1) % args.gradient_accumulation_steps == 0:
                if accelerator.is_main_process:
                    save_checkpoint(llm, tokenizer, total_logs, args, steps=overall_step+1, epoch=None, final=False, accelerator=accelerator)
                pass
            
            # evaluation step
            if (overall_step != 0) and ((overall_step + 1) % args.eval_steps == 0) \
                and (step + 1) % args.gradient_accumulation_steps == 0 \
                and args.do_eval:
                # Do eval
                pass
            
        # Epoch ends, callback
        if args.eval_on_epoch_end:
            # Do eval
            pass
        
        if args.save_on_epoch_end:
            if accelerator.is_main_process:
                is_final_epoch = (epoch == args.num_train_epochs)
                save_checkpoint(llm, tokenizer, total_logs, args, steps=overall_step+1, epoch=epoch, final=is_final_epoch, accelerator=accelerator)
                pass
    
    # Train ends, callback
    if accelerator.is_main_process:
        save_merged_checkpoint(llm, tokenizer, args, accelerator=accelerator)
    
    if accelerator.is_main_process and args.log_to_wandb:
        wandb.finish()
if __name__ == "__main__":
    args = parse_args()
    dpo(args)
