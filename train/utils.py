import os
import sys
import json
import math
import torch
import random
import string
import logging
import inspect
from datetime import datetime

from types import MethodType
from typing import List, Tuple, Any, Dict, Optional, Union
from copy import deepcopy
from functools import partial

from peft import PeftModelForCausalLM, LoraConfig, TaskType, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import PreTrainedTokenizer, PretrainedConfig, PreTrainedModel
from torch.utils.data import Dataset
from safetensors.torch import save_file, load_file

## logger
def get_logger(name: str) -> logging.Logger:
    r"Initialize logger"
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S"
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


def load_config(args) -> PretrainedConfig:
    r"Loading model config. Return PretrainedConfig"
    return AutoConfig.from_pretrained(args.model_name_or_path)

def is_bf16_supported():
    # 获取当前设备的计算能力
    major, minor = torch.cuda.get_device_capability()
    compute_capability = major + minor / 10.0
    print(f"当前GPU的计算能力: {compute_capability}")
    
    # 判断是否支持bfloat16
    if compute_capability >= 8.0:
        return True
    else:
        return False

def is_bf16_available():
    return torch.cuda.is_bf16_supported()

def load_model_with_resize_vocab(args, tokenizer: PreTrainedTokenizer) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    r"Loading llm and init lora adapter. Return PretrainedModel"
    config = load_config(args)
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_available() and is_bf16_supported() else torch.float16
    if args.finetuning_type == "full":
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, use_flash_attention_2=is_bf16_supported(), torch_dtype=dtype)
    resize_token_embeddings_with_init(model, tokenizer)
    ## Derived from LLaMA-Factory/src/llamafactory/model/patcher.py/patch_model
    if args.do_train:
        prepare_model_for_training(model, args)
    
    model = init_adapter(model, args)

    if args.do_train:
        model.train()
    else:
        model.requires_grad_(False)
        model.eval()
    
    trainable_params, all_params = count_parameters(model)
    if args.do_train:
        param_stats = "trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
            trainable_params, all_params, 100 * trainable_params / all_params
        )
    else:
        param_stats = "all params: {:d}".format(all_params)
    
    logger.info(param_stats)
    return model, tokenizer

def load_model(args) -> PreTrainedModel:
    r"Loading llm and init lora adapter. Return PretrainedModel"
    config = load_config(args)
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    if args.finetuning_type == "full":
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, use_flash_attention_2=is_bf16_supported(), torch_dtype=dtype)
    ## Derived from LLaMA-Factory/src/llamafactory/model/patcher.py/patch_model
    if args.do_train:
        prepare_model_for_training(model, args)
    
    model = init_adapter(model, args)

    if args.do_train:
        model.train()
    else:
        model.requires_grad_(False)
        model.eval()
    
    trainable_params, all_params = count_parameters(model)
    if args.do_train:
        param_stats = "trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
            trainable_params, all_params, 100 * trainable_params / all_params
        )
    else:
        param_stats = "all params: {:d}".format(all_params)
    
    logger.info(param_stats)
    return model
    
## Tool function for loading peft models
def _find_all_linear_module(model: PreTrainedModel) -> List[str]:
    r"""
    Finds all available lora modules.
    """
    forbidden_modules = {"lm_head"}
    module_names = set()
    for name, module in model.named_modules():
        if any(forbidden_module in name for forbidden_module in forbidden_modules):
            continue

        if "Linear" in module.__class__.__name__ and "Embedding" not in module.__class__.__name__:
            module_names.add(name.split(".")[-1])
    logger.info("Found linear modules: {}".format(",".join(module_names)))
    return list(module_names)
    

def _setup_full_tuning(model, args):
    # We do not do anything.
    pass

def _setup_lora_tuning(model: PreTrainedModel, args) -> PeftModel:
    r"""目前只支持初始化新的LoRA Adapter，不支持加载已经训练好的adapter上继续训练。
    (todo..)"""
    target_modules = _find_all_linear_module(model=model)
    
    if args.resize_vocab:
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()
        module_names = set()
        for name, module in model.named_modules():
            if module in [input_embeddings, output_embeddings]:
                module_names.add(name.split(".")[-1])
        args.additional_target = None
        # If add lm_head and embed_token to LoRA target, adapter will be added to embeddings layer.
        #target_modules = target_modules + list(module_names)
        logger.warning("Vocab has been resized, add {} to trainable parameters.".format(",".join(module_names)))
    else:
        args.additional_target = None
    
    if args.adapter_name_or_path is None:    
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            use_rslora=False,
            use_dora=False,
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=args.additional_target, #这个需要调整，因为如果要改词表的话，Embeddings和LM-head需要调整。
            target_modules=target_modules
        )
        lora_config.base_model_name_or_path = args.model_name_or_path
        model = PeftModelForCausalLM(model=model, peft_config=lora_config, adapter_name="default")
        
        # If the vocab is resized, the input embeddings and output embeddings (lm_head) need to be trainable.
        if args.resize_vocab:
            for name, param in model.named_parameters():
                if "embed_tokens" in name or "lm_head" in name:
                    param.requires_grad = True
    else:
        model = PeftModel.from_pretrained(model, args.adapter_name_or_path)
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            if "embed_tokens" in name or "lm_head" in name:
                param.requires_grad = True
    
    for param in filter(lambda p: p.requires_grad, model.parameters()):
        param.data = param.data.to(torch.float32)
    
    # pytorch的张量是以完整tensor为整体的，因此没有办法让完整的tensor其中一部分需要梯度，只能在forward的过程中forward两次，将其中的一次进行detach之后进行拼接。因此暂时先不实现这个了，让所有的embeddings参数都参与训练。
    return model
    

def init_adapter(model: PreTrainedModel, args) -> PreTrainedModel:
    r"Initializes lora adapter if needs"
    is_trainable = args.do_train
    
    if (not is_trainable) and args.adapter_name_or_path is None: 
        logger.info("Adapter is not found at evaluation, load the base model.")
        return model
    
    if is_trainable and args.finetuning_type == "full": # This branch hasn't been test yet. Do test before using.
        _setup_full_tuning(model, args)
        return model
    elif is_trainable and args.finetuning_type == "lora":
        model = _setup_lora_tuning(model, args)
        return model
    else:
        raise NotImplementedError()    

def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by 2
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "quant_storage") and hasattr(param.quant_storage, "itemsize"):
                num_bytes = param.quant_storage.itemsize
            elif hasattr(param, "element_size"):  # for older pytorch version
                num_bytes = param.element_size()
            else:
                num_bytes = 1

            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param

## prepare model for training

def prepare_model_for_training(model: PreTrainedModel, args, output_layer_name="lm_head"):
    r"""
    Includes:
        (1) cast the layernorm in fp32
        (2) make output embedding layer require grads (make lm_head trainable)
        (3) add the upcasting of the lm_head in fp32
        (4) enable gradient checkpointing
    Inspired by: https://github.com/huggingface/peft/blob/v0.7.1/src/peft/utils/other.py#L72
    """
    # upcast layernorm to fp32
    if True: 
        logger.info("Upcasting layernorm weights in float32.")
        for name, param in model.named_parameters():           
            if param.ndim == 1 and any(ln_name in name for ln_name in ["norm", "ln"]):
                param.data = param.data.to(torch.float32)
    
    # gradient checkpointing
    if True:
        if not getattr(model, "supports_gradient_checkpointing", False):
            logger.warning("Current model does not support gradient checkpointing.")
        else:
            model.gradient_checkpointing_enable = MethodType(_gradient_checkpointing_enable, model)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
            setattr(model.config, "use_cache", False)  # turn off when gradient checkpointing is enabled
            logger.info("Gradient checkpointing enabled.")
    
    # upcast lm_head to fp32
    if True and hasattr(model, output_layer_name):
        logger.info("Upcasting lm_head outputs in float32.")
        output_layer = getattr(model, output_layer_name)
        if isinstance(output_layer, torch.nn.Module) and output_layer.weight.dtype != torch.float32:
            output_layer.register_forward_hook(_fp32_forward_post_hook)
        pass

def _fp32_forward_post_hook(
    module: "torch.nn.Module", args: Tuple["torch.Tensor"], output: "torch.Tensor"
) -> "torch.Tensor":
    return output.to(torch.float32)

def _gradient_checkpointing_enable(
    self: "PreTrainedModel", gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None
) -> None:
    r"""
    Activates gradient checkpointing for the current model.

    Modification of the original method to enable gradient checkpointing for block-wise optimizer.
    """
    from torch.utils.checkpoint import checkpoint

    if not self.supports_gradient_checkpointing:
        raise ValueError("{} does not support gradient checkpointing.".format(self.__class__.__name__))

    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {"use_reentrant": True}

    gradient_checkpointing_func = partial(checkpoint, **gradient_checkpointing_kwargs)

    def custom_gradient_checkpointing_func(func, *args, **kwargs):
        module: "torch.nn.Module" = func.__self__

        if any(param.requires_grad for param in module.parameters()):
            for arg in args:
                if torch.is_tensor(arg) and torch.is_floating_point(arg):
                    arg.requires_grad_(True)

        return gradient_checkpointing_func(func, *args, **kwargs)

    if "value" in inspect.signature(self._set_gradient_checkpointing).parameters:  # old GC format
        self.apply(partial(self._set_gradient_checkpointing, value=True))
        self.enable_input_require_grads()
        logger.warning("You are using the old GC format, some features (e.g. BAdam) will be invalid.")
    else:  # have already enabled input require gradients
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=custom_gradient_checkpointing_func)

## Add Special Tokens
def resize_token_embeddings_with_init(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, init=False):
    added_special_tokens = [
        "<sub-question>", "</sub-question>", "<paragraph>", "</paragraph>",
        "[Atomic Question]", "[Non-Atomic Question]", "[Question]", "[Remaining Question]", "[Final Answer]",
        "[Relevant]", "[Irrelevant]",
        "[Fully supported]", "[Partially supported]", "[Not supported]",
        "[4]", "[3]", "[2]", "[1]", "[0]",
        "[Partially Relevant]"
    ]
    tokenized_special_tokens = {
        k : tokenizer.encode(k, add_special_tokens=False) for k in added_special_tokens
    } 
    num_new_tokens = tokenizer.add_special_tokens({
        "additional_special_tokens": added_special_tokens
    })
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    
    
    if num_new_tokens <= 0:
        return 
    if init:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        for special_token, special_token_ids_before in tokenized_special_tokens.items():
            special_token_ids_after = tokenizer.encode(special_token, add_special_tokens=False)
            assert len(special_token_ids_after) == 1
            print(f"special_token: {special_token}, special_token_ids_before: {special_token_ids_before}, special_token_ids_after: {special_token_ids_after}")
            average_input_embedding = input_embeddings[special_token_ids_before].mean(dim=0, keepdim=True)
            average_output_embedding = output_embeddings[special_token_ids_before].mean(dim=0, keepdim=True)

            input_embeddings[special_token_ids_after[0]] = average_input_embedding
            output_embeddings[special_token_ids_after[0]] = average_output_embedding
    else:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
    return 
    


def calculate_lr_scheduler(args, per_epoch_data_num):
    r"Get learning rate scheduler related information. Return Dict{num_training_steps, num_warmup_steps, total_batch_size}"
    if not torch.cuda.is_available():
        device_count = 1
    else:
        device_count = torch.cuda.device_count()
    per_device_batch_size = args.train_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    #total_batch_size = device_count * per_device_batch_size * gradient_accumulation_steps
    total_batch_size = per_device_batch_size * gradient_accumulation_steps
    total_data_num = per_epoch_data_num * args.num_train_epochs
    train_steps = math.ceil(total_data_num / total_batch_size)
    
    if args.warmup_steps is not None:
        warmup_steps = args.warmup_steps
    else:
        warmup_steps = math.ceil(train_steps * args.warmup_ratio)
    return {
        "num_training_steps" : train_steps,
        "num_warmup_steps" : warmup_steps,
        "total_batch_size" : total_batch_size
    }




MISTRAL_INST_TOKEN_IDS = [733, 16289, 28793]
MISTRAL_EINST_TOKEN_IDS = [733, 28748, 16289, 28793]
LLAMA2_INST_TOKEN_IDS = [518, 25580, 29962]
LLAMA2_EINST_TOKEN_IDS = [518, 29914, 25580, 29962]
LLAMA3_INST_TOKEN_IDS = [58, 65562, 60]
LLAMA3_EINST_TOKEN_IDS = [66028, 65562, 60]
QWEN25_INST_TOKEN_IDS = [58, 64462, 60]
QWEN25_EINST_TOKEN_IDS = [64928, 64462, 60]


CHAT_TEMPLATE = "[INST] {} [/INST] {}"

BEGIN_OF_INSTRUCTION = {
    "llama2" : LLAMA2_INST_TOKEN_IDS,
    "llama3" : LLAMA3_INST_TOKEN_IDS,
    "mistral" : MISTRAL_INST_TOKEN_IDS,
    "qwen25" : QWEN25_INST_TOKEN_IDS
}

END_OF_INSTRUCTION = {
    "llama2" : LLAMA2_EINST_TOKEN_IDS,
    "llama3" : LLAMA3_EINST_TOKEN_IDS,
    "mistral" : MISTRAL_EINST_TOKEN_IDS,
    "qwen25" : QWEN25_EINST_TOKEN_IDS
}

SQS, SQE, PS, PE = "<sub-question>", "</sub-question>", "<paragraph>", "</paragraph>"
ATOM, NATOM = "[Atomic Question]", "[Non-Atomic Question]"
REL, IRREL = "[Relevant]", "[Irrelevant]"
Q, RQ, A = "[Question]", "[Remaining Question]", "[Final Answer]"
# Use customized chat template: {} prompt {} completion
class SFTDatasetNew(Dataset):
    """
    Implement the instruction-output attention mask.
    """
    def __init__(self, datas, tokenizer: PreTrainedTokenizer, args):
        self.datas = datas
        self.tokenizer = tokenizer
        self.args = args

    def __len__(self):
        return len(self.datas)

    def get_mask(self, input_ids):
        if self.args.model == "llama2":
            pass
        elif self.args.model == "llama3":
            pass
        elif self.args.model == "mistral":
            pass
        elif self.args.model in ["qwen2", "qwen25"]:
            pass
        else:
            raise NotImplementedError(f"{self.args.model} is not supported currently.")
        
        boi_length = len(BEGIN_OF_INSTRUCTION[self.args.model])
        eoi_length = len(END_OF_INSTRUCTION[self.args.model])
        boi_pos = 0
        eoi_pos = len(input_ids) - 1
        for i in range(len(input_ids)):
            if input_ids[i:i+boi_length] == BEGIN_OF_INSTRUCTION[self.args.model]:
                boi_pos = i + boi_length
            if input_ids[i:i+eoi_length] == END_OF_INSTRUCTION[self.args.model]:
                eoi_pos = i + eoi_length
        # input_ids[bos_pos] 是prompt的第一个token
        # input_ids[eoi_pos] 是completion的第一个token
        # 通常来说，需要训练的位置为 [eoi_pos:]
        return boi_pos, eoi_pos  

## Data format for oracle baseline
class OracleSFTDataset(SFTDatasetNew):
    def __init__(self, datas, tokenizer: PreTrainedTokenizer, args):
        super().__init__(datas, tokenizer, args)
    
    def __getitem__(self, idx):
        data = self.datas[idx]
        instruction = data["instruction"]
        output = data["output"]
        input_sentence = CHAT_TEMPLATE.format(instruction, output)
        input_ids = self.tokenizer.encode(input_sentence, add_special_tokens=False)
        if self.tokenizer.bos_token_id is not None:
            input_ids = [self.tokenizer.bos_token_id] + input_ids + [self.tokenizer.eos_token_id]
        else:
            input_ids = input_ids + [self.tokenizer.eos_token_id]
        boi_pos, eoi_pos = self.get_mask(input_ids)
        labels = input_ids.copy()
        labels = [-100 if i < eoi_pos else _ for i, _ in enumerate(labels)]
        
        return input_ids, labels
## Data format for DPO
# prompt: str
# chosen: str
# rejected: str

class PreferenceDataset(SFTDatasetNew):
    def __init__(self, datas, tokenizer: PreTrainedTokenizer, args):
        super().__init__(datas, tokenizer, args)
    
    def _mask_context(self, input_ids, labels, context_markups):
        i = 0
        while i < len(input_ids):
            if input_ids[i] == context_markups[0]: # start position of one match
                start_idx = i + 1
                end_idx = None
                for j in range(start_idx, len(input_ids)):
                    if input_ids[j] == context_markups[1]:
                        end_idx = j + 1
                        break
                if end_idx is not None:
                    labels[start_idx:end_idx] = [-100] * (end_idx - start_idx)
                i = end_idx + 1 if end_idx is not None else len(input_ids)
            else:
                i += 1
        return labels
    
    def __getitem__(self, idx):
        data = self.datas[idx]
        instruction, chosen_data, rejected_data = data["instruction"], data["chosen"], data["rejected"]
        input_chosen = CHAT_TEMPLATE.format(instruction, chosen_data)
        input_rejected = CHAT_TEMPLATE.format(instruction, rejected_data)
        input_chosen_ids = self.tokenizer.encode(input_chosen, add_special_tokens=False)
        input_rejected_ids = self.tokenizer.encode(input_rejected, add_special_tokens=False)
        input_chosen_ids = [self.tokenizer.bos_token_id] + input_chosen_ids + [self.tokenizer.eos_token_id]
        input_rejected_ids = [self.tokenizer.bos_token_id] + input_rejected_ids + [self.tokenizer.eos_token_id]
        labels_chosen = input_chosen_ids.copy()
        labels_rejected = input_rejected_ids.copy()
        _, eoi_chosen = self.get_mask(input_chosen_ids)
        _, eoi_rejected = self.get_mask(input_rejected_ids)
        labels_chosen = [-100 if i < eoi_chosen else _ for i, _ in enumerate(labels_chosen)]
        labels_rejected = [-100 if i < eoi_rejected else _ for i, _ in enumerate(labels_rejected)]
        labels_chosen = self._mask_context(input_chosen_ids, labels_chosen, [PS, PE])
        labels_rejected = self._mask_context(input_rejected_ids, labels_rejected, [PS, PE])
        return input_chosen_ids, labels_chosen, input_rejected_ids, labels_rejected
        


##　Data format for critic input datas
# input: str
# completion (rating): str
# completion (explanation): str
## Training mode for critic model
# 1. direct: input -> rating
# 2. pre_cot: input -> explanation, rating (recommended)
# 3. post_cot: input -> rating, explanation

class OursCriticSFTDataset(SFTDatasetNew):
    def __init__(self, datas, tokenizer: PreTrainedTokenizer, args):
        super().__init__(datas, tokenizer, args)
        self.training_mode = args.training_mode
    
    def __getitem__(self, idx):
        data = self.datas[idx]
        instruction = data["instruction"]
        rating = data["output"]["rating"]
        explanation = data["output"]["explanation"]
        if self.training_mode == "direct":
            output = rating
        elif self.training_mode == "pre_cot":
            output = f"Explanation: {explanation}\nRating: {rating}"
        elif self.training_mode == "post_cot":
            output = f"Rating: {rating}\nExplanation: {explanation}"
        else:
            raise NotImplementedError(f"{self.training_mode} is not supported currently.")

        input_sentence = CHAT_TEMPLATE.format(instruction, output)
        input_ids = self.tokenizer.encode(input_sentence, add_special_tokens=False)
        if self.tokenizer.bos_token_id is not None:
            input_ids = [self.tokenizer.bos_token_id] + input_ids + [self.tokenizer.eos_token_id]
        else:
            input_ids = input_ids + [self.tokenizer.eos_token_id]
        boi_pos, eoi_pos = self.get_mask(input_ids)
        
        labels = input_ids.copy()
        labels = [-100 if i < eoi_pos else _ for i, _ in enumerate(labels)]
        return input_ids, labels
    
class OursT2SFTDataset(SFTDatasetNew):
    def __init__(self, datas, tokenizer: PreTrainedTokenizer, args):
        super().__init__(datas, tokenizer, args)

    def __getitem__(self, idx):
        data = self.datas[idx]
        instruction = data["instruction"]
        output = data["output"]
        input_sentence = CHAT_TEMPLATE.format(instruction, output)
        input_ids = self.tokenizer.encode(input_sentence, add_special_tokens=False)
        if self.tokenizer.bos_token_id is not None:
            input_ids = [self.tokenizer.bos_token_id] + input_ids + [self.tokenizer.eos_token_id]
        else:
            input_ids = input_ids + [self.tokenizer.eos_token_id]
        boi_pos, eoi_pos = self.get_mask(input_ids)
        context_markups = self.tokenizer.convert_tokens_to_ids([PS, PE])    
        
        # get labels through mask on input_ids
        labels = input_ids.copy()
        labels = [-100 if i < eoi_pos else _ for i, _ in enumerate(labels)]
        
        # mask the retrieval context
        i = 0
        while i < len(input_ids):
            if input_ids[i] == context_markups[0]: # start position of one match
                start_idx = i + 1
                end_idx = None
                for j in range(start_idx, len(input_ids)):
                    if input_ids[j] == context_markups[1]:
                        end_idx = j + 1
                        break
                if end_idx is not None:
                    labels[start_idx:end_idx] = [-100] * (end_idx - start_idx)
                i = end_idx + 1 if end_idx is not None else len(input_ids)
            else:
                i += 1
                
        return input_ids, labels


class DecomposerSFTDataset(SFTDatasetNew):
    def __init__(self, datas, tokenizer: PreTrainedTokenizer, args):
        super().__init__(datas, tokenizer, args)
    
    def __getitem__(self, idx):
        data = self.datas[idx]
        instruction = data["instruction"]
        output = data["output"]
        input_sentence = CHAT_TEMPLATE.format(instruction, output)
        input_ids = self.tokenizer.encode(input_sentence, add_special_tokens=False)
        if self.tokenizer.bos_token_id is not None:
            input_ids = [self.tokenizer.bos_token_id] + input_ids + [self.tokenizer.eos_token_id]
        else:
            input_ids = input_ids + [self.tokenizer.eos_token_id]
        boi_pos, eoi_pos = self.get_mask(input_ids)
        
        # get labels through mask on input_ids
        labels = input_ids.copy()
        labels = [-100 if i < eoi_pos else _ for i, _ in enumerate(labels)]
        return input_ids, labels
        
class ResponserSFTDataset(SFTDatasetNew):
    def __init__(self, datas, tokenizer: PreTrainedTokenizer, args):
        super().__init__(datas, tokenizer, args)
    
    def __getitem__(self, idx):
        data = self.datas[idx]
        instruction = data["instruction"]
        output = data["output"]
        input_sentence = CHAT_TEMPLATE.format(instruction, output)
        input_ids = self.tokenizer.encode(input_sentence, add_special_tokens=False)
        if self.tokenizer.bos_token_id is not None:
            input_ids = [self.tokenizer.bos_token_id] + input_ids + [self.tokenizer.eos_token_id]
        else:
            input_ids = input_ids + [self.tokenizer.eos_token_id]
        boi_pos, eoi_pos = self.get_mask(input_ids)
        
        # get labels through mask on input_ids
        labels = input_ids.copy()
        labels = [-100 if i < eoi_pos else _ for i, _ in enumerate(labels)]
        return input_ids, labels
          

## loadding data, saveing ckpt

DATASET_MAP = {
    "gpt-4o-mini" : {
        "decomposer" : "/home/zchu/codes/train_2412/data_generation/dr_distillation/outputs/datasets/2wikimqa_dr_gpt-4o-mini_f1_70/decomposer.json",
        "responser" : "/home/zchu/codes/train_2412/data_generation/dr_distillation/outputs/datasets/2wikimqa_dr_gpt-4o-mini_f1_70/responser.json"
    },
    "deepseek" : {
        "decomposer" : None,
        "responser" : None
    }
}

def load_datas(args):
    # 已经不用了
    fp = DATASET_MAP[args.teacher_model][args.role]
    datas = json.load(open(fp))
    if args.role == "decomposer":
        logger.info(f"Loading Decomposer Dataset: {args.teacher_model}_{args.role}, Path: {fp}")
        logger.info(f"Dataset Infos: Num of datas: {len(datas)}")
    elif args.role == "responser":
        logger.info(f"Loading Responser Dataset: {args.teacher_model}_{args.role}, Path: {fp}")
        logger.info(f"Only keep datas with words less 200")
        logger.info(f"Dataset Infos: Num of datas (Before filter): {len(datas)}")
        datas = [
            item for item in datas if
            len(f"{item['instruction']} {item['output']}".split()) < 200
        ]
        logger.info(f"Dataset Infos: Num of datas (After filter): {len(datas)}")
    return datas

def save_merged_checkpoint(llm, tokenizer, args, accelerator=None):
    llm = llm.cpu()
    save_dir = f"{args.output_dir}/merged_checkpoint"
    if isinstance(llm, PeftModel):
        llm = llm.merge_and_unload()
    elif isinstance(llm, PreTrainedModel):
        pass
    elif isinstance(accelerator.unwrap_model(llm), PeftModel):
        llm = accelerator.unwrap_model(llm).merge_and_unload()
    elif isinstance(accelerator.unwrap_model(llm), PreTrainedModel):
        llm = accelerator.unwrap_model(llm)
    else:
        raise ValueError(f"Unsupported model type: {type(llm)}")
    tokenizer.save_pretrained(save_dir)
    llm.save_pretrained(save_dir)
    logger.info(f"Saving merged checkpoints to {save_dir}")
    

def save_checkpoint(llm, tokenizer: PreTrainedTokenizer, logs, args, steps, epoch=None, final=False, accelerator=None):
    if final is True:
        save_dir = args.output_dir
    else:
        save_dir = f"{args.output_dir}/checkpoint-{steps}"
        if epoch is not None:
            save_dir += f"-ep{epoch + 1}"
    if isinstance(llm, (PreTrainedModel, PeftModel)):
        llm.save_pretrained(save_dir)
        embedding_state_dict = {
            "lm_head.weight" : llm.get_output_embeddings().weight,
            "model.embed_tokens.weight" : llm.get_input_embeddings().weight
        }
    elif isinstance(accelerator.unwrap_model(llm), (PreTrainedModel, PeftModel)):
        accelerator.unwrap_model(llm).save_pretrained(save_dir)
        embedding_state_dict = {
            "lm_head.weight" : accelerator.unwrap_model(llm).get_output_embeddings().weight,
            "model.embed_tokens.weight" : accelerator.unwrap_model(llm).get_input_embeddings().weight
        }
    save_file(embedding_state_dict, f"{save_dir}/embeddings.safetensors")
    
    tokenizer.save_pretrained(save_dir)
    with open(f"{args.output_dir}/train_log.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(logs, indent=2))
    
    logger.info(f"Saving checkpoint to {save_dir}")
    
def get_random_suffix(n=2):
    # 获取当前时间戳，精度到分钟
    timestamp = datetime.now().strftime("%m%d-%H%M")
    chars = string.ascii_letters + string.digits
    random_str = "".join(random.choices(chars, k=n))
    # 组合时间戳和随机字符串
    return f"{timestamp}-{random_str}"

def infer_model_name(model_name_or_path):
    model_name = model_name_or_path.split(os.sep)[-1]
    model_name = model_name.lower().replace("-", "").replace("_", "")
    if "llama2" in model_name:
        return "llama2"
    elif "llama3" in model_name:
        return "llama3"
    elif "mistral" in model_name:
        return "mistral"
    elif "qwen2.5" in model_name:
        return "qwen25"
    elif "qwen" in model_name:
        return "qwen2"
    else:
        return model_name

logger = get_logger(__name__)
