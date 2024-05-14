# -*- coding: utf-8 -*-

import os
import pathlib
import argparse
from copy import deepcopy
import random
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from transformers import Trainer, BitsAndBytesConfig, deepspeed
from peft import LoraConfig, get_peft_model

from src.utils import get_parameters, rank0_print, save_model_hdfs, safe_save_model_for_hf_trainer
from src.args import ModelArguments, DataArguments, \
    TrainingArguments, LoraArguments, \
        MoEArguments, TokenizerArguments
from src.dataset import SupervisedDataset, UnsupervisedDataset, SemiSupervisedDataset, InstructionTuningDataset
from src.moe.lora_moe import get_lora_moe_model

# 创建自定义回调，用于每个epoch保存模型
class SaveModelAfterEpochCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = state.epoch
        output_dir = os.path.join(self.output_dir, f"epoch-{round(epoch)}")
        control.should_save = True
        self._save_model(output_dir, model)

    def _save_model(self, output_dir, model):
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)

def add_special_tokens(model, tokenizer, tokenizer_args):
    """
    加入自定义的special tokens，可以用于如下两种情况：
    - 下游任务需要special_token来代替标签，比如标签类别很多的任务中，a|b|c|d英文字母不够用；
    - 对特征分桶得到离散特征后，使用special_token来代替桶特征；
    """
    tokenizer_original = deepcopy(tokenizer)
    if tokenizer_args and tokenizer_args.add_special_tokens:
        special_tokens = dict()
        for k, v in tokenizer_args.special_tokens.items():
            if v: special_tokens[k] = v
        tokenizer.add_special_tokens(special_tokens)
        rank0_print("Add special_tokens: {}".format(special_tokens))
        model.resize_token_embeddings(len(tokenizer))

        # 对于新加入的token，可以使用已有的token平均来对其进行初始化
        # refer to https://stackoverflow.com/questions/71443134/adding-new-vocabulary-tokens-to-the-models-and-saving-it-for-downstream-model
        additional_special_tokens = special_tokens.get("additional_special_tokens", [])
        if tokenizer_args.initialize_additional_special_tokens and additional_special_tokens:
            rank0_print("Initializing added special tokens.")
            embedding_weights = model.get_input_embeddings().weight
            with torch.no_grad():
                added_token_emb = []
                for added_token in additional_special_tokens:
                    added_token_ids = tokenizer_original(added_token)["input_ids"]
                    added_token_weights = embedding_weights[added_token_ids]
                    added_token_weights_mean = torch.mean(added_token_weights, axis=0)
                    added_token_emb.append(added_token_weights_mean)
                embedding_weights.data[-len(additional_special_tokens):,:].copy_(
                    torch.vstack(added_token_emb)
                )
    
    return model, tokenizer

def load_model_and_tokenizer(training_args, tokenizer_args, model_args, lora_args):
    model_load_kwargs = {
        "low_cpu_mem_usage": False
    }

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # Load model and tokenizer
    config = transformers.AutoConfig.from_pretrained(
        model_args.pretrain_model_path,
        cache_dir=training_args.cache_dir,
    )
    config.use_cache = False
    if model_args.use_flash_attention:
        print("Using Flash Attention")
        config._attn_implementation = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        model_args.pretrain_model_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        if lora_args and lora_args.use_lora and lora_args.q_lora
        else None,
        **model_load_kwargs,
    )
    model_args.tokenizer_path = model_args.tokenizer_path if model_args.tokenizer_path else model_args.pretrain_model_path
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_path,
        cache_dir=training_args.cache_dir,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )

    # 增加special_tokens和special_tokens初始化
    model, tokenizer = add_special_tokens(model, tokenizer, tokenizer_args)

    # 部分模型的tokenizer缺少bos_token、eos_token、pad_token，需要进行赋值处理，
    # 否则，dataset处理会转化错误。
    # This part will be deprecated in the future, and implemented using `add_special_tokens`
    if config.model_type in ["qwen2", "qwen2_moe"]:
        tokenizer.bos_token = "<|im_start|>"
        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids([tokenizer.bos_token])[0]
    tokenizer.pad_token = tokenizer.pad_token if tokenizer.pad_token else tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
    assert tokenizer.bos_token and tokenizer.eos_token and tokenizer.pad_token

    return model, tokenizer

if __name__ == "__main__":
    # 设置随机种子
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    parser = argparse.ArgumentParser()
    parser.add_argument("--params", help = "parameters yaml file to launch training")
    args = parser.parse_args()
    params = get_parameters(args.params)
    
    # Parse args
    model_args = transformers.HfArgumentParser(ModelArguments).parse_dict(params["model-params"])[0]
    tokenizer_args = transformers.HfArgumentParser(TokenizerArguments).parse_dict(params["tokenizer-params"])[0] if "tokenizer-params" in params else None
    data_args = transformers.HfArgumentParser(DataArguments).parse_dict(params["data-params"])[0]
    training_args = transformers.HfArgumentParser(TrainingArguments).parse_dict(params["trainer-params"])[0]
    lora_args = transformers.HfArgumentParser(LoraArguments).parse_dict(params["lora-params"])[0] if "lora-params" in params else None
    moe_args = transformers.HfArgumentParser(MoEArguments).parse_dict(params["moe-params"])[0] if "moe-params" in params else None
    print("\nParams loaded ...")
    print("==> Model args: {}".format(model_args))
    print("==> Tokenizer args: {}".format(tokenizer_args))
    print("==> Data args: {}".format(data_args))
    print("==> Training args: {}".format(training_args))
    print("==> Lora args: {}".format(lora_args))
    print("==> MoE args: {}".format(moe_args))

    if training_args.wandb_project_name:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project_name

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(training_args, tokenizer_args, model_args, lora_args)
    print("\nModel and tokenizer loaded!")

    # Configure lora
    if lora_args and lora_args.use_lora:
        if tokenizer_args and tokenizer_args.add_special_tokens:
            if int(os.environ['RANK']) == 0 and int(os.environ['LOCAL_RANK']) == 0:
                rpmp = f"{training_args.output_dir}/resized_pretrain_model"
                print(f"saving resized model to {rpmp}")
                model.save_pretrained(rpmp, from_pt=True)
                tokenizer.save_pretrained(rpmp)
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )

        if moe_args and moe_args.use_moe:
            model = get_lora_moe_model(model, lora_args, moe_args)
        else:
            model = get_peft_model(model, lora_config)

            # Print peft trainable params
            model.print_trainable_parameters()

            if training_args.gradient_checkpointing:
                model.enable_input_require_grads()

    # Load dataset
    if data_args.dataset_type == "pretrain":
        train_dataset = UnsupervisedDataset(
            data_path=data_args.train_file,
            tokenizer=tokenizer,
            max_len=data_args.max_len,
            dynamic_length=data_args.dynamic_length
        )
    elif data_args.dataset_type == "continual_pretrain":
        train_dataset = SemiSupervisedDataset(
            data_path=data_args.train_file,
            tokenizer=tokenizer,
            max_len=data_args.max_len,
            dynamic_length=data_args.dynamic_length
        )
    elif data_args.dataset_type == "instruction_tuning":
        train_dataset = InstructionTuningDataset(
            data_path=data_args.train_file,
            tokenizer=tokenizer,
            max_len=data_args.max_len,
            dynamic_length=data_args.dynamic_length,
            model_type=model.config.model_type
        )
    elif data_args.dataset_type == "supervised_finetune":
        train_dataset = SupervisedDataset(
            data_path=data_args.train_file,
            tokenizer=tokenizer,
            max_len=data_args.max_len,
            dynamic_length=data_args.dynamic_length,
            mask_instruction=data_args.mask_instruction,
            padding_side=data_args.padding_side
        )
    else:
        raise ValueError("No such `dataset_type`: `{}`, it should be one of `pretrain`, `continual_pretrain`, `instruction_tuning` or `supervised_finetune`".format(data_args.dataset_type))
    print("\nDataset loaded!")
    
    save_callback = SaveModelAfterEpochCallback(training_args.output_dir)

    # Start trainer
    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=train_dataset, 
        data_collator=train_dataset.collate_fn,
        callbacks=[save_callback]  # 添加自定义回调
    )

    # `not lora_args.use_lora` is a temporary workaround for the issue that there are problems with
    # loading the checkpoint when using LoRA with DeepSpeed.
    # Check this issue https://github.com/huggingface/peft/issues/746 for more information.
    if (
        list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        # and not lora_args.use_lora
    ):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_model()

    safe_save_model_for_hf_trainer(
        trainer=trainer, 
        output_dir=training_args.output_dir, 
        use_lora=lora_args and lora_args.use_lora, 
        bias="none" if not lora_args else lora_args.lora_bias
    )

    ## do the model copy on local rank 0 only
    if int(os.environ['RANK']) == 0 and int(os.environ['LOCAL_RANK']) == 0:
        save_model_hdfs(params)
    print("\nTraining complete!")