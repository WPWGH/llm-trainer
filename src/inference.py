# -*- coding: utf-8 -*-

import os
import argparse

import datasets
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import deepspeed
import transformers
from transformers.deepspeed import HfDeepSpeedConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_pt_utils import SequentialDistributedSampler, distributed_concat
from peft import PeftModel

from src.utils import get_parameters, rank0_print
from src.args import ModelArguments, TokenizerArguments, DataArguments, GenerateArguments
from src.dataset import SemiSupervisedDataset, InstructionTuningDataset, SupervisedDataset

def return_deepspeed_config(model_hidden_size, train_batch_size):
    ds_config = {
        "fp16": {
            "enabled": True
        },
        "bf16": {
            "enabled": False
        },
        "zero_optimization": 
        {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": model_hidden_size * model_hidden_size,
            "allgather_bucket_size": model_hidden_size * model_hidden_size,
            "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
            "stage3_param_persistence_threshold": 10 * model_hidden_size
        },
        "steps_per_print": 2000,
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": 1,
        "wall_clock_breakdown": False
    }

    ## If the code throws memory errors or Cublas error, Try to reduce batch size or
    ## off-load params to CPU by adding this to the zero_optimization config. This is
    ## significantly slow but the last option.

    '''
    "offload_param": {"device": "cpu", 
        "pin_memory": True
    },
    '''

    return ds_config

def add_special_tokens(model, tokenizer, tokenizer_args):
    """
    加入自定义的special tokens，可以用于如下两种情况：
    - 下游任务需要special_token来代替标签，比如标签类别很多的任务中，a|b|c|d英文字母不够用；
    - 对特征分桶得到离散特征后，使用special_token来代替桶特征；
    **注意**：推理阶段并没有对model的embed进行resize操作，
    """
    if tokenizer_args and tokenizer_args.add_special_tokens:
        special_tokens = dict()
        for k, v in tokenizer_args.special_tokens.items():
            if v: special_tokens[k] = v
        tokenizer.add_special_tokens(special_tokens)
        rank0_print("Add special_tokens: {}".format(special_tokens))
    
    return model, tokenizer

def load_model_and_tokenizer(model_args, tokenizer_args, generate_args):
    # Load config
    config = transformers.AutoConfig.from_pretrained(
        model_args.pretrain_model_path
    )
    config.use_cache = True
    if model_args.use_flash_attention: 
        config._attn_implementation = "flash_attention_2"
    
    # Initialize deepspeed config
    model_hidden_size = config.hidden_size
    batch_size = generate_args.batch_size * int(os.environ['WORLD_SIZE'])
    ds_config = return_deepspeed_config(model_hidden_size, batch_size)
    hfdsc = HfDeepSpeedConfig(ds_config)

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_path if model_args.tokenizer_path else model_args.pretrain_model_path,
        padding_side="left",
        use_fast=False,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.pretrain_model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    # Add special tokens
    model, tokenizer = add_special_tokens(model, tokenizer, tokenizer_args)

    # Load lora params
    if model_args.lora_path:
        model = PeftModel.from_pretrained(model, model_args.lora_path)

    # Initialize ds_engine
    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    model = ds_engine.module
    model = model.eval()
    print('Loaded model.')
    

    # 部分模型的tokenizer缺少bos_token、eos_token、pad_token，需要进行赋值处理，
    # 否则，dataset处理会转化错误。
    if config.model_type in ["qwen2", "qwen2_moe"]:
        tokenizer.bos_token = "<|im_start|>"
        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids([tokenizer.bos_token])[0]
    tokenizer.pad_token = tokenizer.pad_token if tokenizer.pad_token else tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
    assert tokenizer.bos_token and tokenizer.eos_token and tokenizer.pad_token

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    return model, tokenizer

def inference(data_args, generate_args, test_dataset, model, tokenizer):

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = torch.device("cuda", local_rank)
    
    # get the dataloader
    sampler = SequentialDistributedSampler(test_dataset, num_replicas = world_size, rank = rank)
    data_loader = DataLoader(test_dataset, sampler = sampler, batch_size = generate_args.batch_size, collate_fn=test_dataset.collate_fn)
    
    ## No tensor should be beyond this length. This will be used for distributed concat
    max_sequence_length = data_args.max_len + generate_args.max_new_tokens
    max_new_tokens = generate_args.max_new_tokens
    pad_token_id = tokenizer.pad_token_id
    num_total_examples = len(test_dataset)

    output_dic = {}
    ctr = 0
    all_sequences = []
    all_generates = []
    all_scores = []
    all_indexes = []
    with torch.no_grad():
        for batch in data_loader:
            ctr += 1
            
            input_ids, attention_mask, indexes = batch["input_ids"], batch["attention_mask"], batch["index"]
            input_ids, attention_mask, indexes = input_ids.to(device), attention_mask.to(device), indexes.to(device)

            if local_rank == 0:
                print("Batch No: ", ctr)
            
            ## need to do this, otherwise distributed model.generate throws cuda error!
            if ctr == 1:
                model(input_ids = input_ids, attention_mask = attention_mask)

            generated_dict = model.generate(input_ids, 
                attention_mask = attention_mask, 
                do_sample = generate_args.do_sample, 
                top_p = generate_args.top_p, 
                top_k = generate_args.top_k, 
                num_beams = generate_args.num_beams, 
                max_new_tokens = generate_args.max_new_tokens, 
                eos_token_id = tokenizer.eos_token_id, 
                return_dict_in_generate = True, 
                output_scores = True, 
                synced_gpus = True
            )
            sequences = generated_dict["sequences"]  # [B, L_b]
            generates = generated_dict["sequences"][:, input_ids.shape[-1]:]  # [B, O_b]
            scores = generated_dict["scores"]  # ([B, V], [B, V], ...)

            # Pad to max length for distributed concat            
            sequences = torch.nn.functional.pad(sequences, (0, max_sequence_length - sequences.shape[-1]), value=pad_token_id)
            generates = torch.nn.functional.pad(generates, (0, max_new_tokens - generates.shape[-1]), value=pad_token_id)
            scores = torch.stack(scores, dim=1)
            scores = torch.nn.functional.pad(scores, (0, 0, 0, max_new_tokens - scores.shape[1]), value=0)

            # Get care tokens to return
            if generate_args.care_tokens is not None and len(generate_args.care_tokens) > 0:
                scores = scores[:, :, generate_args.care_tokens].contiguous().detach()
            else:
                scores = scores[:, :, 0].contiguous().detach()

            all_sequences.append(sequences)
            all_generates.append(generates)
            all_scores.append(scores)
            all_indexes.append(indexes)
        
        all_sequences = torch.cat(all_sequences, dim=0)
        all_generates = torch.cat(all_generates, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        all_indexes = torch.cat(all_indexes, dim=0)

        # Distributed concat to gather all tensors
        all_sequences = distributed_concat(all_sequences, num_total_examples).cpu().tolist()
        all_generates = distributed_concat(all_generates, num_total_examples).cpu().tolist()
        all_scores = distributed_concat(all_scores, num_total_examples).cpu().tolist()
        all_indexes = distributed_concat(all_indexes, num_total_examples).cpu().tolist()

        # Decode sequence and output
        all_sequences = tokenizer.batch_decode(all_sequences, skip_special_tokens = True)
        all_generates = tokenizer.batch_decode(all_generates, skip_special_tokens = True)

        # Format output dict
        for index, seq, gen, sco in zip(all_indexes, all_sequences, all_generates, all_scores):
            output_dic[index] = {
                "sequence": seq,
                "output": gen,
                "score": sco
            }
        
    assert num_total_examples == len(output_dic), "Major indexing concat error, needs fix"
    output_dic_items = sorted(output_dic.items(), key = lambda x:x[0])
    outputs = [i[1] for i in output_dic_items]

    return outputs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--params", help = "parameters yaml file to launch inference")
    args = parser.parse_args()
    params = get_parameters(args.params)

    ## set the init_process_group first
    dist.init_process_group('nccl', rank = int(os.environ['RANK']), 
                            world_size = int(os.environ['WORLD_SIZE']))

    # Parse args
    model_args = transformers.HfArgumentParser(ModelArguments).parse_dict(params["model-params"])[0]
    tokenizer_args = transformers.HfArgumentParser(TokenizerArguments).parse_dict(params["tokenizer-params"])[0] if "tokenizer-params" in params else None
    data_args = transformers.HfArgumentParser(DataArguments).parse_dict(params["data-params"])[0]
    generate_args = transformers.HfArgumentParser(GenerateArguments).parse_dict(params["generate-params"])[0]
    print("\nParams loaded ...")

    # get the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args, tokenizer_args, generate_args)
    print("\nModel and tokenizer loaded!")
    
    # get the dataset
    if data_args.dataset_type == "continual_pretrain":
        test_dataset = SemiSupervisedDataset(
            data_path=data_args.test_file,
            tokenizer=tokenizer,
            max_len=data_args.max_len,
            dynamic_length=True,
            padding_side="left",
            return_index=True,
            mode="test"
        )
    elif data_args.dataset_type == "instruction_tuning":
        test_dataset = InstructionTuningDataset(
            data_path=data_args.test_file,
            tokenizer=tokenizer,
            max_len=data_args.max_len,
            dynamic_length=True,
            model_type=model.config.model_type,
            padding_side="left",
            return_index=True,
            mode="test"
        )
    elif data_args.dataset_type == "supervised_finetune":
        test_dataset = SupervisedDataset(
            data_path=data_args.test_file,
            tokenizer=tokenizer,
            max_len=data_args.max_len,
            dynamic_length=True,
            padding_side="left",
            return_index=True,
            mode="test"
        )
    else:
        raise ValueError("Not supported `dataset_type`: `{}`, it should be one of `continual pretrain`, `instruction_tuning` or `supervised_finetune`".format(data_args.dataset_type))
    print("\nTest dataset loaded ...")
    
    ## generate the outputs
    outputs = inference(data_args, generate_args, test_dataset, model, tokenizer)
    print("\nOutput generation done ...")
    
    ## add outputs to the test_dataset and write to csv
    if int(os.environ['RANK']) == 0 and int(os.environ['LOCAL_RANK']) == 0:
        test_dataset = datasets.load_dataset('json', data_files = {'test' : data_args.test_file})['test']
        test_dataset = test_dataset.add_column('model_predictions', outputs)
        test_dataset.to_json(data_args.prediction_file_name, force_ascii=False)

        ## move it to hdfs location
        if data_args.prediction_file_dir != "":
            os.system(f"hdfs dfs -put -f {data_args.prediction_file_name} {data_args.prediction_file_dir}")
    dist.barrier()
    print("\nPredictions file written successfully ...")
