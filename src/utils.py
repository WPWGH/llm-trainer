# -*- coding: utf-8 -*-

import os
import yaml
import transformers
from transformers import deepspeed
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from glob import glob

def rank0_print(*args):
    if "LOCAL_RANK" in os.environ:    
        if int(os.environ['LOCAL_RANK']) == 0:
            print(*args)
    else:
        print(*args)


def get_filename(filepath):

    if 'hdfs' in filepath:
        ## do the copying only on local rank 0
        if int(os.environ['LOCAL_RANK']) == 0:
            os.system(f'hdfs dfs -get {filepath} .')
        filename = [f for f in filepath.split('/') if f != ''][-1]

    else:
        filename = filepath

    # dist.barrier() ## all processes will wait till the file is copied to local

    return filename

def get_parameters(filepath):

    ## params could be in hdfs or locally if the nfs is mounted during the job
    filename = get_filename(filepath)
    params = yaml.load(open(filename), yaml.FullLoader)
    return params

## function to check if hdfs directory/file exists
def hexists(file_path):
    if file_path.startswith("hdfs"):
        return os.system(f"hdfs dfs -test -e {file_path}") == 0
    return os.path.exists(file_path)

## function to save the model in hdfs
def save_model_hdfs(params):

    files = glob(os.path.join(params['trainer-params']['output_dir'], "*"))
    checkpoints = [f for f in files if "checkpoint" in f]

    if params['trainer-params']['output_hdfs_dir'] == "":
        print("[Warning] output_hdfs_dir is empty! skip saving!")
        return 
    
    ## create output dir if it doesnt exist ..    
    if not hexists(params['trainer-params']['output_hdfs_dir']):
        os.system(f"hdfs dfs -mkdir -p {params['trainer-params']['output_hdfs_dir']}")

    ## loop to store all possible checkpoints

    for cur_checkpoint in checkpoints:
        dir_name = cur_checkpoint.split('/')[-1]
        hdfs_dir_path = os.path.join(params['trainer-params']['output_hdfs_dir'], dir_name)

        if not hexists(hdfs_dir_path):
            os.system(f"hdfs dfs -mkdir {hdfs_dir_path}")

        if params['trainer-params']['output_fp32_model']:
            print("\nStarted deepspeed fp16 to fp32 weights conversion for ", dir_name, " ...")
            os.system(f"python3 {cur_checkpoint}/zero_to_fp32.py {cur_checkpoint} {cur_checkpoint}/pytorch_model.bin")
            print("\nConversion successful ...")

        ## copy the latest checkpoint and the logs to output_model_path
        os.system(f"hdfs dfs -put -f {cur_checkpoint}/pytorch_model.bin {hdfs_dir_path}")
        os.system(f"hdfs dfs -put -f {cur_checkpoint}/*.json {hdfs_dir_path}")
        os.system(f"hdfs dfs -put -f {cur_checkpoint}/training_args.bin {hdfs_dir_path}")

    ## copy tensorboard and wandb logs
    os.system(f"hdfs dfs -put -f {params['trainer-params']['logging_dir']} {params['trainer-params']['output_hdfs_dir']}")
    os.system(f"hdfs dfs -put -f wandb {params['trainer-params']['output_hdfs_dir']}")

def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return

def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer, output_dir: str, use_lora: bool = False, bias="none"
):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)