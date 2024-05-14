# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Dict, Optional, List

import transformers

@dataclass
class ModelArguments:
    pretrain_model_path: Optional[str] = field(default="Qwen/Qwen-7B")
    tokenizer_path: Optional[str] = field(default="Qwen/Qwen-7B")
    lora_path: Optional[str] = field(default="")  # Only used in inference stage
    use_flash_attention: bool = True

@dataclass
class TokenizerArguments:
    add_special_tokens: bool = False
    initialize_additional_special_tokens: bool = False
    special_tokens: Dict = field(default_factory=lambda: {
        "bos_token": "",
        "eos_token": "",
        "unk_token": "",
        "sep_token": "",
        "pad_token": "",
        "additional_special_tokens": []
    })

@dataclass
class DataArguments:
    dataset_type: str = field(
        default="supervised_finetune"
    )
    train_file: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    test_file: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    prediction_file_dir: Optional[str] = field(default="")
    prediction_file_name: Optional[str] = field(default="")
    max_len: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    dynamic_length: bool = False
    mask_instruction: bool = True
    padding_side: str = "right"

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    wandb_project_name: str = field(default="wpw")
    run_name: str = field(default="wpw_1")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    output_hdfs_dir: str = field(default="")
    output_fp32_model: bool = False

@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    use_lora: bool = False
    q_lora: bool = False

@dataclass
class MoEArguments:
    use_moe: bool = False
    expert_nums: int = 3
    router_type: str = "token"  # or "instance"

@dataclass
class GenerateArguments:
    care_tokens: List[int] = field(default_factory=lambda: [])
    batch_size: int = 16
    max_new_tokens: int = 128
    do_sample: bool = False
    top_p: float = 1.0
    temperature: float = 1.0
    top_k: int = 1
    num_beams: int = 1