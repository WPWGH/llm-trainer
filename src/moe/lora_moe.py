# -*- coding: utf-8 -*-

"""
Copy and modified from:
https://code.byted.org/ecom_govern/EasyGuard/blob/lzb_dev/examples/valley/valley/train/lora_moe.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, lora_config, **kwargs):
        super(LoRALayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.lora_config = lora_config

        self.scaling = lora_config.lora_alpha / lora_config.lora_r
        self.dropout = lora_config.lora_dropout

        # Actual trainable parameters
        self.lora_A = nn.Linear(self.in_features, self.lora_config.lora_r, bias=False)
        self.lora_B = nn.Linear(self.lora_config.lora_r, self.out_features, bias=False)

        # Initialize lora weights
        self.init_linear()

    def init_linear(self):
        # initialize A the same way as the default for nn.Linear and B to zero
        # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        previous_dtype = x.dtype
        
        x = x.to(self.lora_A.weight.dtype)
        if self.training:
            x = F.dropout(x, p=self.dropout)
        lora_result = self.lora_B(self.lora_A(x)) * self.scaling

        lora_result = lora_result.to(previous_dtype)
        return lora_result

class LoRAMoELayer(nn.Module):
    def __init__(self, base_layer, lora_config, moe_config):
        super(LoRAMoELayer, self).__init__()

        self.lora_config = lora_config
        self.moe_config = moe_config
        self.base_layer = base_layer  # the target sub-module
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.expert_nums = moe_config.expert_nums
        self.router_type = moe_config.router_type
        
        # Frozen the base parameters
        for key, module in self.base_layer.named_modules():
            module.requires_grad_(False)

        self.lora_moe_gate = nn.Linear(
            self.in_features, 
            self.moe_config.expert_nums, 
            bias=False
        )
        self.lora_moe = nn.ModuleList([
            LoRALayer(self.in_features, self.out_features, lora_config) 
            for _ in range(self.expert_nums)])

    def forward(self, x, *args, **kwargs):
        result = self.base_layer(x, *args, **kwargs)

        lora_result = []
        for idx in range(len(self.lora_moe)):
            lora = self.lora_moe[idx]
            lora_result += [lora(x)]
        lora_result = torch.stack(lora_result, dim=-2)  # [B, N, E, d]

        gating = self.lora_moe_gate(x)  # [B, N, E]
        soft_gating = torch.softmax(gating, dim=-1)

        if self.router_type == 'instance':
            batch_size, token_nums, expert_nums = soft_gating.shape
            soft_gating = torch.mean(soft_gating, dim=-2, keepdim=True).repeat(1, token_nums, 1)

        # [B, N, 1, E] @ [B, N, E, d] -> [B, N, 1, d] -> [B, N, d]
        lora_result = torch.matmul(soft_gating.unsqueeze(-2), lora_result).squeeze(-2)

        result += lora_result
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep

class LoRAMoE:
    def __init__(self, model, lora_config, moe_config):
        super(LoRAMoE, self).__init__()

        self.lora_config = lora_config
        self.moe_config = moe_config
        self.lora_target_modules = set(lora_config.lora_target_modules)

        self.model = model
        for key, module in self.model.named_modules():
            module.requires_grad_(False)

        self.recursive_replace()

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent, child_name, new_module):
        setattr(parent, child_name, new_module)

    def recursive_replace(self):
        for key, _ in self.model.named_modules():
            target_name = key.split(".")[-1]
            if target_name in self.lora_target_modules:
                parent, target, target_name = self._get_submodules(key)
                new_module = LoRAMoELayer(target, self.lora_config, self.moe_config)
                self._replace_module(parent, target_name, new_module)

def get_lora_moe_model(model, lora_config, moe_config):
    lora_model = LoRAMoE(model, lora_config, moe_config)
    return lora_model.model

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM

    class LoraArguments:
        lora_r: int = 64
        lora_alpha: int = 16
        lora_dropout: float = 0.05
        lora_target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ]
        lora_weight_path: str = ""
        lora_bias: str = "none"
        use_lora: bool = True
        q_lora: bool = False

    class MoEArguments:
        use_moe: bool = True
        expert_nums: int = 3
        router_type: str = "token"  # or "instance"

    model = AutoModelForCausalLM.from_pretrained("/mnt/bn/liuyuhang-yg/users/liuyuhang/llm_models/public/Qwen1.5-1.8B", device_map=None)
    model = get_lora_moe_model(model, LoraArguments(), MoEArguments())
    print(model)
