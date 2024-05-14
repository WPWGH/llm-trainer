# -*- coding: utf-8 -*-

import json
from copy import deepcopy
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother

from src.utils import rank0_print

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

class UnsupervisedDataset(Dataset):
    """Dataset for unsupervised pretrain."""

    def __init__(self, data_path, tokenizer, max_len: int, dynamic_length = False) -> None:
        super(UnsupervisedDataset, self).__init__()

        rank0_print("Formatting unsupervised inputs...")
        all_data = []
        with open(data_path, "r") as fp:
            for line in tqdm(fp):
                if line.strip() == "": continue
                all_data.append(line)
        
        self.all_data = all_data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dynamic_length = dynamic_length

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, i):
        line_dict = json.loads(self.all_data[i])
        
        text = line_dict["text"]
        input_tokens = [self.tokenizer.bos_token] + \
            self.tokenizer.tokenize(text) + \
                [self.tokenizer.eos_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)  # [L_i, ]
        labels = deepcopy(input_ids)
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }
    
    def collate_fn(self, batch):
        max_length_in_batch = max(len(b["input_ids"]) for b in batch)

        if self.dynamic_length:
            padding_length = min(max_length_in_batch, self.max_len)
        else:
            padding_length = self.max_len
        
        for i in range(len(batch)):
            if len(batch[i]["input_ids"]) < padding_length:
                batch[i]["input_ids"] += [self.tokenizer.pad_token_id] * (padding_length - len(batch[i]["input_ids"]))
                batch[i]["labels"] += [IGNORE_TOKEN_ID] * (padding_length - len(batch[i]["labels"]))
                batch[i]["attention_mask"] += [0] * (padding_length - len(batch[i]["attention_mask"]))
            else:
                batch[i]["input_ids"] = batch[i]["input_ids"][:padding_length]
                batch[i]["labels"] = batch[i]["labels"][:padding_length]
                batch[i]["attention_mask"] = batch[i]["attention_mask"][:padding_length]
        
        return {
            "input_ids": torch.tensor([b["input_ids"] for b in batch]),
            "labels": torch.tensor([b["labels"] for b in batch]),
            "attention_mask": torch.tensor([b["attention_mask"] for b in batch])
        }

class SemiSupervisedDataset(Dataset):
    """Dataset for semi-supervised continue pretrain, as well as instruction tuning."""

    def __init__(
        self, data_path, tokenizer, max_len: int, dynamic_length = False,
        padding_side: str = "right", return_index: bool = False, mode: str = "train"
    ):
        super(SemiSupervisedDataset, self).__init__()
        assert padding_side in ["right", "left"]
        assert mode in ["train", "test"]

        rank0_print("Formatting semi-supervised inputs...")
        all_data = []
        with open(data_path, "r") as fp:
            for line in tqdm(fp):
                if line.strip() == "": continue
                all_data.append(line)
        
        self.all_data = all_data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dynamic_length = dynamic_length
        self.padding_side = padding_side
        self.return_index = return_index
        self.mode = mode

    def __len__(self):
        return len(self.all_data)

    def apply_chat_template(self, messages: list, add_generation_prompt = False):
        input_ids, labels, attention_mask = [], [], []
        for msg_dict in messages:
            if msg_dict["role"] in ["system", "user"]:
                current_tokens = [self.tokenizer.bos_token] + \
                    self.tokenizer.tokenize("[{}]\n{}".format(msg_dict["role"], msg_dict["content"])) + \
                        [self.tokenizer.eos_token]
                current_ids = self.tokenizer.convert_tokens_to_ids(current_tokens)

                input_ids.extend(current_ids)
                attention_mask.extend([1] * len(current_ids))
                labels.extend([IGNORE_TOKEN_ID] * len(current_ids))  # loss mask for `system` and `user`
            else:
                # Masked tokens
                current_tokens = [self.tokenizer.bos_token] + \
                    self.tokenizer.tokenize("[{}]\n".format(msg_dict["role"]))
                current_ids = self.tokenizer.convert_tokens_to_ids(current_tokens)

                input_ids.extend(current_ids)
                attention_mask.extend([1] * len(current_ids))
                labels.extend([IGNORE_TOKEN_ID] * len(current_ids))

                # Optimized tokens
                current_tokens = self.tokenizer.tokenize(msg_dict["content"]) + [self.tokenizer.eos_token]
                current_ids = self.tokenizer.convert_tokens_to_ids(current_tokens)

                input_ids.extend(current_ids)
                attention_mask.extend([1] * len(current_ids))
                labels.extend(current_ids)
        
        # 主要在推理阶段使用，把`<bos>[assistant]\n`加在后面，从而引导模型进行输出
        if add_generation_prompt:
            current_tokens = [self.tokenizer.bos_token] + \
                self.tokenizer.tokenize("[{}]\n".format("assistant"))
            current_ids = self.tokenizer.convert_tokens_to_ids(current_tokens)

            input_ids.extend(current_ids)
            attention_mask.extend([1] * len(current_ids))
            labels.extend(current_ids)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }

    def __getitem__(self, i):
        line_dict = json.loads(self.all_data[i])

        """
        - 数据格式：
        {
            "messages": [
                {"role": "system", "content": "xxx"},
                {"role": "user", "content": "xxx"},
                {"role": "assistant", "content": "xxx"},
                ...
            ]
        }
        - 统一模板：
        <bos>[system]
        system_content<eos><bos>[user]
        user_content<eos><bos>[assistant]
        assistant_content<eos>
        - 在训练过程中，`system`、`user`和`assitant前缀`的tokens都会被mask
        """
        
        ret_dict = self.apply_chat_template(line_dict["messages"], add_generation_prompt=(self.mode == "test"))
        ret_dict["index"] = i
        return ret_dict
    
    def collate_fn(self, batch):
        max_length_in_batch = max(len(b["input_ids"]) for b in batch)

        if self.dynamic_length:
            padding_length = min(max_length_in_batch, self.max_len)
        else:
            padding_length = self.max_len
        
        for i in range(len(batch)):
            if len(batch[i]["input_ids"]) < padding_length:
                if self.padding_side == "right":
                    batch[i]["input_ids"] += [self.tokenizer.pad_token_id] * (padding_length - len(batch[i]["input_ids"]))
                    batch[i]["labels"] += [IGNORE_TOKEN_ID] * (padding_length - len(batch[i]["labels"]))
                    batch[i]["attention_mask"] += [0] * (padding_length - len(batch[i]["attention_mask"]))
                else:
                    batch[i]["input_ids"] = [self.tokenizer.pad_token_id] * (padding_length - len(batch[i]["input_ids"])) + batch[i]["input_ids"]
                    batch[i]["labels"] = [IGNORE_TOKEN_ID] * (padding_length - len(batch[i]["labels"])) + batch[i]["labels"]
                    batch[i]["attention_mask"] = [0] * (padding_length - len(batch[i]["attention_mask"])) + batch[i]["attention_mask"]
            else:
                if self.padding_side == "right":
                    batch[i]["input_ids"] = batch[i]["input_ids"][:padding_length]
                    batch[i]["labels"] = batch[i]["labels"][:padding_length]
                    batch[i]["attention_mask"] = batch[i]["attention_mask"][:padding_length]
                else:
                    batch[i]["input_ids"] = batch[i]["input_ids"][-padding_length:]
                    batch[i]["labels"] = batch[i]["labels"][-padding_length:]
                    batch[i]["attention_mask"] = batch[i]["attention_mask"][-padding_length:]
        
        output_dict = {
            "input_ids": torch.tensor([b["input_ids"] for b in batch]),
            "labels": torch.tensor([b["labels"] for b in batch]),
            "attention_mask": torch.tensor([b["attention_mask"] for b in batch])
        }
        if self.return_index:
            output_dict["index"] = torch.tensor([b["index"] for b in batch])
        
        return output_dict

class InstructionTuningDataset(Dataset):
    """Dataset for instruction tuning. This `Dataset` class is suitable for finetuning from a
    public `Chat` model, e.g., Qwen1.5-72B-Chat, which uses `apply_chat_template` to format the 
    prompt template. If you finetuning from `Base` model, e.g., Qwen1.5-72B, `SemiSupervisedDataset`
    is recommended!!!"""
    def __init__(
        self, data_path, tokenizer, max_len: int, dynamic_length = False, 
        model_type = "qwen2", padding_side: str = "right", return_index: bool = False, 
        mode: str = "train"
    ):
        super(InstructionTuningDataset, self).__init__()
        assert model_type in ["qwen2"], "model_type `{}` has not been supported, please contact the developer for help.".format(model_type)
        assert padding_side in ["right", "left"]
        assert mode in ["train", "test"]

        rank0_print("Formatting instruction tuning inputs...")
        all_data = []
        with open(data_path, "r") as fp:
            for line in tqdm(fp):
                if line.strip() == "": continue
                all_data.append(line)
        
        self.all_data = all_data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dynamic_length = dynamic_length
        self.model_type = model_type
        self.padding_side = padding_side
        self.return_index = return_index
        self.mode = mode

        # For Qwen1.5
        self.special_token_id = {
            "qwen2": {
                "bos_token_id": self.tokenizer.bos_token_id,
                "system": self.tokenizer.convert_tokens_to_ids(["system"])[0],
                "user": self.tokenizer.convert_tokens_to_ids(["user"])[0],
                "assistant": self.tokenizer.convert_tokens_to_ids(["assistant"])[0]
            }
        }

    def __len__(self):
        return len(self.all_data)

    def _is_system_field(self, text_ids, index):
        if self.model_type == "qwen2":
            if text_ids[index] == self.special_token_id[self.model_type]["bos_token_id"] and \
                (index + 1 < len(text_ids)) and \
                    text_ids[index + 1] == self.special_token_id[self.model_type]["system"]:
                return True
            else:
                return False
        else:
            raise ValueError("model_type `{}` is not supported!".format(self.model_type))

    def _is_user_field(self, text_ids, index):
        if self.model_type == "qwen2":
            if text_ids[index] == self.special_token_id[self.model_type]["bos_token_id"] and \
                (index + 1 < len(text_ids)) and \
                    text_ids[index + 1] == self.special_token_id[self.model_type]["user"]:
                return True
            else:
                return False
        else:
            raise ValueError("model_type `{}` is not supported!".format(self.model_type))

    def _is_assistant_field(self, text_ids, index):
        if self.model_type == "qwen2":
            if text_ids[index] == self.special_token_id[self.model_type]["bos_token_id"] and \
                (index + 1 < len(text_ids)) and \
                    text_ids[index + 1] == self.special_token_id[self.model_type]["assistant"]:
                return True
            else:
                return False
        else:
            raise ValueError("model_type `{}` is not supported!".format(self.model_type))

    def apply_chat_template(self, messages: list, add_generation_prompt = False):
        text_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            padding=True,
            max_length=self.max_len,
            truncation=True,
        )
        input_ids = deepcopy(text_ids)

        # Add loss mask to system and user tokens
        target_ids = []
        should_mask = True
        i = 0
        while i < len(text_ids):
            t = text_ids[i]
            if t == self.tokenizer.pad_token_id:
                target_ids.append(IGNORE_TOKEN_ID)
            elif self._is_system_field(text_ids, i):
                should_mask = True
                target_ids.append(IGNORE_TOKEN_ID)
            elif self._is_user_field(text_ids, i):
                should_mask = True
                target_ids.append(IGNORE_TOKEN_ID)
            elif self._is_assistant_field(text_ids, i):
                should_mask = False
                target_ids.append(IGNORE_TOKEN_ID)
            else:
                if should_mask:
                    target_ids.append(IGNORE_TOKEN_ID)
                else:
                    target_ids.append(t)
            i += 1
        
        attention_mask = [1 if t != self.tokenizer.pad_token_id else 0 for t in input_ids]
        
        return {
            "input_ids": input_ids,
            "labels": target_ids,
            "attention_mask": attention_mask
        }
    
    def __getitem__(self, i):
        line_dict = json.loads(self.all_data[i])

        """
        - 数据格式：
        {
            "messages": [
                {"role": "system", "content": "xxx"},
                {"role": "user", "content": "xxx"},
                {"role": "assistant", "content": "xxx"},
                ...
            ]
        }
        - 不同的Chat模型会使用不同的模板，比如Qwen1.5使用的模板是：
        <|im_start|>system
        This is a system message!<|im_end|>
        <|im_start|>user
        This is a 1-turn user message!<|im_end|>
        <|im_start|>assistant
        This is a 1-turn assistant message!<|im_end|>
        """
        
        # Set `add_generation_prompt` to True during inference
        ret_dict = self.apply_chat_template(line_dict["messages"], add_generation_prompt=(self.mode == "test"))
        ret_dict["index"] = i
        return ret_dict
    
    def collate_fn(self, batch):
        max_length_in_batch = max(len(b["input_ids"]) for b in batch)

        if self.dynamic_length:
            padding_length = min(max_length_in_batch, self.max_len)
        else:
            padding_length = self.max_len
        
        for i in range(len(batch)):
            if len(batch[i]["input_ids"]) < padding_length:
                if self.padding_side == "right":
                    batch[i]["input_ids"] += [self.tokenizer.pad_token_id] * (padding_length - len(batch[i]["input_ids"]))
                    batch[i]["labels"] += [IGNORE_TOKEN_ID] * (padding_length - len(batch[i]["labels"]))
                    batch[i]["attention_mask"] += [0] * (padding_length - len(batch[i]["attention_mask"]))
                else:
                    batch[i]["input_ids"] = [self.tokenizer.pad_token_id] * (padding_length - len(batch[i]["input_ids"])) + batch[i]["input_ids"]
                    batch[i]["labels"] = [IGNORE_TOKEN_ID] * (padding_length - len(batch[i]["labels"])) + batch[i]["labels"]
                    batch[i]["attention_mask"] = [0] * (padding_length - len(batch[i]["attention_mask"])) + batch[i]["attention_mask"]
            else:
                if self.padding_side == "right":
                    batch[i]["input_ids"] = batch[i]["input_ids"][:padding_length]
                    batch[i]["labels"] = batch[i]["labels"][:padding_length]
                    batch[i]["attention_mask"] = batch[i]["attention_mask"][:padding_length]
                else:
                    batch[i]["input_ids"] = batch[i]["input_ids"][-padding_length:]
                    batch[i]["labels"] = batch[i]["labels"][-padding_length:]
                    batch[i]["attention_mask"] = batch[i]["attention_mask"][-padding_length:]
        
        output_dict = {
            "input_ids": torch.tensor([b["input_ids"] for b in batch]),
            "labels": torch.tensor([b["labels"] for b in batch]),
            "attention_mask": torch.tensor([b["attention_mask"] for b in batch])
        }
        if self.return_index:
            output_dict["index"] = torch.tensor([b["index"] for b in batch])
        
        return output_dict

class SupervisedDataset(Dataset):
    """Dataset for supervised finetune."""

    def __init__(
        self,
        data_path, 
        tokenizer, 
        max_len: int, 
        dynamic_length = False, 
        mask_instruction = True, 
        padding_side: str = "right", 
        return_index: bool = False, 
        mode: str = "train"
    ):
        super(SupervisedDataset, self).__init__()
        assert padding_side in ["right", "left"]
        assert mode in ["train", "test"]

        rank0_print("Formatting supervised inputs...")
        all_data = []
        with open(data_path, "r") as fp:
            for line in tqdm(fp):
                if line.strip() == "": continue
                all_data.append(line)
        
        self.all_data = all_data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dynamic_length = dynamic_length
        self.mask_instruction = mask_instruction
        self.padding_side = padding_side
        self.return_index = return_index
        self.mode = mode
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, i):
        line_dict = json.loads(self.all_data[i])

        """
        - 数据格式：
        {
            "instruction": "xxx",
            "input": "xxx",
            "output": "xxx"
        }
        """
        
        # Tokenize instruction and input
        input_tokens = [self.tokenizer.bos_token] + \
            self.tokenizer.tokenize(line_dict['instruction']) + \
                self.tokenizer.tokenize(line_dict['input'])
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        if self.mask_instruction:
            labels = [IGNORE_TOKEN_ID] * len(input_ids)
        else:
            labels = deepcopy(input_ids)
        attention_mask = [1] * len(input_ids)
        
        if self.mode == "train":
            # Tokenize output
            input_tokens = self.tokenizer.tokenize(line_dict['output']) + [self.tokenizer.eos_token]
            input_ids.extend(self.tokenizer.convert_tokens_to_ids(input_tokens))
            labels.extend(self.tokenizer.convert_tokens_to_ids(input_tokens))

            attention_mask.extend([1] * len(input_tokens))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "index": i
        }
    
    def collate_fn(self, batch):
        max_length_in_batch = max(len(b["input_ids"]) for b in batch)

        if self.dynamic_length:
            padding_length = min(max_length_in_batch, self.max_len)
        else:
            padding_length = self.max_len
        
        for i in range(len(batch)):
            if len(batch[i]["input_ids"]) < padding_length:
                if self.padding_side == "right":
                    batch[i]["input_ids"] += [self.tokenizer.pad_token_id] * (padding_length - len(batch[i]["input_ids"]))
                    batch[i]["labels"] += [IGNORE_TOKEN_ID] * (padding_length - len(batch[i]["labels"]))
                    batch[i]["attention_mask"] += [0] * (padding_length - len(batch[i]["attention_mask"]))
                else:
                    batch[i]["input_ids"] = [self.tokenizer.pad_token_id] * (padding_length - len(batch[i]["input_ids"])) + batch[i]["input_ids"]
                    batch[i]["labels"] = [IGNORE_TOKEN_ID] * (padding_length - len(batch[i]["labels"])) + batch[i]["labels"]
                    batch[i]["attention_mask"] = [0] * (padding_length - len(batch[i]["attention_mask"])) + batch[i]["attention_mask"]
            else:
                if self.padding_side == "right":
                    batch[i]["input_ids"] = batch[i]["input_ids"][:padding_length]
                    batch[i]["labels"] = batch[i]["labels"][:padding_length]
                    batch[i]["attention_mask"] = batch[i]["attention_mask"][:padding_length]
                else:
                    batch[i]["input_ids"] = batch[i]["input_ids"][-padding_length:]
                    batch[i]["labels"] = batch[i]["labels"][-padding_length:]
                    batch[i]["attention_mask"] = batch[i]["attention_mask"][-padding_length:]
        
        output_dict = {
            "input_ids": torch.tensor([b["input_ids"] for b in batch]),
            "labels": torch.tensor([b["labels"] for b in batch]),
            "attention_mask": torch.tensor([b["attention_mask"] for b in batch])
        }
        if self.return_index:
            output_dict["index"] = torch.tensor([b["index"] for b in batch])
        
        return output_dict
