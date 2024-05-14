# README

## 简介

## 框架原则

- 兼容性：充分拥抱开源框架，从而保证与最新开源模型的兼容性，且保持代码简洁性、易读性；
- 简洁性：接口统一收口到yml文件，且保证yml文件只保留最重要的参数配置；
- 统一性：版本统一、镜像统一，极致地减少版本之间的依赖以及繁杂冗余的镜像；
- 易用性：提供在最少配置修改的情况下跑通代码；

## Features

- 长度扩展至4k～8k，全面支持flash attention2；
- 统一预训练、SFT场景代码，统一各个场景的数据格式，尽可能统一至最新版本库和镜像；
- 支持更丰富和更大规模模型，包括34B和72B模型；
- 支持更丰富的训练方式，包括全参数、Lora、QLora；
- 提供业务应用参考的数据模式、环境、镜像、Trail等等；

## 使用场景

主要支持3个场景，`预训练(Pretrain)`、`继续预训练(Continual Pretrain)`、`监督微调(Supervised Finetune)`。

### 预训练(Pretrain)

- `预训练(Pretrain)`场景主要利用无监督数据，采用Next Token Prediction对文本所有的token进行优化，为模型注入广泛而又通用的知识；
- 数据格式：jsonl格式，每行一条样本，必须存在`text`字段，表示无监督文本。
```jsonl
{
    "text": "xxx"
}
```
- 运行命令：
```bash
cd safetygpt
bash distributed_train.sh configs/yi_6b_pretrain_stage.yml
```
- 优化目标：采用Next Token Prediction优化所有的tokens。
$ \mathcal{L}=-\frac{1}{N}\sum_{t=1}^{N}\log{P(w_t|w_1,...,w_{t-1})} $

### 继续预训练(Continual Pretrain)

- `继续预训练(Continual Pretrain)`场景主要利用弱监督数据，将其构造成“问-答”对的多轮对话形式，采用带loss mask的Next Token Prediction对文本的部分token进行优化，从而为模型注入特定场景的知识；
- 数据格式：jsonl格式，每行一条样本，必须存在`messages`字段，存储`user`和`assistant`的对话内容，对话中主要有三种角色，即`system`、`user`、`assistant`。
```jsonl
{
    "messages": [
        {"role": "system", "content": "xxx"},
        {"role": "user", "content": "xxx"},
        {"role": "assistant", "content": "xxx"},
        ...
    ]
}
```
- 运行命令：
```bash
cd safetygpt
bash distributed_train.sh configs/yi_6b_continual_pretrain_stage.yml
```
- 优化目标：采用Next Token Prediction优化`assistant`的tokens。
$$
\mathcal{L}=-\frac{1}{N}\sum_{t=1}^{N}\log{P(w_{assist}|w_{user}, w_{system})}
$$

### 监督微调(Supervised Finetune)

- `监督微调(Supervised Finetune)`场景主要利用高质量的下游任务数据，将其构造成单轮“问-答”形式，采用带loss mask的Next Token Prediction对文本的“答案”部分进行优化，让模型解决特定任务。

- 数据格式：jsonl格式，每行一条样本，必须存在`instruction`、`input`、`output`字段。
```jsonl
{
    "instruction": "xxx",
    "input": "xxx",
    "output": "xxx"
}
```
- 运行命令：
```bash
cd safetygpt
bash distributed_train.sh configs/yi_6b_sft_stage.yml
```
- 优化目标：采用Next Token Prediction优化`output`的tokens。
$$
\mathcal{L}=-\frac{1}{N}\sum_{t=1}^{N}\log{P(w_{output}|w_{instruction}, w_{input})}
$$

