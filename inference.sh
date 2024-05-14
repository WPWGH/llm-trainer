#!/bin/bash
PORTS=$ARNOLD_WORKER_0_PORT 
PORT=(${PORTS//,/ })

# 使用硬编码的端口号
AVAILABLE_PORT=12345

torchrun \
    --nproc_per_node $ARNOLD_WORKER_GPU \
    --nnodes $ARNOLD_WORKER_NUM \
    --node_rank=$ARNOLD_ID \
    --master_addr $ARNOLD_WORKER_0_HOST \
    --master_port $AVAILABLE_PORT src/inference.py --params=$1

