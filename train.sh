#!/bin/bash

# Number of GPUs per GPU worker
GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

# Number of GPU workers, for single-worker training, please set to 1
NNODES=${NNODES:-1}

# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
NODE_RANK=${NODE_RANK:-0}

# The ip address of the rank-0 worker, for single-worker training, please set to localhost
MASTER_ADDR=${MASTER_ADDR:-localhost}

# The port for communication
MASTER_PORT=${MASTER_PORT:-6001}

torchrun --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank=$NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT src/inference.py --params=$1
