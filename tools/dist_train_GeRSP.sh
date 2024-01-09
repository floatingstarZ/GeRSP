#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.100 \
    --nproc_per_node 8 \
    --master_port 29500 \
    $(dirname "$0")/train.py \
    configs/selfsup/GeRSP/GeRSP.py \
    --seed 0 \
    --launcher pytorch ${@:3} \
    --work-dir results/GeRSP \
