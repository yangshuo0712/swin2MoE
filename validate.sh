#!/usr/bin/env bash
#
# run_ddp.sh — launch Swin2‑MoSE training on 2 GPUs (device 1 & 2) with PyTorch DDP
#
# Usage: bash run_ddp.sh
# Make sure you are inside the project root or edit PROJECT_DIR below.

set -euo pipefail

EPOCH=${1:-2}

# Absolute path to this script
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Select visible GPUs (physical IDs 1 and 2)
export CUDA_VISIBLE_DEVICES=1,2
export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

# Launch training with torchrun
torchrun \
  --nnodes 1 \
  --nproc-per-node 2 \
  --master_addr 127.0.0.1 \
  --master_port 29501 \
  src/main.py \
    --config cfg_n/sen2venus_exp6_2x_v3.yml \
    --phase test \
    --batch_size 2 \
    --num_workers 16 \
    --epoch "$EPOCH" \
    --output output/2x_DDP_v3_80 \
    --distributed true \
    --AMP true \
    # --eval_method bicubic
