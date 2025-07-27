#!/usr/bin/env bash
#
# run_ddp.sh — launch Swin2‑MoSE training on 2 GPUs (device 1 & 2) with PyTorch DDP
#
# Usage: bash run_ddp.sh
# Make sure you are inside the project root or edit PROJECT_DIR below.

set -euo pipefail

# Absolute path to this script
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Select visible GPUs (physical IDs 1 and 2)
export CUDA_VISIBLE_DEVICES=1,2
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Launch training with torchrun
torchrun \
  --nnodes 1 \
  --nproc-per-node 2 \
  --master_addr 127.0.0.1 \
  --master_port 29500 \
  src/main.py \
    --config cfg_n/sen2venus_exp4_2x_v5.yml \
    --phase train \
    --batch_size 2 \
    --num_workers 4 \
    --distributed true \
    --AMP true \
    --output ./output/4x_ddp
