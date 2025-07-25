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

# Launch training with torchrun
torchrun \
  --nnodes 1 \
  --nproc-per-node 2 \
  --master_addr 127.0.0.1 \
  --master_port 29500 \
  src/main.py \
    --config cfgs/swin2_mose/sen2venus_2x_s2m.yml \
    --phase train \
    --batch_size 1 \
    --num_workers 4 \
    --distributed true \
    --output ./output/2x_ddp
