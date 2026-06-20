#!/bin/bash

set -e

DATA_PATH="/media/homes/rafae/uni/datasets/ImageNet_Sub"
CKPT="/media/homes/balic/projects/spike-driven-transformer-v3/SDT_V3/Classification/Model_Large/09_updated_stdv3_v2_debug_pretrain_2gpu_400ep/checkpoint-399.pth"
OUT_DIR="11_updated_stdv3_v2_finetune_ckpt399_2gpu_100ep"

mkdir -p "$OUT_DIR"

CUDA_VISIBLE_DEVICES=0,2 torchrun --standalone --nproc_per_node=2 \
  main_finetune.py \
  --model spikformer12_512 \
  --data_path "$DATA_PATH" \
  --finetune "$CKPT" \
  --output_dir "$OUT_DIR" \
  --log_dir "$OUT_DIR" \
  --epochs 100 \
  --batch_size 32 \
  --accum_iter 4 \
  --num_workers 8 \
  --blr 0.0006 \
  --weight_decay 0.05 \
  --warmup_epochs 10 \
  --layer_decay 0.75 \
  --drop_path 0.0 \
  --reprob 0.0 \
  --mixup 0.0 \
  --cutmix 0.0 \
  --nb_classes 100 \
  --model_mode ms \
  --pin_mem \
  --dist_eval 2>&1 | tee "$OUT_DIR/terminal_output.txt"
