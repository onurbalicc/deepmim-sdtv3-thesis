#!/bin/bash

# Updated STDV3 v1 fine-tuning from checkpoint-390
# Standard thesis setting:
# - Fine-tuning: 100 epochs
# - GPUs: 2
# - Same fine-tuning recipe as baseline ckpt395 fine-tuning

set -e

cd /media/homes/balic/projects/spike-driven-transformer-v3/SDT_V3/Classification/Model_Large

OUT_DIR="/media/homes/balic/projects/spike-driven-transformer-v3/SDT_V3/Classification/Model_Large/08_updated_stdv3_v1_finetune_ckpt390_2gpu_100ep"
DATA_PATH="/media/homes/rafae/uni/datasets/ImageNet_Sub"
FINETUNE_CKPT="/media/homes/balic/projects/spike-driven-transformer-v3/SDT_V3/Classification/Model_Large/04_updated_stdv3_pretrain_deepmim_logpe_400ep/checkpoint-390.pth"

if [ -f "$OUT_DIR/log.txt" ]; then
  echo "ERROR: $OUT_DIR/log.txt already exists. Refusing to overwrite an existing run."
  exit 1
fi

mkdir -p "$OUT_DIR"

CUDA_VISIBLE_DEVICES=3,7 torchrun --standalone --nproc_per_node=2 \
  main_finetune.py \
  --model spikformer12_512 \
  --data_path "$DATA_PATH" \
  --finetune "$FINETUNE_CKPT" \
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
  --drop_path 0.1 \
  --reprob 0.25 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --nb_classes 100 \
  --model_mode ms \
  --pin_mem \
  --dist_eval
