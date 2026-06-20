#!/bin/bash

set -e
set -o pipefail

cd /media/homes/balic/projects/spike-driven-transformer-v3/SDT_V3/Classification/Model_Large

OUT_DIR="15_updated_v2_linear_probe_ckpt390_original_2gpu_100ep"
DATA_PATH="/media/homes/rafae/uni/datasets/ImageNet_Sub"
CKPT="/media/homes/balic/projects/spike-driven-transformer-v3/SDT_V3/Classification/Model_Large/09_updated_stdv3_v2_debug_pretrain_2gpu_400ep/checkpoint-390.pth"

if [ -f "$OUT_DIR/log.txt" ]; then
  echo "ERROR: $OUT_DIR/log.txt already exists. Refusing to overwrite an existing run."
  exit 1
fi

mkdir -p "$OUT_DIR"

CUDA_VISIBLE_DEVICES=1,3 torchrun --standalone --nproc_per_node=2 \
  main_finetune_linear_probe_jafar.py \
  --spikformer_type original \
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
