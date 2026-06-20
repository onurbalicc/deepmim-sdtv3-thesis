#!/bin/bash

# Updated STDV3 v2 pretraining
# Model idea: Binary Q,K + XNOR + Gray2D + Log-PE 2D
# Standard thesis setting:
# - Pretraining: 400 epochs
# - GPUs: 2
# - Same pretraining recipe as Updated STDV3 v1

set -e

cd /media/homes/balic/projects/spike-driven-transformer-v3/SDT_V3/Classification/Model_Large

OUT_DIR="07_updated_stdv3_v2_pretrain_binaryqk_xnor_gray2d_logpe2d_2gpu_400ep"
DATA_PATH="/media/homes/rafae/uni/datasets/ImageNet_Sub"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  main_pretrain.py \
  --model spikmae_deepmim_12_512 \
  --epochs 400 \
  --batch_size 32 \
  --accum_iter 8 \
  --blr 0.00015 \
  --lr 0.0003 \
  --weight_decay 0.05 \
  --warmup_epochs 20 \
  --mask_ratio 0.5 \
  --input_size 224 \
  --data_path "$DATA_PATH" \
  --output_dir "$OUT_DIR" \
  --log_dir "$OUT_DIR" \
  --model_mode ms \
  --num_workers 2
