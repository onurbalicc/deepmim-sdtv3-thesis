#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# ADE20K Semantic Segmentation
# Setting: ON ORIGINAL Spikformer
# Model: Update 1
# DeepMIM + Log-PE 2D + Quadratic MultiSpike
# Metric: mIoU
# Schedule: 160k iterations, validation every 8k iterations
# ============================================================

source /media/homes/balic/anaconda3/etc/profile.d/conda.sh
conda activate sdt_det

cd /media/homes/balic/projects/spike-driven-transformer-v3/SDT_V3/Segmentation

CONFIG="configs/sem_sdt/fpn_SDT_512x512_55M_T4_ade20k.py"
CKPT="../Classification/Model_Large/04_updated_stdv3_pretrain_deepmim_logpe_400ep/checkpoint-390.pth"
WORK_DIR="work_dirs/04_seg_on_original_update1_ade20k_160k"

GPUS=2
PORT="${PORT:-29532}"
GPUS_TO_USE="${GPUS_TO_USE:-5,6}"

echo "=== ADE20K Segmentation: Update 1 on ORIGINAL Spikformer ==="
echo "Environment: sdt_det"
echo "Config:      $CONFIG"
echo "Checkpoint:  $CKPT"
echo "Work dir:    $WORK_DIR"
echo "GPUs:        $GPUS_TO_USE"
echo "Metric:      mIoU"
echo ""

if [ ! -f "$CKPT" ]; then
  echo "ERROR: Checkpoint not found: $CKPT"
  exit 1
fi

if [ -d "$WORK_DIR" ]; then
  echo "ERROR: Work dir already exists: $WORK_DIR"
  echo "Refusing to overwrite an existing run."
  exit 1
fi

CUDA_VISIBLE_DEVICES="$GPUS_TO_USE" PORT="$PORT" bash tools/dist_train.sh "$CONFIG" "$GPUS" \
  --work-dir "$WORK_DIR" \
  --cfg-options model.backbone.init_cfg.checkpoint="$CKPT" train_dataloader.batch_size=1
