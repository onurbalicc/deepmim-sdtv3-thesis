#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Update 3 pretraining
# DeepMIM + Log-PE 2D + Gray-PE 2D + alpha-XNOR with Q,K LIF
# Corrected alpha-XNOR and scaling version from Jafar
# ============================================================

PROJECT_DIR="/media/homes/balic/projects/spike-driven-transformer-v3/SDT_V3/Classification/Model_Large"
DATA_PATH="/media/homes/rafae/uni/datasets/ImageNet_Sub"

MODEL_FILE="Updated_STDV3_alpha_xnor_gray_logpe_deepmim.py"
MODEL_IMPORT="import Updated_STDV3_alpha_xnor_gray_logpe_deepmim as Updated_STDV3"

OUT_DIR="18_update3_pretrain_alpha_xnor_gray_logpe_deepmim_2gpu_400ep"

# Default GPUs. Can be overridden like:
# GPUS=3,7 ./run_pretrain_update3_alpha_xnor_gray_logpe_deepmim_2gpu_400ep.sh
GPUS="${GPUS:-3,7}"

cd "$PROJECT_DIR"

echo "=== Update 3 pretraining ==="
echo "Project dir:  $PROJECT_DIR"
echo "Data path:    $DATA_PATH"
echo "Model file:   $MODEL_FILE"
echo "Output dir:   $OUT_DIR"
echo "GPUs:         $GPUS"
echo ""

echo "=== Safety checks ==="

if [ ! -f "$MODEL_FILE" ]; then
  echo "ERROR: Model file not found: $MODEL_FILE"
  exit 1
fi

if [ ! -f "main_pretrain.py" ]; then
  echo "ERROR: main_pretrain.py not found."
  exit 1
fi

if [ -f "$OUT_DIR/log.txt" ]; then
  echo "ERROR: $OUT_DIR/log.txt already exists. Refusing to overwrite an existing run."
  exit 1
fi

echo ""
echo "=== Import and factory test ==="
python - <<'PY'
import Updated_STDV3_alpha_xnor_gray_logpe_deepmim as m

required = [
    "spikmae_deepmim_12_512",
    "spikmae_deepmim_12_768",
    "spikmae_deepmim_12_768_no_rpe",
    "MS_Attention_AlphaXNOR_GRAY_LogPE",
]

for name in required:
    print(name, "=", hasattr(m, name))
    assert hasattr(m, name), f"Missing required object: {name}"

print("Import and factory check OK")
PY

echo ""
echo "=== Backup main_pretrain.py ==="
BACKUP_FILE="main_pretrain.py.backup_before_update3_alpha_xnor_gray_$(date +%Y%m%d_%H%M%S)"
cp main_pretrain.py "$BACKUP_FILE"
echo "Backup created: $BACKUP_FILE"

echo ""
echo "=== Patch main_pretrain.py import ==="
python - <<'PY'
from pathlib import Path

path = Path("main_pretrain.py")
text = path.read_text()

new_import = "import Updated_STDV3_alpha_xnor_gray_logpe_deepmim as Updated_STDV3"

lines = text.splitlines()
old_imports = []
changed = False
out = []

for line in lines:
    stripped = line.strip()
    if stripped.startswith("import ") and " as Updated_STDV3" in stripped:
        old_imports.append(line)
        if not changed:
            out.append(new_import)
            changed = True
        else:
            out.append(line)
    else:
        out.append(line)

if not changed:
    raise SystemExit("ERROR: Could not find current Updated_STDV3 import line in main_pretrain.py")

path.write_text("\n".join(out) + "\n")

print("Old Updated_STDV3 import candidates:")
for item in old_imports:
    print("  " + item)

print("New selected import:")
print("  " + new_import)
PY

echo ""
echo "=== Confirm selected import ==="
grep -n "import .*Updated_STDV3" main_pretrain.py | head -20

echo ""
echo "=== Create output folder ==="
mkdir -p "$OUT_DIR"

echo ""
echo "=== Start Update 3 pretraining ==="
CUDA_VISIBLE_DEVICES="$GPUS" torchrun --standalone --nproc_per_node=2 \
  main_pretrain.py \
  --model spikmae_deepmim_12_512 \
  --epochs 400 \
  --batch_size 32 \
  --accum_iter 8 \
  --blr 0.00015 \
  --weight_decay 0.05 \
  --warmup_epochs 20 \
  --mask_ratio 0.5 \
  --data_path "$DATA_PATH" \
  --output_dir "$OUT_DIR" \
  --log_dir "$OUT_DIR" \
  --model_mode ms \
  --num_workers 2 \
  2>&1 | tee "$OUT_DIR/terminal_output.txt"
