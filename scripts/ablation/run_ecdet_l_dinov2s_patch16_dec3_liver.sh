#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="$ROOT_DIR/ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver.yml"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_noreg}"
DATA_ROOT="/cobot/Data/Lesion_det/det_liver"
SEED="${SEED:-42}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/train.log"
exec > >(tee -a "$LOG_FILE") 2>&1

for path in "$CONFIG" \
  "$DATA_ROOT/img" \
  "$DATA_ROOT/annotations/train.json" \
  "$DATA_ROOT/annotations/valid.json" \
  "$DATA_ROOT/annotations/test.json" \
  "/cobot/Code/xiangshaochong/checkpoints/dinov2/dinov2_vits14_pretrain.pth"; do
  test -e "$path" || { echo "Missing required path: $path" >&2; exit 1; }
done

export PYTHONPATH="$ROOT_DIR/ecdetseg${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "Start: $(date)"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "Seed: $SEED"
echo "Official ECDet-L reference: ecdetseg/configs/ecdet/ecdet_l.yml"
echo "Resolved baseline config:"
sed -n '1,260p' "$CONFIG"

if [[ -f "$OUTPUT_DIR/last.pth" ]]; then
  echo "Resuming from $OUTPUT_DIR/last.pth"
  LOAD_ARGS=(-r "$OUTPUT_DIR/last.pth")
else
  LOAD_ARGS=()
fi

cd "$ROOT_DIR"
"$PYTHON_BIN" -m torch.distributed.run --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  ecdetseg/train.py \
  -c "$CONFIG" \
  "${LOAD_ARGS[@]}" \
  --use-amp \
  --seed "$SEED" \
  --output-dir "$OUTPUT_DIR"

BEST_CKPT="$OUTPUT_DIR/best.pth"
if [[ ! -f "$BEST_CKPT" ]]; then
  BEST_CKPT="$OUTPUT_DIR/last.pth"
fi
test -f "$BEST_CKPT" || { echo "No best/last checkpoint produced" >&2; exit 1; }

echo "Final test evaluation: $(date)"
"$PYTHON_BIN" -m torch.distributed.run --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  ecdetseg/train.py \
  -c "$CONFIG" \
  -r "$BEST_CKPT" \
  --test-only \
  --output-dir "$OUTPUT_DIR" \
  -u "val_dataloader.dataset.ann_file=$DATA_ROOT/annotations/test.json"

echo "Completed: $(date)"
