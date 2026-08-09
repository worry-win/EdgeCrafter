#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT="${1:?Usage: $0 full|no_cdn|no_fdr_decode|no_go_ddf|no_fdr_no_go_ddf|no_mosaic|rf_neck}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
PYTHON_BIN="${PYTHON_BIN:-/cobot/miniforge3/envs/lw-detr/bin/python}"
SPLIT="${EVAL_SPLIT:-test}"
DATA_ROOT="/cobot/Data/Lesion_det/det_liver"

case "$EXPERIMENT" in
  full)
    CONFIG="ecdet_l_dinov2s_patch16_dec3_liver.yml"
    OUTPUT="ecdet_l_dinov2s_patch16_dec3_liver_noreg" ;;
  no_cdn)
    CONFIG="ecdet_l_dinov2s_patch16_dec3_liver_no_cdn.yml"
    OUTPUT="ecdet_l_dinov2s_patch16_dec3_liver_noreg_no_cdn" ;;
  no_fdr_decode|no_go_ddf|no_fdr_no_go_ddf|no_mosaic|rf_neck)
    CONFIG="ecdet_l_dinov2s_patch16_dec3_liver_${EXPERIMENT}.yml"
    OUTPUT="ecdet_l_dinov2s_patch16_dec3_liver_${EXPERIMENT}" ;;
  *) echo "Unknown experiment: $EXPERIMENT" >&2; exit 2 ;;
esac

CONFIG="$ROOT_DIR/ecdetseg/configs/ecdet/$CONFIG"
OUTPUT_DIR="$ROOT_DIR/outputs/ablation/$OUTPUT"
CHECKPOINT="$OUTPUT_DIR/best.pth"
ANN_FILE="$DATA_ROOT/annotations/${SPLIT}.json"
LOG_FILE="$OUTPUT_DIR/eval_${SPLIT}_ignore_9_12.log"

for path in "$CONFIG" "$CHECKPOINT" "$ANN_FILE"; do
  test -f "$path" || { echo "Missing required file: $path" >&2; exit 1; }
done

export PYTHONPATH="$ROOT_DIR/ecdetseg${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
cd "$ROOT_DIR"

echo "Experiment: $EXPERIMENT" | tee "$LOG_FILE"
echo "Split: $SPLIT" | tee -a "$LOG_FILE"
echo "Checkpoint: $CHECKPOINT" | tee -a "$LOG_FILE"
echo "Ignored category IDs: 9,10,11,12" | tee -a "$LOG_FILE"

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" \
  ecdetseg/train.py -c "$CONFIG" -r "$CHECKPOINT" --test-only --output-dir "$OUTPUT_DIR" \
  -u "val_dataloader.dataset.ann_file=$ANN_FILE" \
     "evaluator.ignore_category_ids=[9,10,11,12]" \
  2>&1 | tee -a "$LOG_FILE"
