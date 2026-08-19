#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENT="${EC_ABLATION_EXPERIMENT:?Set EC_ABLATION_EXPERIMENT}"
DATA_ROOT="/cobot/Data/Lesion_det/det_liver"
SEED="${SEED:-42}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

case "$EXPERIMENT" in
  no_mosaic_focal)
    CONFIG_NAME="ecdet_l_dinov2s_patch16_dec3_liver_newf1_no_mosaic_focal.yml"
    OUTPUT_NAME="newf1_ec_no_mosaic_focal_bs32_2gpu_seed42" ;;
  no_mosaic_focal_no_dfine)
    CONFIG_NAME="ecdet_l_dinov2s_patch16_dec3_liver_newf1_no_mosaic_focal_no_dfine.yml"
    OUTPUT_NAME="newf1_ec_no_mosaic_focal_no_dfine_bs32_2gpu_seed42" ;;
  no_mosaic_focal_no_dfine_no_cdn)
    CONFIG_NAME="ecdet_l_dinov2s_patch16_dec3_liver_newf1_no_mosaic_focal_no_dfine_no_cdn.yml"
    OUTPUT_NAME="newf1_ec_no_mosaic_focal_no_dfine_no_cdn_bs32_2gpu_seed42" ;;
  rf_neck_p4_points6)
    CONFIG_NAME="ecdet_l_dinov2s_patch16_dec3_liver_newf1_rf_neck_p4_points6.yml"
    OUTPUT_NAME="newf1_ec_rf_neck_p4_points6_bs32_2gpu_seed42" ;;
  rf_neck_p4_points6_no_dfine)
    CONFIG_NAME="ecdet_l_dinov2s_patch16_dec3_liver_newf1_rf_neck_p4_points6_no_dfine.yml"
    OUTPUT_NAME="newf1_ec_rf_neck_p4_points6_no_dfine_bs32_2gpu_seed42" ;;
  rf_neck_p4_points6_no_dfine_no_cdn)
    CONFIG_NAME="ecdet_l_dinov2s_patch16_dec3_liver_newf1_rf_neck_p4_points6_no_dfine_no_cdn.yml"
    OUTPUT_NAME="newf1_ec_rf_neck_p4_points6_no_dfine_no_cdn_bs32_2gpu_seed42" ;;
  *) echo "Unknown EC_ABLATION_EXPERIMENT=$EXPERIMENT" >&2; exit 2 ;;
esac

[[ "$NPROC_PER_NODE" == 2 ]] || {
  echo "These paired experiments require NPROC_PER_NODE=2" >&2
  exit 2
}

CONFIG="$ROOT_DIR/ecdetseg/configs/ecdet/$CONFIG_NAME"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/ablation/$OUTPUT_NAME}"
TEST_ANN_FILE="$DATA_ROOT/annotations/test_ignore_9_12.json"

for path in "$CONFIG" "$DATA_ROOT/img" \
  "$DATA_ROOT/annotations/train_ignore_9_12.json" \
  "$DATA_ROOT/annotations/valid_ignore_9_12.json" \
  "$TEST_ANN_FILE" \
  "/cobot/Code/xiangshaochong/checkpoints/dinov2/dinov2_vits14_pretrain.pth"; do
  test -e "$path" || { echo "Missing required path: $path" >&2; exit 1; }
done

mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$OUTPUT_DIR/train.log") 2>&1

export PYTHONPATH="$ROOT_DIR/ecdetseg${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

NOFILE_LIMIT="${NOFILE_LIMIT:-65536}"
CURRENT_NOFILE="$(ulimit -n)"
if [[ "$CURRENT_NOFILE" != unlimited && "$CURRENT_NOFILE" -lt "$NOFILE_LIMIT" ]]; then
  ulimit -n "$NOFILE_LIMIT" || exit 1
fi

echo "Start: $(date)"
echo "Experiment: $EXPERIMENT"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "Global batch: 32"
echo "Seed: $SEED"
echo "Best checkpoint / early stopping metric: mAP50"
echo "Validation reports: F1@0.50, F1@0.95, F1@0.50:0.95 mean, mAP50, mAP50-95"

LOAD_ARGS=()
if [[ -f "$OUTPUT_DIR/last.pth" ]]; then
  echo "Resuming from $OUTPUT_DIR/last.pth"
  LOAD_ARGS=(-r "$OUTPUT_DIR/last.pth")
fi

cd "$ROOT_DIR"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=2 \
  ecdetseg/train.py -c "$CONFIG" "${LOAD_ARGS[@]}" --use-amp --seed "$SEED" \
  --output-dir "$OUTPUT_DIR"

BEST_CKPT="$OUTPUT_DIR/best.pth"
[[ -f "$BEST_CKPT" ]] || BEST_CKPT="$OUTPUT_DIR/last.pth"
test -f "$BEST_CKPT" || { echo "No best/last checkpoint produced" >&2; exit 1; }

FINAL_TEST_DIR="$OUTPUT_DIR/final_test"
mkdir -p "$FINAL_TEST_DIR"
echo "Final aligned-F1 test evaluation: $(date)"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=2 \
  ecdetseg/train.py -c "$CONFIG" -r "$BEST_CKPT" --test-only \
  --output-dir "$FINAL_TEST_DIR" \
  -u "val_dataloader.dataset.ann_file=$TEST_ANN_FILE"
echo "Completed: $(date)"
