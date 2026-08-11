#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENT="${EC_ABLATION_EXPERIMENT:?Set EC_ABLATION_EXPERIMENT=no_fdr_decode|no_go_ddf|no_fdr_no_go_ddf|no_cdn_ignore9|no_go_lsd_ignore9|no_dfine_ignore9|continuous_with_go_ignore9|no_mosaic|rf_neck|rf_neck_p4_ignore9|ec_full_ignore9|baseline_dec4_ignore9|baseline_dec5_ignore9}"
DATA_ROOT="/cobot/Data/Lesion_det/det_liver"
SEED="${SEED:-42}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

case "$EXPERIMENT" in
  no_fdr_decode)
    CONFIG="$ROOT_DIR/ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver_no_fdr_decode.yml"
    DEFAULT_OUTPUT="$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_no_fdr_decode" ;;
  no_go_ddf)
    CONFIG="$ROOT_DIR/ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver_no_go_ddf.yml"
    DEFAULT_OUTPUT="$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_no_go_ddf" ;;
  no_fdr_no_go_ddf)
    CONFIG="$ROOT_DIR/ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver_no_fdr_no_go_ddf.yml"
    DEFAULT_OUTPUT="$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_no_fdr_no_go_ddf" ;;
  no_cdn_ignore9)
    CONFIG="$ROOT_DIR/ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver_no_cdn_ignore9.yml"
    DEFAULT_OUTPUT="$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_no_cdn_ignore9" ;;
  no_go_lsd_ignore9)
    CONFIG="$ROOT_DIR/ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver_no_go_ddf_ignore9.yml"
    DEFAULT_OUTPUT="$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_no_go_ddf_ignore9" ;;
  no_dfine_ignore9)
    CONFIG="$ROOT_DIR/ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver_no_dfine_ignore9.yml"
    DEFAULT_OUTPUT="$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_no_dfine_ignore9" ;;
  continuous_with_go_ignore9)
    CONFIG="$ROOT_DIR/ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver_continuous_with_go_ignore9.yml"
    DEFAULT_OUTPUT="$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_continuous_with_go_ignore9" ;;
  no_mosaic)
    CONFIG="$ROOT_DIR/ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver_no_mosaic.yml"
    DEFAULT_OUTPUT="$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_no_mosaic" ;;
  rf_neck)
    CONFIG="$ROOT_DIR/ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver_rf_neck.yml"
    DEFAULT_OUTPUT="$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_rf_neck" ;;
  rf_neck_p4_ignore9)
    CONFIG="$ROOT_DIR/ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver_rf_neck_p4_ignore9.yml"
    DEFAULT_OUTPUT="$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_rf_neck_p4_ignore9" ;;
  ec_full_ignore9)
    CONFIG="$ROOT_DIR/ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver_ec_full_ignore9.yml"
    DEFAULT_OUTPUT="$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_ec_full_ignore9" ;;
  baseline_dec4_ignore9)
    CONFIG="$ROOT_DIR/ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec4_liver_ignore9.yml"
    DEFAULT_OUTPUT="$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_patch16_dec4_liver_baseline_2gpu_seed42" ;;
  baseline_dec5_ignore9)
    CONFIG="$ROOT_DIR/ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec5_liver_ignore9.yml"
    DEFAULT_OUTPUT="$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_patch16_dec5_liver_baseline_2gpu_seed42" ;;
  *) echo "Unknown EC_ABLATION_EXPERIMENT=$EXPERIMENT" >&2; exit 2 ;;
esac
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT}"
TEST_ANN_FILE="$DATA_ROOT/annotations/test.json"
if [[ "$EXPERIMENT" == rf_neck_p4_ignore9 || "$EXPERIMENT" == ec_full_ignore9 || "$EXPERIMENT" == no_cdn_ignore9 || "$EXPERIMENT" == no_go_lsd_ignore9 || "$EXPERIMENT" == no_dfine_ignore9 || "$EXPERIMENT" == continuous_with_go_ignore9 || "$EXPERIMENT" == baseline_dec4_ignore9 || "$EXPERIMENT" == baseline_dec5_ignore9 ]]; then
  TEST_ANN_FILE="$DATA_ROOT/annotations/test_ignore_9_12.json"
fi

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/train.log"
exec > >(tee -a "$LOG_FILE") 2>&1

for path in "$CONFIG" "$DATA_ROOT/img" "$DATA_ROOT/annotations/train.json" \
  "$DATA_ROOT/annotations/valid.json" "$DATA_ROOT/annotations/test.json" \
  "/cobot/Code/xiangshaochong/checkpoints/dinov2/dinov2_vits14_pretrain.pth"; do
  test -e "$path" || { echo "Missing required path: $path" >&2; exit 1; }
done
if [[ "$EXPERIMENT" == rf_neck_p4_ignore9 || "$EXPERIMENT" == ec_full_ignore9 || "$EXPERIMENT" == no_cdn_ignore9 || "$EXPERIMENT" == no_go_lsd_ignore9 || "$EXPERIMENT" == no_dfine_ignore9 || "$EXPERIMENT" == continuous_with_go_ignore9 || "$EXPERIMENT" == baseline_dec4_ignore9 || "$EXPERIMENT" == baseline_dec5_ignore9 ]]; then
  for path in "$DATA_ROOT/annotations/train_ignore_9_12.json" \
    "$DATA_ROOT/annotations/valid_ignore_9_12.json" "$TEST_ANN_FILE"; do
    test -e "$path" || { echo "Missing strict 9-class annotation: $path" >&2; exit 1; }
  done
fi

export PYTHONPATH="$ROOT_DIR/ecdetseg${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "Start: $(date)"
echo "Experiment: $EXPERIMENT"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "Seed: $SEED"
sed -n '1,180p' "$CONFIG"

if [[ -f "$OUTPUT_DIR/last.pth" ]]; then
  echo "Resuming from $OUTPUT_DIR/last.pth"
  LOAD_ARGS=(-r "$OUTPUT_DIR/last.pth")
else
  LOAD_ARGS=()
fi

cd "$ROOT_DIR"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" \
  ecdetseg/train.py -c "$CONFIG" "${LOAD_ARGS[@]}" --use-amp --seed "$SEED" --output-dir "$OUTPUT_DIR"

BEST_CKPT="$OUTPUT_DIR/best.pth"
[[ -f "$BEST_CKPT" ]] || BEST_CKPT="$OUTPUT_DIR/last.pth"
test -f "$BEST_CKPT" || { echo "No best/last checkpoint produced" >&2; exit 1; }

echo "Final test evaluation: $(date)"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" \
  ecdetseg/train.py -c "$CONFIG" -r "$BEST_CKPT" --test-only --output-dir "$OUTPUT_DIR" \
  -u "val_dataloader.dataset.ann_file=$TEST_ANN_FILE"
echo "Completed: $(date)"
