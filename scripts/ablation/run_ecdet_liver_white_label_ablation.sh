#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/cobot/miniforge3/envs/lw-detr/bin/python}"
EXPERIMENT="${WHITE_LABEL_EXPERIMENT:?Set WHITE_LABEL_EXPERIMENT}"
DATA_ROOT="/cobot/Data/Lesion_det/det_liver"
SEED="${SEED:-42}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

case "$EXPERIMENT" in
  remove_labels_keep_images)
    CONFIG="$ROOT_DIR/ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver_white_removed_keep_images.yml"
    DEFAULT_OUTPUT="$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_dec3_liver_white_removed_keep_images_bs32_2gpu_seed42" ;;
  remove_white_images)
    CONFIG="$ROOT_DIR/ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver_white_images_removed.yml"
    DEFAULT_OUTPUT="$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_dec3_liver_white_images_removed_bs32_2gpu_seed42" ;;
  neutral_ignore)
    CONFIG="$ROOT_DIR/ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver_white_neutral_ignore.yml"
    DEFAULT_OUTPUT="$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_dec3_liver_white_neutral_ignore_bs32_2gpu_seed42" ;;
  *) echo "Unknown WHITE_LABEL_EXPERIMENT=$EXPERIMENT" >&2; exit 2 ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT}"
VALID_ANN="$DATA_ROOT/annotations/valid_white_ignore_remap.json"
TEST_ANN="$DATA_ROOT/annotations/test_white_ignore_remap.json"
mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$OUTPUT_DIR/train.log") 2>&1

for path in "$PYTHON_BIN" "$CONFIG" "$DATA_ROOT/img" "$VALID_ANN" "$TEST_ANN" \
  "$DATA_ROOT/annotations/train_ignore_9_12.json" \
  "$DATA_ROOT/annotations/train_no_white_images_remap.json" \
  "$DATA_ROOT/annotations/train_white_ignore_remap.json" \
  "/cobot/Code/xiangshaochong/checkpoints/dinov2/dinov2_vits14_pretrain.pth"; do
  test -e "$path" || { echo "Missing required path: $path" >&2; exit 1; }
done

export PYTHONPATH="$ROOT_DIR/ecdetseg${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
NOFILE_LIMIT="${NOFILE_LIMIT:-65536}"
CURRENT_NOFILE="$(ulimit -n)"
if [[ "$CURRENT_NOFILE" != unlimited && "$CURRENT_NOFILE" -lt "$NOFILE_LIMIT" ]]; then
  ulimit -n "$NOFILE_LIMIT"
fi

echo "Start: $(date)"
echo "Experiment: $EXPERIMENT"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "Global batch: 32"
echo "Seed: $SEED"

if [[ -f "$OUTPUT_DIR/last.pth" ]]; then
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

echo "Final ignore-aware test evaluation: $(date)"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" \
  ecdetseg/train.py -c "$CONFIG" -r "$BEST_CKPT" --test-only --output-dir "$OUTPUT_DIR" \
  -u "val_dataloader.dataset.ann_file=$TEST_ANN"
echo "Completed: $(date)"
