#!/usr/bin/env bash
set -euo pipefail

# Submit the three EC DINOv2-S decoder-depth controls. Each training job uses
# two GPUs with total_batch_size=32, i.e. per-rank batch size 16.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMUX_BIN="${TMUX_BIN:-$ROOT_DIR/.local/bin/tmux}"
export LD_LIBRARY_PATH="$ROOT_DIR/.local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
WAIT_SCRIPT="$ROOT_DIR/scripts/ablation/wait_ecdet_l_dinov2s_patch16_dec3_liver_tmux.sh"
TRAIN_SCRIPT="$ROOT_DIR/scripts/ablation/run_ecdet_l_dinov2s_patch16_dec3_liver_dfine_ablation.sh"

test -x "$TMUX_BIN" || { echo "tmux binary not found: $TMUX_BIN" >&2; exit 1; }
test -x "$WAIT_SCRIPT" || { echo "waiter is not executable: $WAIT_SCRIPT" >&2; exit 1; }
test -x "$TRAIN_SCRIPT" || { echo "training script is not executable: $TRAIN_SCRIPT" >&2; exit 1; }

start_waiter() {
  local experiment="$1" session="$2" node="$3" gpus="$4" output="$5"
  if env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" "$TMUX_BIN" has-session -t "$session" 2>/dev/null; then
    echo "Already running: $session"
    return 0
  fi
  local command
  command="EC_ABLATION_EXPERIMENT=$experiment TRAIN_SCRIPT=$TRAIN_SCRIPT TRAIN_OUTPUT_DIR=$output TMUX_SESSION=$session TARGET_NODES='$node' GPU_INDICES='$gpus' REQUIRED_GPU_COUNT=2 POLL_INTERVAL_SEC=300 SEED=42 WAIT_LOG=$output/wait.log bash '$WAIT_SCRIPT'"
  env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" "$TMUX_BIN" new-session -d -s "$session" "$command"
  echo "Started waiter $session -> $experiment, target $node GPUs $gpus, output $output"
}

# The nodes are currently empty. Keep one experiment per node so the three
# waiters cannot race for the same cards; each waiter still rechecks memory
# and tmux state every five minutes before launching.
start_waiter ec_full_ignore9 baseline_ec_dec3 cu02 0,1 "$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_baseline_2gpu_seed42"
start_waiter baseline_dec4_ignore9 baseline_ec_dec4 cu03 0,1 "$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_patch16_dec4_liver_baseline_2gpu_seed42"
start_waiter baseline_dec5_ignore9 baseline_ec_dec5 cu04 0,1 "$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_patch16_dec5_liver_baseline_2gpu_seed42"

echo "Three baseline waiters submitted; polling interval: 300s; per-rank batch: 16; seed: 42"
