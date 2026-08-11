#!/usr/bin/env bash
set -euo pipefail

# Submit the strict-9, 3-layer EC-full/CDN/D-FINE controls. Every job uses
# two GPUs and total_batch_size=32, so the per-rank batch size is 16.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMUX_BIN="${TMUX_BIN:-$ROOT_DIR/.local/bin/tmux}"
export LD_LIBRARY_PATH="$ROOT_DIR/.local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
WAIT_SCRIPT="$ROOT_DIR/scripts/ablation/wait_ecdet_l_dinov2s_patch16_dec3_liver_tmux.sh"
TRAIN_SCRIPT="$ROOT_DIR/scripts/ablation/run_ecdet_l_dinov2s_patch16_dec3_liver_dfine_ablation.sh"

test -x "$TMUX_BIN" || { echo "tmux binary not found: $TMUX_BIN" >&2; exit 1; }
test -x "$WAIT_SCRIPT" || { echo "waiter is not executable: $WAIT_SCRIPT" >&2; exit 1; }
test -x "$TRAIN_SCRIPT" || { echo "training script is not executable: $TRAIN_SCRIPT" >&2; exit 1; }

start_waiter() {
  local experiment="$1" waiter_session="$2" train_session="$3"
  local node="$4" gpus="$5" output="$6"
  if env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" "$TMUX_BIN" has-session -t "$waiter_session" 2>/dev/null; then
    echo "Already running: $waiter_session"
    return 0
  fi
  local command
  command="EC_ABLATION_EXPERIMENT=$experiment TRAIN_SCRIPT=$TRAIN_SCRIPT TRAIN_OUTPUT_DIR=$output TMUX_SESSION=$train_session TARGET_NODES='$node' GPU_INDICES='$gpus' REQUIRED_GPU_COUNT=2 POLL_INTERVAL_SEC=300 SEED=42 WAIT_LOG=$output/wait.log bash '$WAIT_SCRIPT'"
  env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" "$TMUX_BIN" new-session -d -s "$waiter_session" "$command"
  echo "Started $waiter_session -> $train_session on $node GPUs $gpus ($experiment)"
}

OUTPUT_ROOT="$ROOT_DIR/outputs/ablation"
start_waiter ec_full_ignore9 wait_core_ec_full core_ec_full cu01 4,5 \
  "$OUTPUT_ROOT/ecdet_l_dinov2s_patch16_dec3_liver_ec_full_ignore9_bs32_2gpu_seed42"
start_waiter no_cdn_ignore9 wait_core_no_cdn core_no_cdn cu01 6,7 \
  "$OUTPUT_ROOT/ecdet_l_dinov2s_patch16_dec3_liver_no_cdn_ignore9_bs32_2gpu_seed42"
start_waiter no_go_lsd_ignore9 wait_core_no_go_lsd core_no_go_lsd cu02 0,1 \
  "$OUTPUT_ROOT/ecdet_l_dinov2s_patch16_dec3_liver_no_go_lsd_ignore9_bs32_2gpu_seed42"
start_waiter no_dfine_ignore9 wait_core_no_dfine core_no_dfine cu02 2,3 \
  "$OUTPUT_ROOT/ecdet_l_dinov2s_patch16_dec3_liver_no_dfine_ignore9_bs32_2gpu_seed42"
start_waiter continuous_with_go_ignore9 wait_core_continuous_go core_continuous_go cu03 0,1 \
  "$OUTPUT_ROOT/ecdet_l_dinov2s_patch16_dec3_liver_continuous_with_go_ignore9_bs32_2gpu_seed42"

echo "Submitted five waiters: strict-9, decoder=3, two GPUs, global batch=32, seed=42"
