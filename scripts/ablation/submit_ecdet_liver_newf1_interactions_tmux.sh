#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMUX_BIN="${TMUX_BIN:-$ROOT_DIR/.local/bin/tmux}"
export LD_LIBRARY_PATH="$ROOT_DIR/.local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
WAITER="$ROOT_DIR/scripts/ablation/wait_ecdet_l_dinov2s_patch16_dec3_liver_tmux.sh"
TRAINER="$ROOT_DIR/scripts/ablation/run_ecdet_liver_newf1_interactions.sh"
OUTPUT_ROOT="$ROOT_DIR/outputs/ablation"

test -x "$TMUX_BIN"
test -x "$WAITER"
test -x "$TRAINER"

start_waiter() {
  local experiment="$1" waiter_session="$2" train_session="$3"
  local node="$4" gpus="$5" output="$6"
  if env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" "$TMUX_BIN" has-session -t "$waiter_session" 2>/dev/null; then
    echo "Already running: $waiter_session"
    return 0
  fi
  local command
  command="EC_ABLATION_EXPERIMENT=$experiment TRAIN_SCRIPT='$TRAINER' TRAIN_OUTPUT_DIR='$output' TMUX_SESSION='$train_session' TARGET_NODES='$node' GPU_INDICES='$gpus' REQUIRED_GPU_COUNT=2 MIN_FREE_MEM_MB=28000 POLL_INTERVAL_SEC=120 SEED=42 WAIT_LOG='$output/wait.log' bash '$WAITER'"
  env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" "$TMUX_BIN" new-session -d -s "$waiter_session" "$command"
  echo "Started $waiter_session -> $train_session on $node GPUs $gpus ($experiment)"
}

# Two idle pairs launch immediately; four occupied pairs remain isolated waiters.
start_waiter no_mosaic_focal wait_nf1_nm_focal nf1_nm_focal cu04 4,5 \
  "$OUTPUT_ROOT/newf1_ec_no_mosaic_focal_bs32_2gpu_seed42"
start_waiter no_mosaic_focal_no_dfine wait_nf1_nm_focal_ndf nf1_nm_focal_ndf cu04 6,7 \
  "$OUTPUT_ROOT/newf1_ec_no_mosaic_focal_no_dfine_bs32_2gpu_seed42"
start_waiter no_mosaic_focal_no_dfine_no_cdn wait_nf1_nm_focal_ndf_ncdn nf1_nm_focal_ndf_ncdn cu01 0,1 \
  "$OUTPUT_ROOT/newf1_ec_no_mosaic_focal_no_dfine_no_cdn_bs32_2gpu_seed42"
start_waiter rf_neck_p4_points6 wait_nf1_rf6 nf1_rf6 cu01 2,3 \
  "$OUTPUT_ROOT/newf1_ec_rf_neck_p4_points6_bs32_2gpu_seed42"
start_waiter rf_neck_p4_points6_no_dfine wait_nf1_rf6_ndf nf1_rf6_ndf cu03 0,1 \
  "$OUTPUT_ROOT/newf1_ec_rf_neck_p4_points6_no_dfine_bs32_2gpu_seed42"
start_waiter rf_neck_p4_points6_no_dfine_no_cdn wait_nf1_rf6_ndf_ncdn nf1_rf6_ndf_ncdn cu03 2,3 \
  "$OUTPUT_ROOT/newf1_ec_rf_neck_p4_points6_no_dfine_no_cdn_bs32_2gpu_seed42"

echo "Six aligned-F1 interaction jobs queued: 2 GPUs, global batch 32, seed 42"
