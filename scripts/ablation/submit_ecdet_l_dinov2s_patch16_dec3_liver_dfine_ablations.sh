#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMUX_BIN="${TMUX_BIN:-$ROOT_DIR/.local/bin/tmux}"
export LD_LIBRARY_PATH="$ROOT_DIR/.local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
WAIT_SCRIPT="$ROOT_DIR/scripts/ablation/wait_ecdet_l_dinov2s_patch16_dec3_liver_tmux.sh"
ONLY_EXPERIMENT="${ONLY_EXPERIMENT:-}"

test -x "$TMUX_BIN" || { echo "tmux binary not found: $TMUX_BIN" >&2; exit 1; }
test -x "$WAIT_SCRIPT" || { echo "waiter is not executable: $WAIT_SCRIPT" >&2; exit 1; }

start_waiter() {
  local experiment="$1" session="$2" node="$3" gpus="$4"
  if env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" "$TMUX_BIN" has-session -t "$session" 2>/dev/null; then
    echo "Already running: $session"
    return 0
  fi
  local command
  command="EC_ABLATION_EXPERIMENT=$experiment TRAIN_SCRIPT=$ROOT_DIR/scripts/ablation/run_ecdet_l_dinov2s_patch16_dec3_liver_dfine_ablation.sh TMUX_SESSION=ecdet_l_dinov2s_${experiment} TARGET_NODES='$node' GPU_INDICES='$gpus' WAIT_LOG=$ROOT_DIR/outputs/ablation/${experiment}/wait.log POLL_INTERVAL_SEC=300 bash '$WAIT_SCRIPT'"
  env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" "$TMUX_BIN" new-session -d -s "$session" "$command"
  echo "Started waiter $session -> $experiment, target $node GPUs $gpus"
}

# cu03 is empty; split it between the first two experiments. cu04:4-7 is
# free while the existing full baseline occupies cu04:0-3.
if [[ -z "$ONLY_EXPERIMENT" || "$ONLY_EXPERIMENT" == no_fdr_decode ]]; then
  start_waiter no_fdr_decode wait_ec_no_fdr_decode cu03 0,1,2,3
fi
if [[ -z "$ONLY_EXPERIMENT" || "$ONLY_EXPERIMENT" == no_go_ddf ]]; then
  start_waiter no_go_ddf wait_ec_no_go_ddf cu03 4,5,6,7
fi
if [[ -z "$ONLY_EXPERIMENT" || "$ONLY_EXPERIMENT" == no_fdr_no_go_ddf ]]; then
  start_waiter no_fdr_no_go_ddf wait_ec_no_fdr_no_go_ddf cu04 4,5,6,7
fi
if [[ -z "$ONLY_EXPERIMENT" || "$ONLY_EXPERIMENT" == no_mosaic ]]; then
  start_waiter no_mosaic wait_ec_no_mosaic cu02 4,5,6,7
fi
if [[ -z "$ONLY_EXPERIMENT" || "$ONLY_EXPERIMENT" == rf_neck ]]; then
  start_waiter rf_neck wait_ec_rf_neck "cu01 cu02 cu03 cu04" ""
fi
if [[ -z "$ONLY_EXPERIMENT" || "$ONLY_EXPERIMENT" == rf_neck_p4_ignore9 ]]; then
  start_waiter rf_neck_p4_ignore9 wait_ec_rf_neck_p4_ignore9 cu02 0,1,2,3
fi

echo "Waiters poll every 300 seconds. Use: $TMUX_BIN ls"
