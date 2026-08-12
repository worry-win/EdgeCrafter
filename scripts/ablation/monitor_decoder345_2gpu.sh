#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMUX_BIN="$ROOT_DIR/.local/bin/tmux"
export LD_LIBRARY_PATH="$ROOT_DIR/.local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
LOG_FILE="$ROOT_DIR/outputs/monitoring/decoder345_2gpu.log"

mkdir -p "$(dirname "$LOG_FILE")"

check_experiment() {
  local depth="$1" session="$2" train_log="$3" state progress error
  if "$TMUX_BIN" has-session -t "$session" 2>/dev/null; then
    state="RUNNING"
  elif grep -q "Completed:" "$train_log" 2>/dev/null; then
    state="COMPLETED"
  else
    state="STOPPED"
  fi

  progress="$(grep -E 'Epoch: \[[0-9]+\].*\[[[:space:]]*(0|500|922)/923\]' "$train_log" 2>/dev/null | tail -n 1 || true)"
  error="$(grep -E 'Too many open files|Communication with the workers|ChildFailedError|CUDA out of memory|Traceback' "$train_log" 2>/dev/null | tail -n 1 || true)"
  echo "decoder=$depth state=$state ${progress:-no-progress}"
  [[ -z "$error" ]] || echo "decoder=$depth error=$error"
}

while true; do
  {
    echo "===== $(date '+%F %T') ====="
    check_experiment 3 core_dec3_nofile65536 "$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_baseline_nofile65536_2gpu_seed42/train.log"
    check_experiment 4 core_dec4_nofile65536 "$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_patch16_dec4_liver_baseline_nofile65536_2gpu_seed42/train.log"
    check_experiment 5 core_dec5_nofile65536 "$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_patch16_dec5_liver_baseline_nofile65536_2gpu_seed42/train.log"
  } >> "$LOG_FILE"
  sleep 300
done
