#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMUX_BIN="${TMUX_BIN:-$ROOT_DIR/.local/bin/tmux}"
TMUX_LIB_DIR="${TMUX_LIB_DIR:-$ROOT_DIR/.local/lib}"
export LD_LIBRARY_PATH="$TMUX_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$ROOT_DIR/scripts/ablation/run_ecdet_l_dinov2s_patch16_dec3_liver.sh}"
SESSION_NAME="${TMUX_SESSION:-ecdet_l_dinov2s_dec3_liver}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-300}"
REQUIRED_GPU_COUNT="${REQUIRED_GPU_COUNT:-4}"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-28000}"
TARGET_NODES="${TARGET_NODES:-cu01 cu02 cu03 cu04}"
GPU_INDICES="${GPU_INDICES:-}"
SSH_CONNECT_TIMEOUT_SEC="${SSH_CONNECT_TIMEOUT_SEC:-5}"
LOG_FILE="${WAIT_LOG:-$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_noreg/wait.log}"

mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

test -x "$TMUX_BIN" || { echo "tmux binary not found: $TMUX_BIN"; exit 1; }
command -v ssh >/dev/null || { echo "ssh not found"; exit 1; }
test -x "$TRAIN_SCRIPT" || { echo "Training script is not executable: $TRAIN_SCRIPT"; exit 1; }

current_node="$(hostname -s)"

run_on_node() {
  local node="$1"
  shift
  if [[ "$node" == "$current_node" ]]; then
    "$@"
  else
    ssh -o BatchMode=yes -o ConnectTimeout="$SSH_CONNECT_TIMEOUT_SEC" "$node" "$@"
  fi
}

free_gpu_indices() {
  local node="$1"
  run_on_node "$node" nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null |
    awk -F',' -v minimum="$MIN_FREE_MEM_MB" '{gsub(/ /, "", $1); gsub(/ /, "", $2); if (($2 + 0) >= minimum) print $1}'
}

launch_training() {
  local node="$1"
  local gpu_csv="$2"
  local train_command="EC_ABLATION_EXPERIMENT=$EC_ABLATION_EXPERIMENT CUDA_VISIBLE_DEVICES=$gpu_csv NPROC_PER_NODE=$REQUIRED_GPU_COUNT bash '$TRAIN_SCRIPT'"

  if [[ "$node" == "$current_node" ]]; then
    env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" "$TMUX_BIN" new-session -d -s "$SESSION_NAME" "$train_command"
  else
    ssh -o BatchMode=yes -o ConnectTimeout="$SSH_CONNECT_TIMEOUT_SEC" "$node" \
      "LD_LIBRARY_PATH='$LD_LIBRARY_PATH' '$TMUX_BIN' new-session -d -s '$SESSION_NAME' \"$train_command\""
  fi
}

read -r -a nodes <<< "$TARGET_NODES"
echo "Polling nodes [${nodes[*]}] every ${POLL_INTERVAL_SEC}s for ${REQUIRED_GPU_COUNT} GPUs with >=${MIN_FREE_MEM_MB} MiB free"
while true; do
  selected_node=""
  selected_candidates=()
  for node in "${nodes[@]}"; do
    if ! run_on_node "$node" test -x "$TMUX_BIN" >/dev/null 2>&1; then
      echo "[$(date '+%F %T')] $node skipped: tmux binary unavailable"
      continue
    fi
    if run_on_node "$node" env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" "$TMUX_BIN" has-session -t "$SESSION_NAME" >/dev/null 2>&1; then
      echo "[$(date '+%F %T')] $node skipped: session $SESSION_NAME already exists"
      continue
    fi
    mapfile -t candidates < <(free_gpu_indices "$node" | while read -r idx; do
      if [[ -z "$GPU_INDICES" || ",${GPU_INDICES}," == *",${idx},"* ]]; then echo "$idx"; fi
    done)
    echo "[$(date '+%F %T')] $node free candidates: ${candidates[*]:-unavailable/none}"
    if (( ${#candidates[@]} >= REQUIRED_GPU_COUNT && ${#candidates[@]} > ${#selected_candidates[@]} )); then
      selected_node="$node"
      selected_candidates=("${candidates[@]}")
    fi
  done

  if [[ -n "$selected_node" ]]; then
    selected=("${selected_candidates[@]:0:REQUIRED_GPU_COUNT}")
    gpu_csv="$(IFS=,; echo "${selected[*]}")"
    echo "Launching tmux session $SESSION_NAME on $selected_node GPUs $gpu_csv"
    launch_training "$selected_node" "$gpu_csv"
    echo "Attach command: ssh -t $selected_node tmux attach -t $SESSION_NAME"
    exit 0
  fi
  sleep "$POLL_INTERVAL_SEC"
done
