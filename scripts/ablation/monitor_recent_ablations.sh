#!/usr/bin/env bash
set -u

ROOT_DIR="/cobot/Code/wanrui/EdgeCrafter"
LOG_FILE="$ROOT_DIR/outputs/monitoring/recent_ablations_2gpu.log"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-120}"
STALE_AFTER_SEC="${STALE_AFTER_SEC:-900}"
mkdir -p "$(dirname "$LOG_FILE")"

EXPERIMENTS=(
  "cu01|ec_full_neck_pair|$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_ec_full_neck_pair_ignore9_bs32_2gpu_seed42"
  "cu01|rf_neck_p4_points6|$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_rf_neck_p4_points6_ignore9_bs32_2gpu_seed42"
  "cu01|mosaic_focal|$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_mosaic_focal_ignore9_bs32_2gpu_seed42"
  "cu01|dec5_no_dfine|$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_patch16_dec5_liver_no_dfine_ignore9_bs32_2gpu_seed42"
  "cu02|no_dfine_no_cdn|$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_no_dfine_no_cdn_ignore9_bs32_2gpu_seed42"
  "cu02|no_dfine_existing|$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_no_dfine_ignore9_bs32_2gpu_seed42"
  "cu04|dec5_baseline_existing|$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_patch16_dec5_liver_baseline_2gpu_seed42"
)

remote_process_count() {
  local node="$1" output_dir="$2"
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$node" \
    "ps -eo pid,args | awk -v target='$output_dir' 'index(\$0,target) && (\$0 ~ /torch[.]distributed[.]run/ || \$0 ~ /ecdetseg[/]train[.]py/) && \$0 !~ /awk -v target/ {n++} END {print n+0}'" 2>/dev/null || echo 0
}

remote_launcher_count() {
  local node="$1" output_dir="$2"
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$node" \
    "ps -eo pid,args | awk -v target='$output_dir' 'index(\$0,target) && \$0 ~ /torch[.]distributed[.]run/ && \$0 !~ /awk -v target/ {n++} END {print n+0}'" 2>/dev/null || echo 0
}

snapshot() {
  local now node name output_dir train_log process_count launcher_count mtime size age state progress error
  now="$(date +%s)"
  echo "===== $(date '+%F %T') ====="
  for item in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r node name output_dir <<< "$item"
    train_log="$output_dir/train.log"
    process_count="$(remote_process_count "$node" "$output_dir" | tail -n 1)"
    process_count="${process_count//[[:space:]]/}"
    [[ "$process_count" =~ ^[0-9]+$ ]] || process_count=0
    launcher_count="$(remote_launcher_count "$node" "$output_dir" | tail -n 1)"
    launcher_count="${launcher_count//[[:space:]]/}"
    [[ "$launcher_count" =~ ^[0-9]+$ ]] || launcher_count=0
    if [[ -f "$train_log" ]]; then
      mtime="$(stat -c %Y "$train_log" 2>/dev/null || echo 0)"
      size="$(stat -c %s "$train_log" 2>/dev/null || echo 0)"
      age=$((now - mtime))
      progress="$(grep -E 'Epoch: \[[0-9]+\].*\[[[:space:]]*[0-9]+/[0-9]+\]' "$train_log" 2>/dev/null | tail -n 1 | sed -E 's/[[:space:]]+loss:.*$//' || true)"
      error="$(grep -E 'Traceback|RuntimeError|ChildFailedError|CUDA out of memory|Too many open files|Communication with the workers' "$train_log" 2>/dev/null | tail -n 1 || true)"
    else
      size=0; age=-1; progress="no-log"; error=""
    fi
    if grep -q 'Completed:' "$train_log" 2>/dev/null; then
      state="COMPLETED"
    elif [[ -n "$error" && "$age" -gt "$STALE_AFTER_SEC" ]]; then
      state="FAILED"
    elif (( launcher_count > 0 && age >= 0 && age > STALE_AFTER_SEC )); then
      state="SUSPECT_STALLED"
    elif (( launcher_count > 0 )); then
      state="RUNNING"
    elif [[ -n "$error" ]]; then
      state="FAILED"
    else
      state="STOPPED_OR_WAITING"
    fi
    printf '%s node=%s state=%s launcher=%s procs=%s log_age=%ss size=%s progress=%s\n' \
      "$name" "$node" "$state" "$launcher_count" "$process_count" "$age" "$size" "${progress:-no-progress}"
    [[ -z "$error" ]] || printf '%s error=%s\n' "$name" "$error"
  done
  for node in cu01 cu02 cu03 cu04; do
    gpu="$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$node" \
      "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits" 2>/dev/null | paste -sd ';' - || true)"
    echo "gpu node=$node ${gpu:-unavailable}"
  done
}

while true; do
  snapshot >> "$LOG_FILE"
  sleep "$POLL_INTERVAL_SEC"
done
