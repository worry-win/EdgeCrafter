#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export TRAIN_SCRIPT="$ROOT_DIR/scripts/ablation/run_ecdet_l_dinov2s_patch16_dec3_liver_no_cdn.sh"
export TMUX_SESSION="${TMUX_SESSION:-ecdet_l_dinov2s_dec3_liver_no_cdn}"
export WAIT_LOG="${WAIT_LOG:-$ROOT_DIR/outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_noreg_no_cdn/wait.log}"

exec bash "$ROOT_DIR/scripts/ablation/wait_ecdet_l_dinov2s_patch16_dec3_liver_tmux.sh"
