#!/usr/bin/env bash
#SBATCH --job-name=ec-kidney-reeval-f1
#SBATCH --partition=debug01
#SBATCH --array=0-3%2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/opt/wanrui/EdgeCrafter/outputs/slurm/%x-%A_%a.out
#SBATCH --error=/opt/wanrui/EdgeCrafter/outputs/slurm/%x-%A_%a.err

set -euo pipefail

REPO_ROOT=/opt/wanrui/EdgeCrafter
TRAIN_ROOT="${REPO_ROOT}/ecdetseg"
RESULT_ROOT="${REPO_ROOT}/outputs/reeval/kidney_extended_f1"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    EXPERIMENT=normal-dec3
    CONFIG_PATH="${TRAIN_ROOT}/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729.yml"
    CHECKPOINT_PATH="${TRAIN_ROOT}/outputs/ablation/ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729/best.pth"
    ;;
  1)
    EXPERIMENT=normal-dec5
    CONFIG_PATH="${TRAIN_ROOT}/configs/ecdet/ecdet_l_dinov2s_patch16_dec5_kidney_v2_260729.yml"
    CHECKPOINT_PATH="${TRAIN_ROOT}/outputs/ablation/ecdet_l_dinov2s_patch16_dec5_kidney_v2_260729/best.pth"
    ;;
  2)
    EXPERIMENT=csgv2-dec3
    CONFIG_PATH="${TRAIN_ROOT}/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m.yml"
    CHECKPOINT_PATH="${TRAIN_ROOT}/outputs/ablation/ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m/best.pth"
    ;;
  3)
    EXPERIMENT=csgv2-dec5
    CONFIG_PATH="${TRAIN_ROOT}/configs/ecdet/ecdet_l_dinov2s_patch16_dec5_kidney_v2_260729_csg200m.yml"
    CHECKPOINT_PATH="${TRAIN_ROOT}/outputs/ablation/ecdet_l_dinov2s_patch16_dec5_kidney_v2_260729_csg200m/best.pth"
    ;;
  *)
    echo "Unsupported array index: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${RESULT_ROOT}/${EXPERIMENT}"
JOB_KEY="${SLURM_ARRAY_JOB_ID:-dry-run}_${SLURM_ARRAY_TASK_ID}"
METRICS_LOG="${OUTPUT_DIR}/metrics-${JOB_KEY}.log"

for required_file in "${CONFIG_PATH}" "${CHECKPOINT_PATH}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing required file: ${required_file}" >&2
    exit 1
  fi
done

echo "Experiment: ${EXPERIMENT}"
echo "Partition: ${SLURM_JOB_PARTITION:-debug01}"
echo "Config: ${CONFIG_PATH}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Metrics log: ${METRICS_LOG}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

source /home/wanrui/miniconda3/etc/profile.d/conda.sh
conda activate ec

export OMP_NUM_THREADS=8
export PYTHONPATH="${TRAIN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

python ecdetseg/train.py \
  --config "${CONFIG_PATH}" \
  --resume "${CHECKPOINT_PATH}" \
  --test-only \
  --output-dir "${OUTPUT_DIR}" \
  --seed 42 \
  2>&1 | tee "${METRICS_LOG}"
