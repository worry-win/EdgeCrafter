#!/usr/bin/env bash
#SBATCH --job-name=ec-kidney-speed
#SBATCH --partition=debug01
#SBATCH --array=0-1%1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/opt/wanrui/EdgeCrafter/outputs/slurm/%x-%A_%a.out
#SBATCH --error=/opt/wanrui/EdgeCrafter/outputs/slurm/%x-%A_%a.err

set -euo pipefail

REPO_ROOT=/opt/wanrui/EdgeCrafter
TRAIN_ROOT="${REPO_ROOT}/ecdetseg"
RESULT_ROOT="${REPO_ROOT}/outputs/benchmark/kidney_csgv2_model_only"
BENCHMARK_SCRIPT="${TRAIN_ROOT}/tools/benchmark/torch_benchmark.py"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    EXPERIMENT=csgv2-dec3
    CONFIG_PATH="${TRAIN_ROOT}/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m.yml"
    CHECKPOINT_PATH="${TRAIN_ROOT}/outputs/ablation/ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m/best.pth"
    ;;
  1)
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
BENCHMARK_LOG="${OUTPUT_DIR}/benchmark-${JOB_KEY}.log"

for required_file in "${BENCHMARK_SCRIPT}" "${CONFIG_PATH}" "${CHECKPOINT_PATH}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing required file: ${required_file}" >&2
    exit 1
  fi
done

echo "Experiment: ${EXPERIMENT}"
echo "Partition: ${SLURM_JOB_PARTITION:-debug01}"
echo "Config: ${CONFIG_PATH}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Benchmark log: ${BENCHMARK_LOG}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

source /home/wanrui/miniconda3/etc/profile.d/conda.sh
conda activate ec

export PYTHONUNBUFFERED=1

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

python "${BENCHMARK_SCRIPT}" \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --device cuda:0 \
  --batch-size 1 \
  --warmup 100 \
  --iterations 1000 \
  --drop-fastest 50 \
  --drop-slowest 50 \
  2>&1 | tee "${BENCHMARK_LOG}"
