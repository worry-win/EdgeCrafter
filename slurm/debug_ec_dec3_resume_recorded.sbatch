#!/usr/bin/env bash
#SBATCH --job-name=ec-debug-dec3
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:15:00
#SBATCH --output=/opt/wanrui/EdgeCrafter/outputs/slurm/%x-%j.out
#SBATCH --error=/opt/wanrui/EdgeCrafter/outputs/slurm/%x-%j.err

set -euo pipefail

REPO_ROOT=/opt/wanrui/EdgeCrafter
TRAIN_ROOT="${REPO_ROOT}/ecdetseg"
CONFIG_PATH="${TRAIN_ROOT}/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m.yml"
CHECKPOINT_PATH="${TRAIN_ROOT}/outputs/ablation/ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m/last.pth"
OUTPUT_DIR="${TRAIN_ROOT}/outputs/debug/recorded-dec3-${SLURM_JOB_ID}"
TORCHRUN_LOG_DIR="${REPO_ROOT}/outputs/slurm/torchrun-${SLURM_JOB_ID}"

source /home/wanrui/miniconda3/etc/profile.d/conda.sh
conda activate ec

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_DMABUF_ENABLE=0
export NCCL_SOCKET_IFNAME="=bond0"
export NCCL_RAS_ADDR="localhost:$((30000 + SLURM_JOB_ID % 20000))"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=INFO

mkdir -p "${OUTPUT_DIR}" "${TORCHRUN_LOG_DIR}"

cd "${REPO_ROOT}"
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  --log-dir "${TORCHRUN_LOG_DIR}" --redirects 3 --tee 3 \
  scripts/debug/recorded_train.py \
  --config "${CONFIG_PATH}" \
  --resume "${CHECKPOINT_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --seed 42 \
  --use-amp
