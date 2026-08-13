#!/usr/bin/env bash
#SBATCH --job-name=ec-2gpu-collective
#SBATCH --partition=batch
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/opt/wanrui/EdgeCrafter/outputs/slurm/%x-%A_%a.out
#SBATCH --error=/opt/wanrui/EdgeCrafter/outputs/slurm/%x-%A_%a.err

set -euo pipefail

source /home/wanrui/miniconda3/etc/profile.d/conda.sh
conda activate ec

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export NCCL_DMABUF_ENABLE=0
export NCCL_SOCKET_IFNAME="=bond0"
export NCCL_RAS_ADDR="localhost:$((30000 + SLURM_JOB_ID % 20000))"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    export NCCL_SHM_DISABLE=1
    MODE=socket
    ;;
  1)
    export NCCL_SHM_DISABLE=0
    export NCCL_CUMEM_HOST_ENABLE=0
    MODE=shm
    ;;
  *)
    exit 2
    ;;
esac

echo "Collective mode: ${MODE}"
echo "Node: ${SLURM_JOB_NODELIST}"
echo "NCCL RAS address: ${NCCL_RAS_ADDR}"

cd /opt/wanrui/EdgeCrafter
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/verify_ec_2gpu_collectives.py
