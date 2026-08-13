#!/usr/bin/env bash
#SBATCH --job-name=ec-nccl-smoke
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/opt/wanrui/EdgeCrafter/outputs/slurm/%x-%j.out
#SBATCH --error=/opt/wanrui/EdgeCrafter/outputs/slurm/%x-%j.err

set -euo pipefail

source /home/wanrui/miniconda3/etc/profile.d/conda.sh
conda activate ec

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export NCCL_DMABUF_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO

python /opt/wanrui/eomt_seg/scripts/verify_blackwell_cuda.py
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  /opt/wanrui/eomt_seg/scripts/verify_blackwell_ddp.py
