#!/usr/bin/env bash
#SBATCH --job-name=ec-kidney-d3d5
#SBATCH --partition=batch
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=240:00:00
#SBATCH --output=/opt/wanrui/EdgeCrafter/outputs/slurm/%x-%A_%a.out
#SBATCH --error=/opt/wanrui/EdgeCrafter/outputs/slurm/%x-%A_%a.err

set -euo pipefail

REPO_ROOT=/opt/wanrui/EdgeCrafter
TRAIN_ROOT="${REPO_ROOT}/ecdetseg"
CHECKPOINT_PATH="${REPO_ROOT}/artifacts/checkpoints/dinov2/dinov2_vits14_pretrain.pth"
TRAIN_JSON="/opt/public/wangzhiwei/Ultrasound_Data/COCO_Output/肾脏/annotations/v2_260729/train.json"
VAL_JSON="/opt/public/wangzhiwei/Ultrasound_Data/COCO_Output/肾脏/annotations/v2_260729/val.json"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    DECODER_DEPTH=3
    CONFIG_PATH="${TRAIN_ROOT}/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729.yml"
    OUTPUT_DIR="${TRAIN_ROOT}/outputs/ablation/ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729"
    ;;
  1)
    DECODER_DEPTH=5
    CONFIG_PATH="${TRAIN_ROOT}/configs/ecdet/ecdet_l_dinov2s_patch16_dec5_kidney_v2_260729.yml"
    OUTPUT_DIR="${TRAIN_ROOT}/outputs/ablation/ecdet_l_dinov2s_patch16_dec5_kidney_v2_260729"
    ;;
  *)
    echo "Unsupported array index: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

for required_file in "${CONFIG_PATH}" "${CHECKPOINT_PATH}" "${TRAIN_JSON}" "${VAL_JSON}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing required file: ${required_file}" >&2
    exit 1
  fi
done

source /home/wanrui/miniconda3/etc/profile.d/conda.sh
conda activate ec

# Required by the verified 4x RTX 5090 NCCL path on this cluster.
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export NCCL_DMABUF_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO

mkdir -p "${OUTPUT_DIR}"

resume_args=()
if [[ -f "${OUTPUT_DIR}/last.pth" ]]; then
  resume_args=(--resume "${OUTPUT_DIR}/last.pth")
fi

echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "Decoder layers: ${DECODER_DEPTH}"
echo "Config: ${CONFIG_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Environment: ${CONDA_DEFAULT_ENV}"
python -c 'import torch; print(f"PyTorch/CUDA/NCCL: {torch.__version__}/{torch.version.cuda}/{torch.cuda.nccl.version()}")'
echo "DINOv2 checkpoint: ${CHECKPOINT_PATH}"
echo "Global batch size: 32 (4 GPUs x 8 samples/GPU)"
echo "Epochs/patience/seed: 150/30/42"

cd "${TRAIN_ROOT}"
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  train.py \
  --config "${CONFIG_PATH}" \
  --seed 42 \
  --use-amp \
  "${resume_args[@]}"
