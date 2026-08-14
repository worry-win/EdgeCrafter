#!/usr/bin/env bash
#SBATCH --job-name=ec-csgv2-ablate
#SBATCH --partition=batch
#SBATCH --array=0-7%4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=240:00:00
#SBATCH --output=/opt/wanrui/EdgeCrafter/outputs/slurm/%x-%A_%a.out
#SBATCH --error=/opt/wanrui/EdgeCrafter/outputs/slurm/%x-%A_%a.err

set -euo pipefail

REPO_ROOT=/opt/wanrui/EdgeCrafter
TRAIN_ROOT="${REPO_ROOT}/ecdetseg"
CONFIG_ROOT="${TRAIN_ROOT}/configs/ecdet"
CHECKPOINT_PATH="${REPO_ROOT}/artifacts/checkpoints/dinov2/dinov2_vits14_pretrain.pth"
IMAGE_ROOT="/opt/public/wangzhiwei/Ultrasound_Data/COCO_Output/肾脏/images_CSG_200m"
TRAIN_JSON="/opt/public/wangzhiwei/Ultrasound_Data/COCO_Output/肾脏/annotations/v2_260729/train.json"
VAL_JSON="/opt/public/wangzhiwei/Ultrasound_Data/COCO_Output/肾脏/annotations/v2_260729/val.json"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    EXPERIMENT=no-cdn
    CONFIG_NAME=ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m_no_cdn.yml
    OUTPUT_NAME=ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m_no_cdn
    ;;
  1)
    EXPERIMENT=no-go-lsd
    CONFIG_NAME=ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m_no_go_lsd.yml
    OUTPUT_NAME=ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m_no_go_lsd
    ;;
  2)
    EXPERIMENT=no-dfine
    CONFIG_NAME=ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m_no_dfine.yml
    OUTPUT_NAME=ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m_no_dfine
    ;;
  3)
    EXPERIMENT=continuous-go
    CONFIG_NAME=ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m_continuous_go.yml
    OUTPUT_NAME=ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m_continuous_go
    ;;
  4)
    EXPERIMENT=mosaic-focal
    CONFIG_NAME=ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m_mosaic_focal.yml
    OUTPUT_NAME=ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m_mosaic_focal
    ;;
  5)
    EXPERIMENT=no-mosaic-focal
    CONFIG_NAME=ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m_no_mosaic_focal.yml
    OUTPUT_NAME=ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m_no_mosaic_focal
    ;;
  6)
    EXPERIMENT=no-dfine-no-cdn
    CONFIG_NAME=ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m_no_dfine_no_cdn.yml
    OUTPUT_NAME=ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m_no_dfine_no_cdn
    ;;
  7)
    EXPERIMENT=dec5-no-dfine
    CONFIG_NAME=ecdet_l_dinov2s_patch16_dec5_kidney_v2_260729_csg200m_no_dfine.yml
    OUTPUT_NAME=ecdet_l_dinov2s_patch16_dec5_kidney_v2_260729_csg200m_no_dfine
    ;;
  *)
    echo "Unsupported array index: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

CONFIG_PATH="${CONFIG_ROOT}/${CONFIG_NAME}"
OUTPUT_DIR="${TRAIN_ROOT}/outputs/ablation/${OUTPUT_NAME}"

echo "Experiment: ${EXPERIMENT}"
echo "Config: ${CONFIG_PATH}"
echo "Images: ${IMAGE_ROOT}"
echo "Output: ${OUTPUT_DIR}"
echo "Global batch size: 32 (2 GPUs x 16 samples/GPU)"
echo "Epochs/patience/seed: 150/30/42"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

for required_path in "${CONFIG_PATH}" "${CHECKPOINT_PATH}" "${IMAGE_ROOT}" "${TRAIN_JSON}" "${VAL_JSON}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required path: ${required_path}" >&2
    exit 1
  fi
done

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
export NCCL_DEBUG=INFO

mkdir -p "${OUTPUT_DIR}"

resume_args=()
if [[ -f "${OUTPUT_DIR}/last.pth" ]]; then
  resume_args=(--resume "${OUTPUT_DIR}/last.pth")
fi

echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "Environment: ${CONDA_DEFAULT_ENV}"
python -c 'import torch; print(f"PyTorch/CUDA/NCCL: {torch.__version__}/{torch.version.cuda}/{torch.cuda.nccl.version()}")'
echo "DINOv2 checkpoint: ${CHECKPOINT_PATH}"

cd "${TRAIN_ROOT}"
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  train.py \
  --config "${CONFIG_PATH}" \
  --seed 42 \
  --use-amp \
  "${resume_args[@]}"
