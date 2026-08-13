#!/usr/bin/env bash
#SBATCH --job-name=ec-csg-kidney-3exp
#SBATCH --partition=batch
#SBATCH --array=0-2
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
CHECKPOINT_ROOT=/opt/public/wangzhiwei/Detect_Model/EC-1+2+decoder
IMAGE_ROOT="/opt/public/wangzhiwei/Ultrasound_Data/COCO_Output/肾脏/images_CSG_200m"
TRAIN_JSON="/opt/public/wangzhiwei/Ultrasound_Data/COCO_Output/肾脏/annotations/v2_260729/train.json"
VAL_JSON="/opt/public/wangzhiwei/Ultrasound_Data/COCO_Output/肾脏/annotations/v2_260729/val.json"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    EXPERIMENT_NAME=dinov2s-dec4
    CONFIG_PATH="${TRAIN_ROOT}/configs/ecdet/ecdet_l_dinov2s_patch16_dec4_kidney_v2_260729_csg200m.yml"
    OUTPUT_DIR="${TRAIN_ROOT}/outputs/ablation/ecdet_l_dinov2s_patch16_dec4_kidney_v2_260729_csg200m"
    TUNING_CHECKPOINT=""
    FEATAUG_REQUIRED=0
    ;;
  1)
    EXPERIMENT_NAME=ecdetx-coco-decoder
    CONFIG_PATH="${TRAIN_ROOT}/configs/ecdet/ecdet_x_kidney_v2_260729_csg200m_coco_decoder_150e.yml"
    OUTPUT_DIR="${TRAIN_ROOT}/outputs/ablation/ecdet_x_kidney_v2_260729_csg200m_coco_decoder_150e"
    TUNING_CHECKPOINT="${CHECKPOINT_ROOT}/EC-1+2+coco-decoder.pth"
    FEATAUG_REQUIRED=0
    ;;
  2)
    EXPERIMENT_NAME=ecdetx-organ-decoder-feataug
    CONFIG_PATH="${TRAIN_ROOT}/configs/ecdet/ecdet_x_kidney_v2_260729_csg200m_organ_decoder_150e.yml"
    OUTPUT_DIR="${TRAIN_ROOT}/outputs/ablation/ecdet_x_kidney_v2_260729_csg200m_organ_decoder_150e"
    TUNING_CHECKPOINT="${CHECKPOINT_ROOT}/EC-1+2+organ-decoder.pth"
    FEATAUG_REQUIRED=1
    ;;
  *)
    echo "Unsupported array index: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

for required_path in "${CONFIG_PATH}" "${IMAGE_ROOT}" "${TRAIN_JSON}" "${VAL_JSON}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required path: ${required_path}" >&2
    exit 1
  fi
done
if [[ -n "${TUNING_CHECKPOINT}" && ! -f "${TUNING_CHECKPOINT}" ]]; then
  echo "Missing tuning checkpoint: ${TUNING_CHECKPOINT}" >&2
  exit 1
fi

echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Config: ${CONFIG_PATH}"
echo "Images: ${IMAGE_ROOT}"
echo "Output: ${OUTPUT_DIR}"
echo "FeatAug required: ${FEATAUG_REQUIRED}"
if [[ -n "${TUNING_CHECKPOINT}" ]]; then
  echo "Tuning checkpoint: ${TUNING_CHECKPOINT}"
else
  echo "Tuning checkpoint: none"
fi
echo "Global batch size: 32 (4 GPUs x 8 samples/GPU)"
echo "Epochs/patience/seed: 150/30/42"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

source /home/wanrui/miniconda3/etc/profile.d/conda.sh
conda activate ec

if [[ -n "${TUNING_CHECKPOINT}" ]]; then
  preflight_args=(
    --config "${CONFIG_PATH}"
    --checkpoint "${TUNING_CHECKPOINT}"
  )
  if [[ "${FEATAUG_REQUIRED}" == "1" ]]; then
    preflight_args+=(--require-feataug)
  fi
  python "${REPO_ROOT}/scripts/ablation/check_ecdet_tuning.py" "${preflight_args[@]}"
fi

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  exit 0
fi

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

checkpoint_args=()
if [[ -f "${OUTPUT_DIR}/last.pth" ]]; then
  checkpoint_args=(--resume "${OUTPUT_DIR}/last.pth")
elif [[ -n "${TUNING_CHECKPOINT}" ]]; then
  checkpoint_args=(--tuning "${TUNING_CHECKPOINT}")
fi

cd "${TRAIN_ROOT}"
python -c 'import torch; print(f"PyTorch/CUDA/NCCL: {torch.__version__}/{torch.version.cuda}/{torch.cuda.nccl.version()}")'
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  train.py \
  --config "${CONFIG_PATH}" \
  --seed 42 \
  --use-amp \
  "${checkpoint_args[@]}"
