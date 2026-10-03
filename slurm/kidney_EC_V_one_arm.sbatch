#!/usr/bin/env bash
#SBATCH --job-name=kidney_ec_v
#SBATCH --partition=batch
#SBATCH --array=0-8%3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_geforce_rtx_5090:2
#SBATCH --time=3-00:00:00
#SBATCH --output=/opt/wanrui/EdgeCrafter/outputs/ablation/kidney_EC_V_v1/slurm_logs/V%A_%a.out
set -euo pipefail

ARM=V${SLURM_ARRAY_TASK_ID:?}
case "$ARM" in V[0-8]) ;; *) echo "invalid arm: $ARM" >&2; exit 2;; esac
RUN=/opt/wanrui/EdgeCrafter/outputs/ablation/kidney_EC_V_v1
CODE=$RUN/code_snapshot_v1
SHARED=$RUN/shared
OUT=$RUN/$ARM
PY=/home/wanrui/miniconda3/envs/ec/bin/python
SRUN=/opt/slurm/bin/srun
CONFIG=$CODE/ecdetseg/configs/ecdet/ecdet_l_dinov2s_kidney_EC${ARM}.yml
INIT=$SHARED/student_init_seed42.pth
MANIFEST=$SHARED/manifest.json
TRAINER=$CODE/scripts/ablation/train_cmp5L_ec_v.py
EVALUATOR=$CODE/scripts/ablation/evaluate_cmp5L_qbeh_checkpoint.py
SUMMARY=$CODE/scripts/ablation/summarize_cmp5L_ec_v.py
IMAGE_ROOT=/opt/public/wangzhiwei/Ultrasound_Data/COCO_Output/肾脏/images_CSG_200m
VALID_ANN=/opt/public/wangzhiwei/Ultrasound_Data/COCO_Output/肾脏/annotations/Lesion_Det/v2_260729/val.json
TRAIN_ANN=/opt/public/wangzhiwei/Ultrasound_Data/COCO_Output/肾脏/annotations/Lesion_Det/v2_260729/train.json
for path in "$RUN/PREFLIGHT_COMPLETED.json" "$PY" "$SRUN" "$CONFIG" "$INIT" "$MANIFEST" "$TRAINER" "$EVALUATOR" "$SUMMARY" "$IMAGE_ROOT" "$VALID_ANN" "$TRAIN_ANN"; do
  test -r "$path" || { echo "missing or unreadable: $path" >&2; exit 4; }
done
mkdir -p "$OUT"
test ! -e "$OUT/COMPLETED.json"
export PYTHONPATH="$CODE/ecdetseg:$CODE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 EC_MP_SHARING_STRATEGY=file_descriptor
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 EC_DIST_TIMEOUT_SEC=3600
export EC_SNAPSHOT_GIT_HEAD=afe652f495dba2f3978508c705bc8d574d1a7f8e
cd "$CODE"
INIT_SHA=$("$PY" -B -c 'import json,sys;print(json.load(open(sys.argv[1]))["student_init_sha256"])' "$MANIFEST")
COMMON=(-c "$CONFIG" --v-arm "$ARM" --manifest "$MANIFEST" --init-sha256 "$INIT_SHA" --seed 42 --use-amp)
if [[ ! -f "$OUT/smoke/SMOKE_COMPLETED.json" ]]; then
  if [[ -d "$OUT/smoke" ]]; then echo "partial smoke preserved; inspect before rerun" >&2; exit 5; fi
  mkdir -p "$OUT/smoke"
  "$SRUN" --cpu-bind=none "$PY" -B -m torch.distributed.run --standalone --nproc_per_node=2 \
    "$CODE/scripts/ablation/run_with_high_nofile.py" "$TRAINER" \
    "${COMMON[@]}" -r "$INIT" --output-dir "$OUT/smoke" --smoke \
    2>&1 | tee "$OUT/smoke/stdout.log"
fi
RESUME=$INIT
if [[ -f "$OUT/last.pth" ]]; then RESUME=$OUT/last.pth; fi
"$SRUN" --cpu-bind=none "$PY" -B -m torch.distributed.run --standalone --nproc_per_node=2 \
  "$CODE/scripts/ablation/run_with_high_nofile.py" "$TRAINER" \
  "${COMMON[@]}" -r "$RESUME" --output-dir "$OUT" \
  2>&1 | tee -a "$OUT/train_stdout.log"
for checkpoint in checkpoint0049.pth checkpoint0059.pth checkpoint0079.pth last.pth best.pth; do
  test -s "$OUT/$checkpoint"
done
for label in best final; do
  if [[ "$label" == best ]]; then CKPT=$OUT/best.pth; else CKPT=$OUT/last.pth; fi
  EVAL_OUT=$OUT/validation_$label
  if [[ ! -f "$EVAL_OUT/metrics.json" ]]; then
    mkdir -p "$EVAL_OUT"
    "$SRUN" --cpu-bind=none "$PY" -B "$EVALUATOR" \
      --config "$CONFIG" --checkpoint "$CKPT" --weights ema \
      --img-folder "$IMAGE_ROOT" --ann-file "$VALID_ANN" \
      --num-classes 6 --batch-size 8 --num-workers 4 --expected-images 1198 \
      --out "$EVAL_OUT/metrics.json" 2>&1 | tee "$EVAL_OUT/stdout.log"
  fi
done
"$PY" -B "$SUMMARY" --arm "$( [[ "$ARM" == V0 ]] && echo ECX0 || echo ECX3 )" \
  --v-arm "$ARM" --out-dir "$OUT" --config "$CONFIG" --manifest "$MANIFEST" \
  --init-sha256 "$INIT_SHA" --ann-file "$VALID_ANN" --expected-images 1198
