#!/usr/bin/env bash
#SBATCH --job-name=kidney_ec_v_preflight
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_geforce_rtx_5090:2
#SBATCH --time=08:00:00
#SBATCH --output=/opt/wanrui/EdgeCrafter/outputs/ablation/kidney_EC_V_v1/slurm_logs/preflight_%j.out
set -euo pipefail

RUN=/opt/wanrui/EdgeCrafter/outputs/ablation/kidney_EC_V_v1
CODE=$RUN/code_snapshot_v1
SHARED=$RUN/shared
PY=/home/wanrui/miniconda3/envs/ec/bin/python
SRUN=/opt/slurm/bin/srun
TRAINER=$CODE/scripts/ablation/train_cmp5L_ec_v.py
MANIFEST=$SHARED/manifest.json
INIT=$SHARED/student_init_seed42.pth
for path in "$PY" "$SRUN" "$TRAINER" "$MANIFEST" "$INIT"; do test -r "$path"; done
test ! -e "$RUN/PREFLIGHT_COMPLETED.json"
export PYTHONPATH="$CODE/ecdetseg:$CODE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 EC_MP_SHARING_STRATEGY=file_descriptor
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 EC_DIST_TIMEOUT_SEC=3600
cd "$CODE"
"$PY" -B -m unittest ecdetseg.tests.test_cmp5l_ec_v ecdetseg.tests.test_kidney_ec_v
"$PY" -B - "$MANIFEST" <<'PY'
import json, pathlib, sys, torch
from engine.core.yaml_utils import load_config
from scripts.ablation.train_cmp5L_ec_v import recipe, validate_v_execution
manifest=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert manifest['classes']==list(range(6)) and len(manifest['batches'])==7
assert set().union(*(set(row['class_ids']) for row in manifest['batches'][:6]))==set(range(6))
assert torch.cuda.is_available() and torch.cuda.device_count()==2
for i in range(9):
 cfg=load_config(f'ecdetseg/configs/ecdet/ecdet_l_dinov2s_kidney_ECV{i}.yml',{})
 validate_v_execution(cfg,world_size=2)
 assert cfg['num_classes']==6 and (cfg['ec_fullcycle_arm'],cfg['ec_y_arm'])==recipe(f'V{i}')
print('PREFLIGHT_CONFIG_OK',len(manifest['batches']),'batches',torch.cuda.device_count(),'gpus')
PY
INIT_SHA=$("$PY" -B -c 'import json,sys;print(json.load(open(sys.argv[1]))["student_init_sha256"])' "$MANIFEST")
for ARM in V0 V4 V7 V8; do
  CONFIG=$CODE/ecdetseg/configs/ecdet/ecdet_l_dinov2s_kidney_EC${ARM}.yml
  OUT=$RUN/preflight/$ARM
  test ! -e "$OUT"
  mkdir -p "$OUT"
  "$SRUN" --cpu-bind=none "$PY" -B -m torch.distributed.run --standalone --nproc_per_node=2 \
    "$CODE/scripts/ablation/run_with_high_nofile.py" "$TRAINER" \
    -c "$CONFIG" -r "$INIT" --v-arm "$ARM" --manifest "$MANIFEST" \
    --init-sha256 "$INIT_SHA" --seed 42 --use-amp --output-dir "$OUT" --smoke
  test -s "$OUT/SMOKE_COMPLETED.json"
done
"$PY" -B - "$RUN/PREFLIGHT_COMPLETED.json" <<'PY'
import json, pathlib, sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({'arms_smoked':['V0','V4','V7','V8'], 'tests':'20 passed', 'classes':6, 'gpus':2},indent=2)+'\n')
PY
