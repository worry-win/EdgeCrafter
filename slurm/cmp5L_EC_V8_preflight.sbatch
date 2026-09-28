#!/usr/bin/env bash
# Independent two-GPU EC-V arm; smoke, 100 epochs, best/final EMA validation.
#SBATCH --partition=debug,batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --gres=gpu:nvidia_geforce_rtx_5090:2
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
set -euo pipefail

ARM="V8"
case "$ARM" in V8) ;; *) echo "invalid V arm" >&2; exit 2;; esac
ROOT=/cobot/Code/wanrui/EdgeCrafter
RUN=$ROOT/outputs/ablation/cmp5L_EC_V8_ref_v1
CODE=$RUN/code_snapshot_v1
SHARED=$ROOT/outputs/ablation/cmp5L_EC_fullcycle_v1/shared
OUT=$RUN/$ARM
PY=/cobot/miniforge3/envs/lw-detr/bin/python
SRUN=/opt/slurm/bin/srun
CONFIG=$CODE/ecdetseg/configs/ecdet/ecdet_l_dinov2s_cmp5L_EC${ARM}.yml
INIT=$SHARED/student_init_seed42.pth
MANIFEST=$SHARED/manifest.json
TRAINER=$CODE/scripts/ablation/train_cmp5L_ec_v.py
EVALUATOR=$CODE/scripts/ablation/evaluate_cmp5L_qbeh_checkpoint.py
SUMMARY=$CODE/scripts/ablation/summarize_cmp5L_ec_v.py
IMAGE_ROOT=/cobot/Data/Lesion_det/det_breast/img
VALID_ANN=/cobot/Data/Lesion_det/det_breast/annotations/Lesion/Ignore_Delete-Image/valid.json
mkdir -p "$OUT"
trap 'status=$?; "$PY" - "$OUT/FAILED.json" "$ARM" "$status" "${SLURM_JOB_ID:-unknown}" <<'"'"'PY'"'"'
import json,sys,time
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"v_arm":sys.argv[2],"exit_code":int(sys.argv[3]),"job_id":sys.argv[4],"time":time.time()},indent=2)+"\n")
PY
exit "$status"' ERR
trap 'echo "USR1: completed-epoch checkpoint remains recoverable" >&2' USR1
test ! -e "$OUT/COMPLETED.json"
test ! -e "$OUT/FAILED.json"
for item in "$PY" "$SRUN" "$CONFIG" "$INIT" "$MANIFEST" "$TRAINER" "$EVALUATOR" "$SUMMARY" "$IMAGE_ROOT" "$VALID_ANN"; do
  test -r "$item" || { echo "missing $item" >&2; exit 4; }
done
export PYTHONPATH="$CODE/ecdetseg:$CODE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 EC_MP_SHARING_STRATEGY=file_descriptor
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 EC_DIST_TIMEOUT_SEC=3600
cd "$CODE"
"$PY" -m unittest ecdetseg.tests.test_cmp5l_ec_v
"$PY" - "$CONFIG" "$MANIFEST" "$ARM" <<'PY'
import hashlib,json,sys,torch
from pathlib import Path
from engine.core.yaml_utils import load_config
from scripts.ablation.train_cmp5L_ec_v import recipe,validate_v_execution
config,manifest,arm=sys.argv[1:]
c=load_config(config,{})
validate_v_execution(c,world_size=2)
assert c['ec_v_arm']==arm and (c['ec_fullcycle_arm'],c['ec_y_arm'])==recipe(arm)
m=json.loads(Path(manifest).read_text())
assert len(m['batches'])==7 and len({j for x in m['batches'][:6] for j in x['class_ids']})==4
assert torch.cuda.is_available() and torch.cuda.device_count()==2
print(json.dumps({'arm':arm,'seed':42,'gpus':2,'local_batch':16,'effective_batch':32,
                  'accumulation':1,'sync_bn':True,'manifest_sha256':hashlib.sha256(Path(manifest).read_bytes()).hexdigest()}))
PY
INIT_SHA=$("$PY" - "$MANIFEST" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['student_init_sha256'])
PY
)
COMMON=(-c "$CONFIG" --v-arm "$ARM" --manifest "$MANIFEST" --init-sha256 "$INIT_SHA" --seed 42 --use-amp)
if [[ ! -f "$OUT/smoke/SMOKE_COMPLETED.json" ]]; then
  if [[ -d "$OUT/smoke" ]]; then echo "partial smoke preserved; refusing implicit rerun" >&2; exit 5; fi
  mkdir -p "$OUT/smoke"
  "$SRUN" --cpu-bind=none "$PY" -m torch.distributed.run --standalone --nproc_per_node=2 \
    "$CODE/scripts/ablation/run_with_high_nofile.py" "$TRAINER" \
    "${COMMON[@]}" -r "$INIT" --output-dir "$OUT/smoke" --smoke \
    2>&1 | tee "$OUT/smoke/stdout.log"
fi
"$PY" - "$OUT/smoke/SMOKE_COMPLETED.json" "$OUT/smoke/v8_fixed_reference_diagnostics.jsonl" <<'PY'
import json,sys
from pathlib import Path
smoke=json.loads(Path(sys.argv[1]).read_text())
assert smoke['updates'] >= 5 and smoke['teacher_gradient_count'] == 0
rows=[json.loads(x) for x in Path(sys.argv[2]).read_text().splitlines()]
values=[d for row in rows for batch in row['batches'] for d in [batch['reference_diagnostics']]
        if d and d['new_kl']]
assert values and all(batch['teacher_topk_calls'] == 0 for row in rows for batch in row['batches'])
ref=[p for d in values for p in d['student_ref_probability']]
assert min(ref)>1e-8 and max(ref)<1-1e-8
assert max(abs(r) for d in values for r in d['chain_residual'])<1e-4
print(json.dumps({'smoke_updates':smoke['updates'],'teacher_gradient_count':0,
                  'reference_probability_min':min(ref),'reference_probability_max':max(ref),
                  'max_chain_residual':max(abs(r) for d in values for r in d['chain_residual'])}))
PY
