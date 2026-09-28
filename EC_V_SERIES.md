# EC V-series source (V0–V8)

This branch preserves the cmp5L EC V-series implementation for later adaptation to other detection datasets. V0–V6 come from the frozen `cmp5L_EC_V_eight_v1/code_snapshot_v1`; V7 uses the completed `code_snapshot_v7_fix1`; V8 method code comes from `cmp5L_EC_V8_ref_v1/code_snapshot_v1`. The V8 full training run had not been validated when this source was assembled. Do not treat this branch as an evaluated V8 result.

## What is included

- `scripts/ablation/train_cmp5L_ec_v.py` implements V0–V8. `cmp5L_ec_v_losses.py` contains the V3 candidate-set KL and V8 fixed-background-reference variant.
- The local training, checkpoint evaluation, summary, and support modules imported by that entry point are present under `scripts/ablation/`.
- `ecdetseg/configs/ecdet/*cmp5L_ECV*.yml` and their include chain preserve the breast experiment's model and training settings.
- Selected solver, distributed evaluation, and COCO evaluator changes needed by this implementation are included. The repository's other experimental decoder gates and alternative backbones were left out.
- Four historical launch scripts are force-tracked under `slurm/`: the original V0–V6 launcher, V7 repair, V8 preflight, and V8 full run. They record how the cluster jobs were submitted; their paths refer to frozen snapshots and are **not** reusable unchanged on another dataset.

Weights, images, annotations, manifests, predictions, logs, and `outputs/` are excluded. In particular, the DINOv2 backbone weight and the shared seed-42 detector initialization must be supplied separately and verified by hash.

## Reusing the method on another dataset

1. Start from a separate, approved dataset split and a normal EC baseline and initialization suitable for that dataset. Do not load a mature breast detector by default.
2. Copy the relevant V YAML and replace dataset paths, class count and mapping, image size, initialization, and the original dataset's optimizer and augmentation recipe. Inspect the entire YAML include chain. The current breast config uses four classes, 640×640 images, and cluster-specific paths.
3. Adapt `validate_v_execution`, calibration manifest construction, and the launcher to the approved training budget and hardware. The preserved entry point requires two GPUs, effective batch 32, accumulation 1, SyncBN, 100 epochs, and the breast augmentation stop epochs. It is intentionally dataset-specific.
4. Recalibrate any added KD coefficient using training data only, without optimizer updates; do not copy the breast coefficient. Keep the same predefined checkpoint-selection rule across arms.
5. Run short technical checks before a full run. Save the validation-selected best EMA checkpoint and evaluate by the normal student path. Keep evaluation GT outside the model forward.

The historical Slurm scripts target `/cobot/Code/wanrui/EdgeCrafter` and its frozen `outputs/ablation` snapshots. They are provenance, not a general cluster deployment interface. Their `INIT`, `MANIFEST`, `IMAGE_ROOT`, and annotation paths must resolve in the target environment before use.

## Code checks

On the matching EC Python environment, run:

```sh
PYTHONPATH=.:ecdetseg python -m unittest ecdetseg.tests.test_cmp5l_ec_v
```

The frozen source snapshot and historical launch scripts establish implementation provenance. They do not supply the excluded datasets or checkpoints, and V8's full-run outcome must be checked separately.
