# Historical cmp5L V0–V7 research branch

This branch rolls back to the completed 2026-09-27 V-series source. It does not include ES or V8 method changes. `scripts/ablation/train_cmp5L_ec_v.py` is byte-identical to the completed V7 fix1 trainer; `historical_versions/v1/` preserves the original trainer and test for exact V0–V6 execution snapshots. All other historical model, decoder, solver, evaluator, loss and YAML include files are preserved.

## Eight arms

| Arm | Added supervision / intervention |
|---|---|
| V0 | Normal EC, no added KD |
| V1 | EMA NP teacher, final-layer candidate pair ranking |
| V2 | Same ranking at four decoder layers, averaged |
| V3 | EMA NP teacher, final-layer candidate-set query softmax KL |
| V4 | V1 plus quality-filtered final-layer L1/GIoU box KD |
| V5 | NP-versus-normal teacher ranking-increment weighting |
| V6 | V1 with milder NP background coefficient 0.5 |
| V7 | V1 plus historical stochastic student background intervention; completed fix1 |

V3 does not distill all 300 queries or boxes; V4 is pair ranking plus box KD, not V3 plus box KD.

## Locked historical recipe

- Seed42; two RTX5090 GPUs; per-rank batch16; accumulation1; global batch32; SyncBN; AMP.
- DINOv2-S backbone and EC-L detector, 640×640, 300 normal queries, four decoder layers and four breast classes.
- Same seed42 full detector initialization, SHA256 `94f3f1876b9923b0263c1ba55ee4dea13ce4c5ff504288f2f85675887f4111e0`. Last epoch -1; EMA equals student. Decoder canonical state hash `23116e13844e74391e9a5eafb434809fae8cfc3c1f2f03b28921f74a796ae360` over 159 tensors including buffers.
- 100 epochs, no early stopping. Historical optimizer/LR, Mosaic/MixUp cutoff24 and strong augmentation cutoff98. Original epoch98 best reload retained; final is not a natural continuous endpoint.
- Added KD warmup10–20 and decay50–80. Train-only, no-update calibration at t10 from the original fixed manifest; class target L3 gradient ratio0.30, V4 box target0.10 and cap10. Recompute coefficients, do not force old values or tune with validation.
- Best selected by original validation rule; best/final evaluated by normal student EMA, own Top-K, no GT teacher replay. Validation2975 images. No independent test.

## Execution and provenance

Canonical replay output: `/cobot/Code/wanrui/EdgeCrafter/outputs/ablation/cmp5L_EC_V0_V7_history_2026-10-05_v1`.
Logs: `/cobot/Code/wanrui/EdgeCrafter/logs/slurm/cmp5L_EC_V_history_2026-10-05`.
Historical source: `outputs/ablation/cmp5L_EC_V_eight_v1/code_snapshot_v1` and `code_snapshot_v7_fix1`.
The launcher uses separate immutable execution snapshots: original v1 for V0–V6, fix1 for V7. A read-only wrapper records student/EMA decoder hashes on each rank before optimizer updates. It does not change loss, RNG, optimizer, AMP, EMA or CUDA math. The observer adds synchronization, so this is not a claim of bitwise historical replay.

Each GPU job independently performs CPU tests, a no-update CUDA backward evidence probe, at least five genuine AMP/DDP smoke updates, a fresh initialization load for full training, best/final validation prediction export and completion audits. Failed stages preserve logs and stop; no automatic training resubmission. Jobs do not require a connected laptop.

The CUDA probe is a separate process, repeats the production deformable attention on identical inputs, and checks gradients. Its deterministic-algorithm check is diagnostic only and never applied to real training.

`LOCKED_PROTOCOL.json` records configs including resolved include chains, source hashes, initialization/backbone/calibration hashes, current train/validation annotations and per-image byte manifests, package versions and historical completed markers. `JOBS.json` records actual submissions. Complete historical image-byte and package/driver manifests were not recorded; that historical equivalence remains unproven. Same initialization and seed cannot guarantee identical CUDA gradients or restoration of old AP.

## Future research

Begin later methods from this branch, keep this baseline recipe immutable, and create separate versioned configurations and outputs for each authorized method. Do not silently mix ES early stopping, shared warmup, 100+2 finish phases or V8 reference KL into V recipes. Adapting datasets requires approved splits/class maps, appropriate common initialization, and fresh train-only calibration. Never start another organ from a mature breast V3 best by default.

Checkpoints, data, predictions, logs and outputs are not in Git. Only the necessary Slurm scripts are tracked despite the inherited ignore rule. Cluster asset paths in the historical YAML are explicit provenance; a new environment must provide and verify them before running.

The historical Linux tree contains two test names differing only in `cmp5L` versus `cmp5l`, with identical content. Git preserves both names; case-insensitive macOS may show one physical file. Execution snapshots are verified on Linux against the original full file manifest. This alias has no role in training.

## Authorized extension: V3_4L

One separate breast experiment applies original `candidate_set_kd` at L0/L1/L2/L3. Each layer independently uses its own student Hungarian matches/boxes/logits to select candidates and requires the teacher lesion to rank strictly above all negatives. Four losses average with weights1/4 including invalid zero layers. It uses no reference logit, BCE, box/hidden KD or student intervention. Teacher all-layer export uses the validated V2 path, with child BN/Dropout eval and final-layer parity.

Use `train_cmp5L_ec_v3_four_layer.py` and `slurm/cmp5L_EC_V3_4L_one_arm.sbatch`; original trainer files are unchanged. Full historical V3 recipe/init/split/100 epochs/epoch98 reload remain; t10 coefficients independently recalibrate aggregated loss to L3 gradient ratio0.30. This does not hold total-model KD gradient budget fixed. Compare to contemporaneous V3 Job4526, best EMA main, normal validation only. Output: `outputs/ablation/cmp5L_EC_V3_4L_2026-10-05_v1`.
