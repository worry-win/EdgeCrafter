"""Strict V identity/schedule audit before the existing EC full-cycle metrics audit."""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'ecdetseg'), str(ROOT)]
from scripts.ablation import summarize_cmp5L_ec_fullcycle as base
from scripts.ablation.train_cmp5L_ec_v import recipe, validate_v_execution
from engine.core.yaml_utils import load_config


def main(args):
    root = Path(args.out_dir)
    config = load_config(args.config, {})
    if config['ec_v_arm'] != args.v_arm:
        raise RuntimeError('V configuration identity mismatch')
    validate_v_execution(config, world_size=2)
    expected_arm, expected_schedule = recipe(args.v_arm)
    if args.arm != expected_arm:
        raise RuntimeError('V underlying method mismatch')
    log = base.rows(root / 'log.txt')
    records = base.rows(root / 'ecfull_epoch_summary.jsonl')
    if [r['epoch'] for r in log] != list(range(100)) or [r['epoch'] for r in records] != list(range(100)):
        raise RuntimeError('V training did not complete 100 contiguous epochs')
    for record in records:
        epoch = int(record['epoch'])
        factor = max(0., (80. - epoch) / 30.) if args.v_arm != 'V0' and epoch >= 50 else 1.
        if record['y_arm'] != expected_schedule or abs(record['late_factor_start'] - factor) > 1e-10:
            raise RuntimeError(f'V late schedule mismatch epoch{epoch}')
    calibration = None
    if args.v_arm == 'V0':
        if (root / 'calibration.json').exists() or any(r['coefficient'] is not None for r in records):
            raise RuntimeError('V0 unexpectedly has KD coefficient')
    else:
        calibration = json.loads((root / 'calibration.json').read_text())
        if not math.isfinite(float(calibration['coefficient'])) or calibration['valid_batches'] < 3:
            raise RuntimeError('V calibration incomplete')
        if args.v_arm == 'V4' and not (0 < calibration['box_coefficient'] <= 10):
            raise RuntimeError('V4 box coefficient missing/outside cap')
        diagnostics = base.rows(root / 'ecfull_diagnostics.jsonl')
        if any(row['teacher_topk_calls'] not in (None, 0) for row in diagnostics):
            raise RuntimeError('V teacher reselected Top-K')
        if args.v_arm == 'V6' and not any(row['teacher_memory_delta'] > 0 for row in diagnostics
                                         if row['teacher_memory_delta'] is not None):
            raise RuntimeError('V6 privileged memory was never applied')
        if args.v_arm == 'V7' and not any(row['x5_intervention'] for row in diagnostics):
            raise RuntimeError('V7 student intervention never ran')
    args.y_arm = None  # V audit above accepts ECX0 as the no-KD reference.
    args.expected_epochs = 100
    args.external_stop_reason = None
    args.early_stop_patience = 0
    base.main(args)
    marker = root / 'COMPLETED.json'
    result = json.loads(marker.read_text())
    result.update(v_arm=args.v_arm, underlying_method=expected_arm,
                  schedule_alias=expected_schedule, calibration=calibration,
                  execution={'world_size': 2, 'per_gpu_batch': 16,
                             'gradient_accumulation': 1, 'effective_batch': 32})
    marker.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('arm', 'v-arm', 'out-dir', 'config', 'manifest', 'init-sha256', 'ann-file'):
        p.add_argument('--' + name, required=True)
    main(p.parse_args())
