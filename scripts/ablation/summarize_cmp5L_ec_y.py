"""Strict per-arm validation-only completion audit for EC-X0..X5."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'ecdetseg'), str(ROOT)]
from scripts.ablation.analyze_cmp5L_shared_query_kd import (  # noqa: E402
    _fixed_metrics, fixed_threshold_match, load_predictions,
)


def digest(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_epoch_history(records, expected_epochs):
    if [int(item['epoch']) for item in records] != list(range(expected_epochs)):
        raise RuntimeError(f'not exactly {expected_epochs} contiguous training epochs')


def main(args):
    root = Path(args.out_dir)
    marker = root / ('COMPLETED.json' if args.expected_epochs == 100 else 'EARLY_STOPPED.json')
    if marker.exists():
        raise FileExistsError(marker)
    manifest = json.loads(Path(args.manifest).read_text())
    if manifest['student_init_sha256'] != args.init_sha256:
        raise RuntimeError('shared initialization changed')
    if not (root / 'smoke' / 'SMOKE_COMPLETED.json').is_file():
        raise RuntimeError('in-job GPU smoke missing')
    log = rows(root / 'log.txt')
    validate_epoch_history(log, args.expected_epochs)
    epoch_summary = rows(root / 'ecfull_epoch_summary.jsonl')
    validate_epoch_history(epoch_summary, args.expected_epochs)
    if args.arm in ('ECX1', 'ECX2', 'ECX3', 'ECX4'):
        calibration = json.loads((root / 'calibration.json').read_text())
        if not math.isfinite(float(calibration['coefficient'])) or calibration['valid_batches'] < 3:
            raise RuntimeError('in-job calibration failed')
    else:
        calibration = None
    annotation = json.loads(Path(args.ann_file).read_text())
    image_ids = [int(item['id']) for item in annotation['images']]
    if len(image_ids) != 2975 or len(set(image_ids)) != 2975:
        raise RuntimeError('validation image IDs changed')
    annotations_by_image = {image_id: [] for image_id in image_ids}
    for item in annotation['annotations']:
        annotations_by_image[int(item['image_id'])].append(item)
    positive = [image_id for image_id in image_ids if annotations_by_image[image_id]]
    empty = [image_id for image_id in image_ids if not annotations_by_image[image_id]]
    checkpoints, validation = {}, {}
    for name, checkpoint_file in (('best', 'best.pth'), ('final', 'last.pth')):
        checkpoint_path = root / checkpoint_file
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        for key in ('model', 'ema'):
            if key not in checkpoint:
                raise RuntimeError(f'{name} checkpoint lacks {key}')
        if any('teacher' in key.lower() for key in checkpoint['model']):
            raise RuntimeError('external teacher leaked into Student checkpoint')
        if checkpoint.get('ec_fullcycle_init_sha256') != args.init_sha256:
            raise RuntimeError('checkpoint init lineage changed')
        checkpoints[name] = {'path': str(checkpoint_path), 'sha256': digest(checkpoint_path),
                             'last_epoch': int(checkpoint['last_epoch']),
                             'ema_updates': int(checkpoint['ema']['updates'])}
        del checkpoint
        metrics_path = root / f'validation_{name}' / 'metrics.json'
        prediction_path = metrics_path.with_name('metrics.predictions.jsonl')
        record = json.loads(metrics_path.read_text())
        if record['meta']['weights'] != 'ema' or int(record['meta']['n_images']) != 2975:
            raise RuntimeError(f'{name} EMA/validation coverage mismatch')
        if any(not math.isfinite(float(value)) for key, value in record['metrics'].items()
               if key in ('ap', 'ap50', 'ap75', 'ar100', 'precision', 'recall', 'f1')):
            raise RuntimeError(f'{name} nonfinite official metrics')
        predictions = load_predictions(prediction_path, image_ids)
        matches = {image_id: fixed_threshold_match(
            annotations_by_image[image_id], predictions[image_id], .5, .5)
            for image_id in image_ids}
        validation[name] = {
            'official': record['metrics'],
            'per_class_ap50': record['per_class_ap50'],
            'fixed_rule': {'full': _fixed_metrics(matches, image_ids),
                           'positive': _fixed_metrics(matches, positive),
                           'empty': _fixed_metrics(matches, empty)},
            'metrics_path': str(metrics_path), 'prediction_path': str(prediction_path),
            'prediction_sha256': digest(prediction_path),
        }
    if checkpoints['final']['last_epoch'] != args.expected_epochs - 1:
        raise RuntimeError('final checkpoint epoch does not match completed training log')
    milestones = {}
    frozen_teacher = None
    if args.y_arm:
        if args.arm != 'ECX3' or args.expected_epochs != 100:
            raise RuntimeError('Y arms require the complete X3-derived 100-epoch protocol')
        if any(item.get('y_arm') != args.y_arm for item in epoch_summary):
            raise RuntimeError('Y arm training epoch identity changed')
        for item in epoch_summary:
            epoch = int(item['epoch'])
            expected_late = (max(0., (80. - epoch) / 30.)
                             if args.y_arm in ('Y1', 'Y3') and epoch >= 50 else 1.)
            if abs(float(item['late_factor_start']) - expected_late) > 1e-9:
                raise RuntimeError(f'Y late KD factor mismatch at epoch{epoch}')
        for epoch in (49, 59, 79):
            path = root / f'checkpoint{epoch:04}.pth'
            state = torch.load(path, map_location='cpu', weights_only=True)
            if int(state['last_epoch']) != epoch or state.get('ec_y_arm') != args.y_arm:
                raise RuntimeError(f'Y milestone epoch{epoch} mismatch')
            milestones[str(epoch)] = {'path': str(path), 'sha256': digest(path)}
            del state
        milestones['99'] = checkpoints['final']
        if args.y_arm in ('Y2', 'Y3'):
            path = root / 'frozen_np_teacher_epoch49.pth'
            state = torch.load(path, map_location='cpu', weights_only=True)
            if state['source_epoch'] != 49 or state['init_sha256'] != args.init_sha256:
                raise RuntimeError('Y frozen teacher provenance mismatch')
            frozen_teacher = {'path': str(path), 'sha256': digest(path),
                              'ema_updates_at_freeze': int(state['ema_updates'])}
            del state
        elif (root / 'frozen_np_teacher_epoch49.pth').exists():
            raise RuntimeError('nonfreeze Y arm wrote a frozen teacher')
        fixed_scores_path = root / 'ec_y_fixed_teacher_scores.jsonl'
        fixed_scores = rows(fixed_scores_path)
        if [int(item['epoch']) for item in fixed_scores] != [49, 59, 79, 99]:
            raise RuntimeError('Y fixed diagnostic epochs missing')
        if any(item['y_arm'] != args.y_arm or len(item['batches']) != 6
               or any(batch['teacher_topk_calls'] != 0 for batch in item['batches'])
               for item in fixed_scores):
            raise RuntimeError('Y fixed teacher diagnostic coverage/parity failed')
        fixed_gradient_path = root / 'ec_y_fixed_gradient_ratios.jsonl'
        fixed_gradients = rows(fixed_gradient_path)
        if [int(item['epoch']) for item in fixed_gradients] != [49, 59, 79, 99]:
            raise RuntimeError('Y fixed gradient diagnostic epochs missing')
        if any(item['y_arm'] != args.y_arm or len(item['per_rank']) != 4
               or any(rank['teacher_topk_calls'] != 0 or
                      not math.isfinite(rank['det_l3_norm']) or
                      not math.isfinite(rank['kd_l3_norm'])
                      for rank in item['per_rank']) for item in fixed_gradients):
            raise RuntimeError('Y fixed gradient diagnostic coverage/finite check failed')
    else:
        fixed_scores_path = None
        fixed_gradient_path = None
    result = {
        'arm': args.arm, 'seed': 42, 'epochs': args.expected_epochs,
        'completion_type': 'full_cycle' if args.expected_epochs == 100 else 'externally_early_stopped',
        'early_stop_patience': None if args.expected_epochs == 100 else args.early_stop_patience,
        'split': 'validation only', 'image_count': len(image_ids),
        'positive_images': len(positive), 'empty_images': len(empty),
        'init_sha256': args.init_sha256,
        'manifest_sha256': digest(args.manifest),
        'config': args.config, 'config_sha256': digest(args.config),
        'source_head': os.environ.get('EC_SNAPSHOT_GIT_HEAD', 'not_recorded'),
        'y_arm': args.y_arm, 'milestone_checkpoints': milestones,
        'frozen_teacher': frozen_teacher,
        'fixed_teacher_scores_path': str(fixed_scores_path) if fixed_scores_path else None,
        'fixed_gradient_ratios_path': str(fixed_gradient_path) if fixed_gradient_path else None,
        'calibration': calibration, 'checkpoints': checkpoints,
        'validation': validation,
        'epoch_summary_path': str(root / 'ecfull_epoch_summary.jsonl'),
        'training_log_path': str(root / 'log.txt'),
        'fixed_rule': 'score>=0.5; IoU>=0.5; class-constrained score-ordered one-to-one',
    }
    temporary = marker.with_name(marker.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(result, indent=2, default=str) + '\n')
    temporary.replace(marker)
    print(json.dumps({'arm': args.arm, 'completed': True,
                      'best_ap50': validation['best']['official']['ap50'],
                      'final_ap50': validation['final']['official']['ap50']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--init-sha256', required=True)
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--expected-epochs', type=int, default=100)
    parser.add_argument('--early-stop-patience', type=int, default=20)
    parser.add_argument('--y-arm', choices=['Y0', 'Y1', 'Y2', 'Y3'])
    main(parser.parse_args())
