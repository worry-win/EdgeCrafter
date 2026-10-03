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


def main(args):
    root = Path(args.out_dir)
    marker = root / 'COMPLETED.json'
    if marker.exists():
        raise FileExistsError(marker)
    manifest = json.loads(Path(args.manifest).read_text())
    if manifest['student_init_sha256'] != args.init_sha256:
        raise RuntimeError('shared initialization changed')
    if not (root / 'smoke' / 'SMOKE_COMPLETED.json').is_file():
        raise RuntimeError('in-job GPU smoke missing')
    log = rows(root / 'log.txt')
    if [int(item['epoch']) for item in log] != list(range(100)):
        raise RuntimeError('not exactly 100 contiguous training epochs')
    epoch_summary = rows(root / 'ecfull_epoch_summary.jsonl')
    if [int(item['epoch']) for item in epoch_summary] != list(range(100)):
        raise RuntimeError('missing training epoch diagnostics')
    if args.arm in ('ECX1', 'ECX2', 'ECX3', 'ECX4'):
        calibration = json.loads((root / 'calibration.json').read_text())
        if not math.isfinite(float(calibration['coefficient'])) or calibration['valid_batches'] < 3:
            raise RuntimeError('in-job calibration failed')
    else:
        calibration = None
    annotation = json.loads(Path(args.ann_file).read_text())
    image_ids = [int(item['id']) for item in annotation['images']]
    expected_images = int(getattr(args, 'expected_images', 2975))
    if len(image_ids) != expected_images or len(set(image_ids)) != expected_images:
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
        if record['meta']['weights'] != 'ema' or int(record['meta']['n_images']) != expected_images:
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
    if checkpoints['final']['last_epoch'] != 99:
        raise RuntimeError('final checkpoint is not epoch99')
    result = {
        'arm': args.arm, 'seed': 42, 'epochs': 100,
        'split': 'validation only', 'image_count': len(image_ids),
        'positive_images': len(positive), 'empty_images': len(empty),
        'init_sha256': args.init_sha256,
        'manifest_sha256': digest(args.manifest),
        'config': args.config, 'config_sha256': digest(args.config),
        'source_head': os.environ.get('EC_SNAPSHOT_GIT_HEAD', 'not_recorded'),
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
    parser.add_argument('--expected-images', type=int, default=2975)
    main(parser.parse_args())
