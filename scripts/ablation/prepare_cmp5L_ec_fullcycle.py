"""Lock one seed-42 random detector initialization and train-only calibration batches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'ecdetseg'), str(ROOT)]
from engine.core import YAMLConfig  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(value, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def tensor_targets(targets):
    return [{key: value.as_subclass(torch.Tensor).clone() if torch.is_tensor(value) else value
             for key, value in target.items()} for target in targets]


def run(args):
    output = Path(args.out_dir)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    cfg = YAMLConfig(args.config)
    resolved = cfg.yaml_cfg
    transform = resolved['train_dataloader']['dataset']['transforms']
    collate = resolved['train_dataloader']['collate_fn']
    expected = (100, 24, 24, 98, 2, 16)
    actual = (resolved['epochs'], transform['mosaic_epoch'], collate['mixup_epoch'],
              transform['stop_epoch'], resolved['gradient_accumulation_steps'],
              resolved['train_dataloader']['total_batch_size'])
    if actual != expected or resolved['early_stop_patience'] != 0:
        raise RuntimeError(f'full-cycle recipe changed: {actual} != {expected}')
    model = cfg.model
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    backbone_path = Path(resolved['DinoV2Adapter']['weights_path'])
    if not backbone_path.is_file() or resolved['DinoV2Adapter']['skip_load_backbone']:
        raise RuntimeError('official DINOv2-S backbone was not loaded')
    init_path = output / 'student_init_seed42.pth'
    atomic_torch_save({
        'model': state,
        'ema': {'module': state, 'updates': 0},
        'last_epoch': -1,
        'source': 'seed42 original DINOv2-S backbone only; random EC detector',
        'pretrained_sha256': sha256(backbone_path),
    }, init_path)
    del model, state
    loader = cfg.train_dataloader
    index = {int(image_id): position for position, image_id in enumerate(loader.dataset.ids)}
    if len(index) != len(loader.dataset.ids):
        raise RuntimeError('duplicate train image IDs')
    pairs = [(2, 4), (1, 40), (3, 5), (95, 59), (6, 8), (96, 73)]
    records = []
    all_classes = set()
    for row, image_ids in enumerate(pairs + [(31, 114)]):
        epoch = 10 if row < len(pairs) else 98
        loader.set_epoch(epoch)
        random.seed(4200 + row)
        np.random.seed(4200 + row)
        torch.manual_seed(4200 + row)
        samples, targets = loader.collate_fn([loader.dataset[index[item]] for item in image_ids])
        targets = tensor_targets(targets)
        classes = sorted({int(label) for item in targets for label in item['labels'].reshape(-1).tolist()})
        all_classes.update(classes if row < len(pairs) else [])
        if row == len(pairs) and (classes or any(len(item['labels']) for item in targets)):
            raise RuntimeError('locked empty batch was not empty')
        path = output / f'calibration_batch_{row:02d}.pt'
        atomic_torch_save({'samples': samples.as_subclass(torch.Tensor).clone(),
                           'targets': targets, 'image_ids': image_ids, 'epoch': epoch}, path)
        records.append({'path': str(path), 'sha256': sha256(path), 'image_ids': image_ids,
                        'epoch': epoch, 'class_ids': classes,
                        'gt_count': sum(len(item['labels']) for item in targets),
                        'purpose': 'calibration' if row < len(pairs) else 'empty_zero_check'})
    if all_classes != {0, 1, 2, 3}:
        raise RuntimeError(f'fixed augmented calibration does not cover four classes: {all_classes}')
    manifest = {
        'seed': 42, 'split': 'train', 'selection': 'prelocked historical KQR/KQU pairs at full-cycle epoch 10; empty control at epoch 98',
        'config': args.config, 'config_sha256': sha256(args.config),
        'student_init': str(init_path), 'student_init_sha256': sha256(init_path),
        'pretrained_path': str(backbone_path), 'pretrained_sha256': sha256(backbone_path),
        'model_key_count': len(torch.load(init_path, map_location='cpu', weights_only=True)['model']),
        'recipe': actual, 'batches': records,
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'student_init_sha256': manifest['student_init_sha256'],
                      'pretrained_sha256': manifest['pretrained_sha256'],
                      'model_key_count': manifest['model_key_count'],
                      'class_coverage': sorted(all_classes), 'recipe': actual}, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--out-dir', required=True)
    run(parser.parse_args())
