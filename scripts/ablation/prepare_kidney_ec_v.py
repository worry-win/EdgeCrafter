"""Create the kidney V-series shared seed-42 init and train-only calibration manifest."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from scripts.ablation.prepare_cmp5L_ec_fullcycle import (
    atomic_torch_save, sha256, tensor_targets,
)
from scripts.ablation.train_cmp5L_ec_v import validate_v_execution
from engine.core import YAMLConfig


def select_calibration_image_pairs(annotation, *, num_classes, pairs_per_class=1):
    """Choose disjoint train pairs spread over each class, then an empty control."""
    if pairs_per_class < 1:
        raise ValueError("pairs_per_class must be positive")
    image_ids = {int(image['id']) for image in annotation['images']}
    by_class = {index: set() for index in range(num_classes)}
    occupied = set()
    for item in annotation['annotations']:
        image_id = int(item['image_id'])
        category = int(item['category_id'])
        if image_id not in image_ids:
            raise ValueError(f'annotation references absent image: {image_id}')
        occupied.add(image_id)
        if category in by_class and not item.get('iscrowd', 0):
            by_class[category].add(image_id)
    selected = set()
    pairs = []
    for category, candidates in by_class.items():
        available = sorted(candidates - selected)
        required = 2 * pairs_per_class
        if len(available) < required:
            raise ValueError(f'class {category} lacks {required} disjoint training images')
        positions = [round(i * (len(available) - 1) / (required - 1))
                     for i in range(required)]
        chosen = [available[position] for position in positions]
        for offset in range(0, required, 2):
            pair = chosen[offset:offset + 2]
            pairs.append((category, pair))
            selected.update(pair)
    empty = sorted(image_ids - occupied - selected)
    if len(empty) < 2:
        raise ValueError('train split lacks two empty calibration images')
    pairs.append((None, empty[:2]))
    return pairs


def run(config_path, out_dir, *, pairs_per_class=5):
    output = Path(out_dir)
    if output.exists():
        raise FileExistsError(output)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    cfg = YAMLConfig(config_path)
    resolved = cfg.yaml_cfg
    validate_v_execution(resolved, world_size=2)
    if resolved['ec_v_arm'] != 'V0' or resolved['num_classes'] != 6:
        raise RuntimeError('expected six-class kidney V0 initialization config')
    pretrained = Path(resolved['DinoV2Adapter']['weights_path'])
    if not pretrained.is_file() or resolved['DinoV2Adapter']['skip_load_backbone']:
        raise RuntimeError('DINOv2-S backbone missing or disabled')
    model = cfg.model
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    loader = cfg.train_dataloader
    annotation = json.loads(Path(resolved['train_dataloader']['dataset']['ann_file']).read_text())
    pairs = select_calibration_image_pairs(annotation, num_classes=6,
                                           pairs_per_class=pairs_per_class)
    index = {int(image_id): position for position, image_id in enumerate(loader.dataset.ids)}
    prepared = []
    all_classes = set()
    for row, (category, image_ids) in enumerate(pairs):
        epoch = 98 if category is None else 10
        loader.set_epoch(epoch)
        random.seed(4200 + row)
        np.random.seed(4200 + row)
        torch.manual_seed(4200 + row)
        samples, targets = loader.collate_fn([loader.dataset[index[item]] for item in image_ids])
        targets = tensor_targets(targets)
        classes = sorted({int(label) for target in targets for label in target['labels'].reshape(-1).tolist()})
        if category is None and classes:
            raise RuntimeError('empty calibration control has labels')
        if category is not None and category not in classes:
            raise RuntimeError(f'class {category} lost in augmented calibration batch {row}')
        if category is not None:
            all_classes.update(classes)
        prepared.append((samples.as_subclass(torch.Tensor).clone(), targets,
                         image_ids, classes, epoch, category))
    if all_classes != set(range(6)):
        raise RuntimeError(f'calibration coverage incomplete: {sorted(all_classes)}')
    output.mkdir(parents=True)
    init_path = output / 'student_init_seed42.pth'
    atomic_torch_save({
        'model': state,
        'ema': {'module': state, 'updates': 0},
        'last_epoch': -1,
        'source': 'kidney V-series seed42 DINOv2-S backbone only; random EC-L detector',
        'pretrained_sha256': sha256(pretrained),
    }, init_path)
    records = []
    for row, (samples, targets, image_ids, classes, epoch, category) in enumerate(prepared):
        path = output / f'calibration_batch_{row:02d}.pt'
        atomic_torch_save({'samples': samples, 'targets': targets,
                           'image_ids': image_ids, 'epoch': epoch}, path)
        records.append({'path': str(path), 'sha256': sha256(path),
                        'image_ids': image_ids, 'epoch': epoch, 'class_ids': classes,
                        'gt_count': sum(len(target['labels']) for target in targets),
                        'purpose': 'empty_zero_check' if category is None else 'calibration'})
    manifest = {
        'seed': 42, 'split': 'train',
        'selection': 'deterministic disjoint distributed train image pairs per kidney class; empty control',
        'pairs_per_class': pairs_per_class,
        'config': str(config_path), 'config_sha256': sha256(config_path),
        'student_init': str(init_path), 'student_init_sha256': sha256(init_path),
        'pretrained_path': str(pretrained), 'pretrained_sha256': sha256(pretrained),
        'model_key_count': len(state), 'classes': list(range(6)),
        'batches': records,
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'init_sha256': manifest['student_init_sha256'],
                      'pretrained_sha256': manifest['pretrained_sha256'],
                      'class_coverage': sorted(all_classes), 'batches': len(records)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--pairs-per-class', type=int, default=5)
    args = parser.parse_args()
    run(args.config, args.out_dir, pairs_per_class=args.pairs_per_class)
