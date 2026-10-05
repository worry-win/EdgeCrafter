"""One-shot, read-only six-arm validation comparison; never polls or submits work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ARMS = [f'ECX{index}' for index in range(6)]
CONTRASTS = [('ECX1', 'ECX0'), ('ECX3', 'ECX2'), ('ECX3', 'ECX1'),
             ('ECX4', 'ECX3'), ('ECX5', 'ECX0')]


def discover_arms(root):
    ready = [arm for arm in ARMS if (Path(root) / arm / 'COMPLETED.json').is_file()]
    return ready, [arm for arm in ARMS if arm not in ready]


def run(args):
    import numpy as np
    from scripts.ablation.analyze_cmp5L_shared_query_kd import (
        _flatten_for_coco, coco_ap50_focus, load_predictions,
        remap_bootstrap_sample,
    )

    root = Path(args.result_root)
    ready, pending = discover_arms(root)
    if pending:
        print(json.dumps({'ready': ready, 'pending': pending}, sort_keys=True))
        raise SystemExit(2)
    ground_truth = json.loads(Path(args.ann_file).read_text())
    image_ids = [int(item['id']) for item in ground_truth['images']]
    if len(image_ids) != 2975 or len(set(image_ids)) != 2975:
        raise ValueError('validation image coverage changed')
    completions = {arm: json.loads((root / arm / 'COMPLETED.json').read_text()) for arm in ARMS}
    if any(completions[arm].get('arm') != arm or completions[arm].get('seed') != 42
           or completions[arm].get('epochs') != 100 for arm in ARMS):
        raise ValueError('arm completion metadata mismatch')
    results = {}
    for checkpoint in ('best', 'final'):
        predictions = {}
        for arm in ARMS:
            path = Path(completions[arm]['validation'][checkpoint]['prediction_path'])
            predictions[arm] = load_predictions(path, image_ids)
        capped = {
            arm: {image_id: sorted(predictions[arm][image_id],
                                   key=lambda item: item['score'], reverse=True)[:100]
                  for image_id in image_ids}
            for arm in ARMS
        }
        reference = {arm: coco_ap50_focus(
            ground_truth, _flatten_for_coco(capped[arm], image_ids)) for arm in ARMS}
        for arm in ARMS:
            official = completions[arm]['validation'][checkpoint]['official']['ap50']
            if abs(reference[arm]['ap50'] - official) > .001:
                raise ValueError(f'{arm}/{checkpoint} AP50 parity failed')
        rng = np.random.default_rng(args.seed)
        samples = {f'{left}_minus_{right}': {'ap50': [], 'class_3_ap50': []}
                   for left, right in CONTRASTS}
        for _ in range(args.repetitions):
            draw = rng.choice(image_ids, size=len(image_ids), replace=True)
            sample_gt, sample_predictions = remap_bootstrap_sample(ground_truth, capped, draw)
            scored = {arm: coco_ap50_focus(sample_gt, sample_predictions[arm]) for arm in ARMS}
            for left, right in CONTRASTS:
                for metric in ('ap50', 'class_3_ap50'):
                    samples[f'{left}_minus_{right}'][metric].append(
                        scored[left][metric] - scored[right][metric])
        intervals = {
            contrast: {
                metric: {'point': reference[left][metric] - reference[right][metric],
                         'ci95': np.quantile(values, [.025, .975]).tolist(),
                         'valid_repetitions': len(values)}
                for metric, values in metrics.items()
                for left, right in [next(pair for pair in CONTRASTS
                                         if contrast == f'{pair[0]}_minus_{pair[1]}')]
            }
            for contrast, metrics in samples.items()
        }
        results[checkpoint] = {'reference': reference, 'paired_intervals': intervals}
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'arms': ARMS, 'contrasts': CONTRASTS,
                                  'repetitions': args.repetitions, 'results': results},
                                 indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-root', required=True)
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--repetitions', type=int, default=200)
    parser.add_argument('--seed', type=int, default=20260925)
    run(parser.parse_args())
