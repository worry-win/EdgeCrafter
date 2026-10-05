"""Paired validation analysis for locked KQ-W/KQ-O versus KQ0/KQ2."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.ablation.analyze_cmp5L_shared_query_kd import (
    _fixed_metrics,
    _flatten_for_coco,
    classify_gt_transitions,
    coco_ap50_focus,
    coco_metrics,
    fixed_threshold_match,
    load_predictions,
    remap_bootstrap_sample,
)


GROUPS = ("KQ0", "KQ2", "KQW", "KQO")
CONTRASTS = {
    "KQW_minus_KQ0": ("KQ0", "KQW"),
    "KQW_minus_KQ2": ("KQ2", "KQW"),
    "KQO_minus_KQ0": ("KQ0", "KQO"),
    "KQO_minus_KQ2": ("KQ2", "KQO"),
    "KQO_minus_KQW": ("KQW", "KQO"),
}
METRICS = ("ap", "ap50", "ap75", "ar100", "precision", "recall", "f1")


def _arm_roots(args):
    old = Path(args.original_root)
    new = Path(args.extension_root)
    return {"KQ0": old / "KQ0", "KQ2": old / "KQ2", "KQW": new / "KQW", "KQO": new / "KQO"}


def _load_split(roots, split, image_ids, ann_file):
    metrics = {}
    predictions = {}
    for group in GROUPS:
        path = roots[group] / f"validation_{split}" / "metrics.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        meta = record["meta"]
        if int(meta["n_images"]) != len(image_ids) or meta["weights"] != "ema":
            raise ValueError(f"{group}/{split} is not the expected complete EMA evaluation")
        if Path(meta["ann_file"]).resolve() != Path(ann_file).resolve():
            raise ValueError(f"{group}/{split} annotation path differs")
        for key in METRICS:
            if not math.isfinite(float(record["metrics"][key])):
                raise ValueError(f"{group}/{split} has non-finite {key}")
        pred_path = path.with_name("metrics.predictions.jsonl")
        metrics[group] = record
        predictions[group] = load_predictions(pred_path, image_ids)
    return metrics, predictions


def _matches(ground_truth, annotations_by_image, predictions, image_ids, score_threshold, iou_threshold):
    positive = [image_id for image_id in image_ids if annotations_by_image[image_id]]
    empty = [image_id for image_id in image_ids if not annotations_by_image[image_id]]
    per_image = {group: {} for group in GROUPS}
    fixed = {}
    positive_coco = {}
    for group in GROUPS:
        for image_id in image_ids:
            per_image[group][image_id] = fixed_threshold_match(
                annotations_by_image[image_id], predictions[group][image_id],
                score_threshold, iou_threshold,
            )
        fixed[group] = {
            "full": _fixed_metrics(per_image[group], image_ids),
            "positive_images": _fixed_metrics(per_image[group], positive),
            "empty_images": _fixed_metrics(per_image[group], empty),
        }
        positive_coco[group] = coco_metrics(
            ground_truth, _flatten_for_coco(predictions[group], positive), positive,
        )
    return per_image, fixed, positive_coco


def _contrasts(metrics, fixed, per_image, annotations, image_ids):
    point_deltas = {}
    focus_transitions = {}
    for name, (left, right) in CONTRASTS.items():
        left_metrics = metrics[left]["metrics"]
        right_metrics = metrics[right]["metrics"]
        delta = {key: float(right_metrics[key] - left_metrics[key]) for key in METRICS}
        for category_id in range(4):
            delta[f"class_{category_id}_ap50"] = float(
                metrics[right]["per_class_ap50"][str(category_id)]["ap50"]
                - metrics[left]["per_class_ap50"][str(category_id)]["ap50"]
            )
        for field in ("precision", "recall", "f1", "fp_per_image"):
            delta[f"fixed_{field}"] = float(
                fixed[right]["full"][field] - fixed[left]["full"][field]
            )
        delta["empty_fp_per_image"] = float(
            fixed[right]["empty_images"]["fp_per_image"]
            - fixed[left]["empty_images"]["fp_per_image"]
        )
        point_deltas[name] = delta
        focus_transitions[name] = {}
        for category_id in range(4):
            category = [item for item in annotations if int(item["category_id"]) == category_id]
            left_hits = {
                int(item["id"]): per_image[left][int(item["image_id"])]["gt_matches"].get(int(item["id"]))
                for item in category
            }
            right_hits = {
                int(item["id"]): per_image[right][int(item["image_id"])]["gt_matches"].get(int(item["id"]))
                for item in category
            }
            focus_transitions[name][str(category_id)] = classify_gt_transitions(left_hits, right_hits)
    return point_deltas, focus_transitions


def _paired_bootstrap(ground_truth, predictions, image_ids, repetitions, seed):
    capped = {
        group: {
            image_id: sorted(predictions[group][image_id], key=lambda item: item["score"], reverse=True)[:100]
            for image_id in image_ids
        }
        for group in GROUPS
    }
    rng = np.random.default_rng(seed)
    keys = ("ap50", "class_2_ap50", "class_3_ap50")
    samples = {name: {key: [] for key in keys} for name in CONTRASTS}
    for index in range(repetitions):
        draw = rng.choice(image_ids, size=len(image_ids), replace=True)
        sample_gt, sample_arms = remap_bootstrap_sample(ground_truth, capped, draw)
        scored = {group: coco_ap50_focus(sample_gt, arm) for group, arm in sample_arms.items()}
        for name, (left, right) in CONTRASTS.items():
            for key in keys:
                samples[name][key].append(scored[right][key] - scored[left][key])
        if (index + 1) % 10 == 0:
            print(f"bootstrap {index + 1}/{repetitions}", flush=True)
    intervals = {}
    for name, by_key in samples.items():
        intervals[name] = {}
        for key, values in by_key.items():
            finite = np.asarray(values, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size != repetitions:
                raise ValueError(f"non-finite bootstrap results for {name}/{key}")
            intervals[name][key] = {
                "ci95_percentile": np.quantile(finite, [0.025, 0.975]).tolist(),
                "mean": float(finite.mean()),
                "positive_fraction": float((finite > 0).mean()),
                "valid_repetitions": int(finite.size),
            }
    return intervals


def run(args):
    roots = _arm_roots(args)
    ground_truth = json.loads(Path(args.ann_file).read_text(encoding="utf-8"))
    image_ids = [int(item["id"]) for item in ground_truth["images"]]
    if len(image_ids) != args.expected_images or len(set(image_ids)) != args.expected_images:
        raise ValueError("validation image count/uniqueness differs from expected")
    by_image = defaultdict(list)
    for annotation in ground_truth["annotations"]:
        by_image[int(annotation["image_id"])].append(annotation)
    results = {}
    best_predictions = None
    for split in ("best", "final"):
        metrics, predictions = _load_split(roots, split, image_ids, args.ann_file)
        per_image, fixed, positive_coco = _matches(
            ground_truth, by_image, predictions, image_ids, args.score_threshold, args.iou_threshold,
        )
        deltas, transitions = _contrasts(metrics, fixed, per_image, ground_truth["annotations"], image_ids)
        results[split] = {
            "official": metrics,
            "fixed": fixed,
            "positive_image_coco": positive_coco,
            "point_deltas": deltas,
            "class_gt_transitions": transitions,
        }
        if split == "best":
            best_predictions = predictions
    intervals = _paired_bootstrap(
        ground_truth, best_predictions, image_ids, args.bootstrap_repetitions, args.seed,
    )
    payload = {
        "scope": "cmp5L KQ0/KQ2/KQ-W/KQ-O validation-only; best/final EMA",
        "split": {
            "images": len(image_ids),
            "positive_images": sum(bool(by_image[item]) for item in image_ids),
            "empty_images": sum(not by_image[item] for item in image_ids),
        },
        "fixed_rule": {
            "score_threshold": args.score_threshold,
            "iou_threshold": args.iou_threshold,
            "matching": "score-ordered class-constrained one-to-one",
        },
        "results": results,
        "bootstrap": {
            "checkpoint": "best EMA",
            "method": "paired image resampling with shared draws and COCO AP50 recomputation",
            "repetitions": args.bootstrap_repetitions,
            "seed": args.seed,
            "limitation": "single seed and image-level rather than patient-level resampling",
            "intervals": intervals,
        },
        "arm_roots": {group: str(path) for group, path in roots.items()},
    }
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".COMPLETED").write_text("KQ-W/KQ-O paired validation analysis complete\n")
    print(json.dumps(results["best"]["point_deltas"], ensure_ascii=False, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", required=True)
    parser.add_argument("--extension-root", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-images", type=int, default=2975)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-repetitions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260923)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
