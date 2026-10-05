"""Read-only paired validation analysis for locked KQ-R/KQ-U versus KQ0/KQ2/KQ-O."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.ablation.analyze_cmp5L_shared_query_kd import (
    _fixed_metrics, _flatten_for_coco, classify_gt_transitions,
    coco_ap50_focus, coco_metrics, fixed_threshold_match,
    load_predictions, remap_bootstrap_sample,
)


GROUPS = ("KQ0", "KQ2", "KQO", "KQR", "KQU")
CONTRASTS = {
    f"{arm}_minus_{control}": (control, arm)
    for arm in ("KQR", "KQU")
    for control in ("KQO", "KQ2", "KQ0")
}
METRICS = ("ap", "ap50", "ap75", "ar100", "precision", "recall", "f1")


def arm_roots(args):
    original, extension, new = map(Path, (args.original_root, args.extension_root, args.new_root))
    return {
        "KQ0": original / "KQ0", "KQ2": original / "KQ2",
        "KQO": extension / "KQO", "KQR": new / "KQR", "KQU": new / "KQU",
    }


def load_split(roots, split, image_ids, source_ann_file):
    official, predictions = {}, {}
    for group in GROUPS:
        path = roots[group] / f"validation_{split}" / "metrics.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        meta = record["meta"]
        if int(meta["n_images"]) != len(image_ids) or meta["weights"] != "ema":
            raise ValueError(f"{group}/{split} is not complete EMA validation")
        if Path(meta["ann_file"]) != Path(source_ann_file):
            raise ValueError(f"{group}/{split} annotation differs")
        if any(not math.isfinite(float(record["metrics"][key])) for key in METRICS):
            raise ValueError(f"{group}/{split} has non-finite official metrics")
        official[group] = record
        predictions[group] = load_predictions(path.with_name("metrics.predictions.jsonl"), image_ids)
    return official, predictions


def fixed_results(ground_truth, by_image, predictions, image_ids, score_threshold, iou_threshold):
    positive = [image_id for image_id in image_ids if by_image[image_id]]
    empty = [image_id for image_id in image_ids if not by_image[image_id]]
    matches, fixed, positive_coco = {}, {}, {}
    for group in GROUPS:
        matches[group] = {
            image_id: fixed_threshold_match(
                by_image[image_id], predictions[group][image_id], score_threshold, iou_threshold,
            )
            for image_id in image_ids
        }
        fixed[group] = {
            "full": _fixed_metrics(matches[group], image_ids),
            "positive_images": _fixed_metrics(matches[group], positive),
            "empty_images": _fixed_metrics(matches[group], empty),
        }
        positive_coco[group] = coco_metrics(
            ground_truth, _flatten_for_coco(predictions[group], positive), positive,
        )
    return matches, fixed, positive_coco


def comparisons(official, fixed, matches, annotations):
    point, transitions = {}, {}
    for name, (left, right) in CONTRASTS.items():
        point[name] = {
            metric: float(official[right]["metrics"][metric] - official[left]["metrics"][metric])
            for metric in METRICS
        }
        for category in range(4):
            point[name][f"class_{category}_ap50"] = float(
                official[right]["per_class_ap50"][str(category)]["ap50"]
                - official[left]["per_class_ap50"][str(category)]["ap50"]
            )
        for field in ("precision", "recall", "f1", "fp_per_image"):
            point[name][f"fixed_{field}"] = float(
                fixed[right]["full"][field] - fixed[left]["full"][field]
            )
        point[name]["empty_fp_per_image"] = float(
            fixed[right]["empty_images"]["fp_per_image"]
            - fixed[left]["empty_images"]["fp_per_image"]
        )
        transitions[name] = {}
        for category in range(4):
            subset = [item for item in annotations if int(item["category_id"]) == category]
            left_hits = {
                int(item["id"]): matches[left][int(item["image_id"])]["gt_matches"].get(int(item["id"]))
                for item in subset
            }
            right_hits = {
                int(item["id"]): matches[right][int(item["image_id"])]["gt_matches"].get(int(item["id"]))
                for item in subset
            }
            transitions[name][str(category)] = classify_gt_transitions(left_hits, right_hits)
    return point, transitions


def bootstrap(ground_truth, predictions, image_ids, repetitions, seed):
    capped = {
        group: {
            image_id: sorted(predictions[group][image_id], key=lambda item: item["score"], reverse=True)[:100]
            for image_id in image_ids
        }
        for group in GROUPS
    }
    reference = {group: coco_ap50_focus(ground_truth, _flatten_for_coco(capped[group], image_ids))
                 for group in GROUPS}
    rng = np.random.default_rng(seed)
    keys = ("ap50", "class_2_ap50", "class_3_ap50")
    samples = {name: {key: [] for key in keys} for name in CONTRASTS}
    for index in range(repetitions):
        draw = rng.choice(image_ids, size=len(image_ids), replace=True)
        sample_gt, sample_arms = remap_bootstrap_sample(ground_truth, capped, draw)
        scored = {group: coco_ap50_focus(sample_gt, sample_arms[group]) for group in GROUPS}
        for name, (left, right) in CONTRASTS.items():
            for key in keys:
                samples[name][key].append(scored[right][key] - scored[left][key])
        if (index + 1) % 10 == 0:
            print(f"bootstrap {index + 1}/{repetitions}", flush=True)
    intervals = {}
    for name, by_key in samples.items():
        intervals[name] = {}
        for key, values in by_key.items():
            array = np.asarray(values, dtype=float)
            if array.size != repetitions or not np.isfinite(array).all():
                raise ValueError(f"incomplete/non-finite bootstrap: {name}/{key}")
            intervals[name][key] = {
                "ci95_percentile": np.quantile(array, [0.025, 0.975]).tolist(),
                "mean": float(array.mean()),
                "positive_fraction": float((array > 0).mean()),
                "valid_repetitions": int(array.size),
            }
    return reference, intervals


def run(args):
    roots = arm_roots(args)
    ground_truth = json.loads(Path(args.ann_file).read_text(encoding="utf-8"))
    image_ids = [int(item["id"]) for item in ground_truth["images"]]
    if len(image_ids) != args.expected_images or len(set(image_ids)) != args.expected_images:
        raise ValueError("validation image count/uniqueness differs")
    by_image = defaultdict(list)
    for annotation in ground_truth["annotations"]:
        by_image[int(annotation["image_id"])].append(annotation)
    results, bootstraps = {}, {}
    for split in ("best", "final"):
        print(f"loading {split}", flush=True)
        official, predictions = load_split(roots, split, image_ids, args.source_ann_file or args.ann_file)
        matched, fixed, positive_coco = fixed_results(
            ground_truth, by_image, predictions, image_ids,
            args.score_threshold, args.iou_threshold,
        )
        point, transitions = comparisons(official, fixed, matched, ground_truth["annotations"])
        reference, intervals = bootstrap(
            ground_truth, predictions, image_ids,
            args.bootstrap_repetitions, args.seed,
        )
        for group in GROUPS:
            if abs(reference[group]["ap50"] - official[group]["metrics"]["ap50"]) > .001:
                raise ValueError(f"{group}/{split} AP50 focus recomputation differs from official")
        results[split] = {
            "official": official, "fixed": fixed, "positive_image_coco": positive_coco,
            "point_deltas": point, "class_gt_transitions": transitions,
        }
        bootstraps[split] = {"reference_ap50_focus": reference, "intervals": intervals}
    payload = {
        "scope": "cmp5L KQ-R/KQ-U versus saved KQ0/KQ2/KQ-O; validation-only best/final EMA",
        "split": {
            "images": len(image_ids),
            "positive_images": sum(bool(by_image[item]) for item in image_ids),
            "empty_images": sum(not by_image[item] for item in image_ids),
        },
        "fixed_rule": {
            "score_threshold": args.score_threshold, "iou_threshold": args.iou_threshold,
            "matching": "score-ordered class-constrained one-to-one",
        },
        "results": results,
        "bootstrap": {
            "method": "paired image resamples with shared draws and COCO AP50 recomputation",
            "repetitions": args.bootstrap_repetitions,
            "seed": args.seed,
            "limitation": "single seed; image-level, not patient-level bootstrap",
            "by_checkpoint": bootstraps,
        },
        "arm_roots": {group: str(path) for group, path in roots.items()},
    }
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out.with_suffix(".COMPLETED").write_text("KQ-R/KQ-U paired validation analysis complete\n")
    print(json.dumps({split: results[split]["point_deltas"] for split in results}, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", required=True)
    parser.add_argument("--extension-root", required=True)
    parser.add_argument("--new-root", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--source-ann-file")
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-images", type=int, default=2975)
    parser.add_argument("--score-threshold", type=float, default=.5)
    parser.add_argument("--iou-threshold", type=float, default=.5)
    parser.add_argument("--bootstrap-repetitions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260925)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
