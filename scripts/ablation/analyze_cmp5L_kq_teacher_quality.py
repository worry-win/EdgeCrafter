"""Offline paired analysis for baseline/KQ0/KQ2 shared-q0 teacher quality."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.ablation.analyze_cmp5L_shared_query_kd import (
    _fixed_metrics,
    _flatten_for_coco,
    classify_gt_transitions,
    coco_metrics,
    fixed_threshold_match,
    load_predictions,
    remap_bootstrap_sample,
)


STATES = ("baseline", "KQ0", "KQ2")
ARMS = ("student", "normal_teacher", "privileged_teacher")
CONTRASTS = {
    "privileged_minus_normal": ("normal_teacher", "privileged_teacher"),
    "privileged_minus_student": ("student", "privileged_teacher"),
    "normal_minus_student": ("student", "normal_teacher"),
}

_BOOTSTRAP_CONTEXT = {}


def _arm_key(state, arm):
    return f"{state}:{arm}"


def _load_results(root, ground_truth):
    image_ids = [int(item["id"]) for item in ground_truth["images"]]
    payloads, predictions, truncation_audit = {}, {}, {}
    for state in STATES:
        directory = root / f"full_{state}_v1"
        payload = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        if int(payload["images"]) != len(image_ids):
            raise ValueError(f"{state} image coverage mismatch")
        payloads[state] = payload
        for arm in ARMS:
            key = _arm_key(state, arm)
            loaded = load_predictions(
                directory / f"{arm}.predictions.jsonl", image_ids
            )
            maximum_eligible = max(
                sum(item["score"] >= 0.5 for item in loaded[image_id])
                for image_id in image_ids
            )
            if maximum_eligible > 100:
                raise ValueError(f"{key} has {maximum_eligible} score>=0.5 candidates in one image")
            predictions[key] = {
                image_id: sorted(loaded[image_id], key=lambda item: item["score"], reverse=True)[:100]
                for image_id in image_ids
            }
            truncation_audit[key] = {
                "source_candidates_per_image": max(len(items) for items in loaded.values()),
                "retained_max_dets": 100,
                "max_score_ge_0_5_per_image": maximum_eligible,
                "fixed_threshold_equivalent": True,
                "coco_equivalent": "COCO maxDets=100",
            }
    return payloads, predictions, image_ids, truncation_audit


def _point_analysis(ground_truth, predictions, image_ids, score_threshold, iou_threshold):
    annotations = defaultdict(list)
    for item in ground_truth["annotations"]:
        annotations[int(item["image_id"])].append(item)
    positive = [item for item in image_ids if annotations[item]]
    empty = [item for item in image_ids if not annotations[item]]
    matches, fixed, coco = {}, {}, {}
    for key, by_image in predictions.items():
        matches[key] = {
            image_id: fixed_threshold_match(
                annotations[image_id], by_image[image_id], score_threshold, iou_threshold
            ) for image_id in image_ids
        }
        fixed[key] = {
            "full": _fixed_metrics(matches[key], image_ids),
            "positive_images": _fixed_metrics(matches[key], positive),
            "empty_images": _fixed_metrics(matches[key], empty),
        }
        coco[key] = coco_metrics(
            ground_truth, _flatten_for_coco(by_image, image_ids), image_ids
        )
    deltas, transitions = {}, {}
    for state in STATES:
        deltas[state], transitions[state] = {}, {}
        for name, (left_arm, right_arm) in CONTRASTS.items():
            left, right = _arm_key(state, left_arm), _arm_key(state, right_arm)
            deltas[state][name] = {
                metric: coco[right][metric] - coco[left][metric]
                for metric in ("ap", "ap50", "ap75", "ar100")
            }
            for metric in ("precision", "recall", "f1", "fp_per_image"):
                deltas[state][name][f"fixed_{metric}"] = (
                    fixed[right]["full"][metric] - fixed[left]["full"][metric]
                )
            deltas[state][name]["empty_fp_per_image"] = (
                fixed[right]["empty_images"]["fp_per_image"]
                - fixed[left]["empty_images"]["fp_per_image"]
            )
            transitions[state][name] = {}
            for class_id in range(4):
                class_annotations = [
                    item for item in ground_truth["annotations"]
                    if int(item["category_id"]) == class_id
                ]
                left_map = {
                    int(item["id"]): matches[left][int(item["image_id"])] ["gt_matches"].get(int(item["id"]))
                    for item in class_annotations
                }
                right_map = {
                    int(item["id"]): matches[right][int(item["image_id"])] ["gt_matches"].get(int(item["id"]))
                    for item in class_annotations
                }
                transitions[state][name][str(class_id)] = classify_gt_transitions(left_map, right_map)
    return {
        "split": {"images": len(image_ids), "positive_images": len(positive), "empty_images": len(empty)},
        "coco_recomputed": coco,
        "fixed": fixed,
        "deltas": deltas,
        "class_transitions": transitions,
    }


def _bootstrap_one(task):
    repetition, seed = task
    ground_truth = _BOOTSTRAP_CONTEXT["ground_truth"]
    predictions = _BOOTSTRAP_CONTEXT["predictions"]
    image_ids = _BOOTSTRAP_CONTEXT["image_ids"]
    score_threshold = _BOOTSTRAP_CONTEXT["score_threshold"]
    iou_threshold = _BOOTSTRAP_CONTEXT["iou_threshold"]
    rng = np.random.default_rng(np.random.SeedSequence([seed, repetition]))
    draw = rng.choice(image_ids, size=len(image_ids), replace=True)
    selected = {_arm_key(state, arm): predictions[_arm_key(state, arm)] for state in STATES for arm in ARMS}
    sample_gt, remapped = remap_bootstrap_sample(ground_truth, selected, draw)
    new_ids = [int(item["id"]) for item in sample_gt["images"]]
    annotations = defaultdict(list)
    for item in sample_gt["annotations"]:
        annotations[int(item["image_id"])].append(item)
    empty_ids = [item for item in new_ids if not annotations[item]]
    scored, fixed = {}, {}
    for key, detections in remapped.items():
        scored[key] = coco_metrics(sample_gt, detections, new_ids)
        by_image = defaultdict(list)
        for item in detections:
            by_image[int(item["image_id"])].append(item)
        matched = {
            image_id: fixed_threshold_match(
                annotations[image_id], by_image[image_id], score_threshold, iou_threshold
            ) for image_id in new_ids
        }
        fixed[key] = {
            "full": _fixed_metrics(matched, new_ids),
            "empty": _fixed_metrics(matched, empty_ids),
        }
    result = {state: {} for state in STATES}
    for state in STATES:
        for contrast, (left_arm, right_arm) in CONTRASTS.items():
            left, right = _arm_key(state, left_arm), _arm_key(state, right_arm)
            row = {
                metric: scored[right][metric] - scored[left][metric]
                for metric in ("ap", "ap50", "ap75", "ar100")
            }
            for metric in ("precision", "recall", "f1"):
                row[f"fixed_{metric}"] = fixed[right]["full"][metric] - fixed[left]["full"][metric]
            row["empty_fp_per_image"] = (
                fixed[right]["empty"]["fp_per_image"] - fixed[left]["empty"]["fp_per_image"]
            )
            result[state][contrast] = row
    return repetition, result


def _bootstrap(ground_truth, predictions, image_ids, repetitions, seed, score_threshold, iou_threshold, workers):
    metric_names = ("ap", "ap50", "ap75", "ar100", "fixed_precision", "fixed_recall", "fixed_f1", "empty_fp_per_image")
    samples = {
        state: {contrast: {metric: [] for metric in metric_names} for contrast in CONTRASTS}
        for state in STATES
    }
    _BOOTSTRAP_CONTEXT.update({
        "ground_truth": ground_truth,
        "predictions": predictions,
        "image_ids": image_ids,
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
    })
    context = mp.get_context("fork")
    completed = 0
    with context.Pool(processes=workers) as pool:
        for repetition, result in pool.imap_unordered(
            _bootstrap_one, [(index, seed) for index in range(repetitions)], chunksize=1
        ):
            del repetition
            for state, contrasts in result.items():
                for contrast, metrics in contrasts.items():
                    for metric, value in metrics.items():
                        samples[state][contrast][metric].append(value)
            completed += 1
            if completed % 10 == 0:
                print(f"bootstrap {completed}/{repetitions}", flush=True)
    intervals = {}
    for state, contrasts in samples.items():
        intervals[state] = {}
        for contrast, metrics in contrasts.items():
            intervals[state][contrast] = {}
            for metric, values in metrics.items():
                array = np.asarray(values, dtype=float)
                array = array[np.isfinite(array)]
                intervals[state][contrast][metric] = {
                    "mean": float(array.mean()),
                    "ci95_percentile": np.quantile(array, [0.025, 0.975]).tolist(),
                    "positive_fraction": float((array > 0).mean()),
                    "valid_repetitions": int(array.size),
                }
    return intervals


def run(args):
    root = Path(args.result_root)
    ground_truth = json.loads(Path(args.ann_file).read_text(encoding="utf-8"))
    payloads, predictions, image_ids, truncation_audit = _load_results(root, ground_truth)
    point = _point_analysis(
        ground_truth, predictions, image_ids, args.score_threshold, args.iou_threshold
    )
    intervals = _bootstrap(
        ground_truth, predictions, image_ids, args.bootstrap_repetitions, args.seed,
        args.score_threshold, args.iou_threshold, args.workers,
    )
    result = {
        "scope": "validation-only shared-q0/r0 KQ teacher quality; no test",
        "fixed_rule": {
            "score_threshold": args.score_threshold,
            "iou_threshold": args.iou_threshold,
            "matching": "score-ordered class-constrained one-to-one",
        },
        "prediction_truncation_audit": truncation_audit,
        "official": {state: payloads[state]["metrics"] for state in STATES},
        "point": point,
        "bootstrap": {
            "method": "paired image resampling; shared draw across all states/arms; fork workers",
            "repetitions": args.bootstrap_repetitions,
            "seed": args.seed,
            "workers": args.workers,
            "intervals": intervals,
        },
    }
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=8)
    run(parser.parse_args())
