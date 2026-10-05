"""Paired validation analysis for cmp5L KQ0--KQ3 shared-query KD.

This is offline analysis of already-saved validation predictions.  It never
loads the test split and uses the same image draw for every arm in each
bootstrap repetition.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


GROUPS = ("KQ0", "KQ1", "KQ2", "KQ3")
CONTRASTS = {
    "KQ1_minus_KQ0": ("KQ0", "KQ1"),
    "KQ2_minus_KQ0": ("KQ0", "KQ2"),
    "KQ2_minus_KQ1": ("KQ1", "KQ2"),
    "KQ3_minus_KQ2": ("KQ2", "KQ3"),
}
FOCUS_CLASSES = (2, 3)


def _iou_xywh(left, right):
    lx0, ly0, lw, lh = map(float, left)
    rx0, ry0, rw, rh = map(float, right)
    lx1, ly1, rx1, ry1 = lx0 + lw, ly0 + lh, rx0 + rw, ry0 + rh
    ix0, iy0 = max(lx0, rx0), max(ly0, ry0)
    ix1, iy1 = min(lx1, rx1), min(ly1, ry1)
    intersection = max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


def fixed_threshold_match(ground_truth, predictions, score_threshold=0.5, iou_threshold=0.5):
    """Greedy score-order, class-constrained, one-to-one image matching."""
    eligible = sorted(
        (item for item in predictions if item["score"] >= score_threshold),
        key=lambda item: item["score"],
        reverse=True,
    )
    unmatched = {int(item["id"]): item for item in ground_truth}
    matches = {}
    false_positives = []
    for prediction in eligible:
        candidates = [
            (annotation_id, _iou_xywh(prediction["bbox"], annotation["bbox"]))
            for annotation_id, annotation in unmatched.items()
            if int(annotation["category_id"]) == int(prediction["category_id"])
        ]
        annotation_id, iou = max(candidates, key=lambda pair: pair[1], default=(None, -1.0))
        if annotation_id is not None and iou >= iou_threshold:
            matches[annotation_id] = True
            del unmatched[annotation_id]
        else:
            false_positives.append(prediction)
    return {
        "tp": len(matches),
        "fp": len(false_positives),
        "fn": len(unmatched),
        "gt_matches": matches,
    }


def classify_gt_transitions(left, right):
    result = {"stable_detected": 0, "rescued": 0, "destroyed": 0, "stable_missed": 0}
    for annotation_id in sorted(set(left) | set(right)):
        left_hit = left.get(annotation_id) is not None
        right_hit = right.get(annotation_id) is not None
        if left_hit and right_hit:
            result["stable_detected"] += 1
        elif not left_hit and right_hit:
            result["rescued"] += 1
        elif left_hit and not right_hit:
            result["destroyed"] += 1
        else:
            result["stable_missed"] += 1
    return result


def load_predictions(path, expected_ids):
    by_image = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            image_id = int(record["image_id"])
            boxes, scores, labels = record["boxes_xyxy"], record["scores"], record["labels"]
            if not (len(boxes) == len(scores) == len(labels)):
                raise ValueError(f"prediction field length mismatch: {path}, image {image_id}")
            predictions = []
            for box, score, label in zip(boxes, scores, labels):
                x0, y0, x1, y1 = map(float, box)
                score = float(score)
                if not all(math.isfinite(value) for value in (x0, y0, x1, y1, score)):
                    raise ValueError(f"non-finite prediction: {path}, image {image_id}")
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "score": score,
                })
            if image_id in by_image:
                raise ValueError(f"duplicate image {image_id}: {path}")
            by_image[image_id] = predictions
    if set(by_image) != set(map(int, expected_ids)):
        raise ValueError(f"prediction coverage mismatch for {path}")
    return by_image


def _flatten_for_coco(predictions, image_ids, max_per_image=100):
    return [
        prediction
        for image_id in image_ids
        for prediction in sorted(
            predictions[int(image_id)], key=lambda item: item["score"], reverse=True
        )[:max_per_image]
    ]


def _fixed_metrics(matches, image_ids):
    tp = sum(matches[image_id]["tp"] for image_id in image_ids)
    fp = sum(matches[image_id]["fp"] for image_id in image_ids)
    fn = sum(matches[image_id]["fn"] for image_id in image_ids)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fp_per_image": fp / len(image_ids) if image_ids else None,
        "n_images": len(image_ids),
    }


def _coco_eval(ground_truth, detections, image_ids=None, iou_thresholds=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = ground_truth
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(detections)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        if image_ids is not None:
            evaluator.params.imgIds = list(map(int, image_ids))
        if iou_thresholds is not None:
            evaluator.params.iouThrs = np.asarray(iou_thresholds, dtype=float)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return evaluator


def coco_metrics(ground_truth, detections, image_ids=None):
    evaluator = _coco_eval(ground_truth, detections, image_ids)
    return {
        "ap": float(evaluator.stats[0]),
        "ap50": float(evaluator.stats[1]),
        "ap75": float(evaluator.stats[2]),
        "ar100": float(evaluator.stats[8]),
    }


def coco_ap50_focus(ground_truth, detections):
    evaluator = _coco_eval(ground_truth, detections, iou_thresholds=[0.5])
    precision = evaluator.eval["precision"][0, :, :, 0, -1]
    valid = precision > -1
    result = {"ap50": float(precision[valid].mean()) if valid.any() else float("nan")}
    category_ids = list(map(int, evaluator.params.catIds))
    for category_id in FOCUS_CLASSES:
        key = f"class_{category_id}_ap50"
        if category_id not in category_ids:
            result[key] = float("nan")
            continue
        curve = precision[:, category_ids.index(category_id)]
        result[key] = float(curve[curve > -1].mean()) if (curve > -1).any() else float("nan")
    return result


def remap_bootstrap_sample(ground_truth, predictions_by_group, sampled_image_ids):
    image_by_id = {int(image["id"]): image for image in ground_truth["images"]}
    annotations_by_image = defaultdict(list)
    for annotation in ground_truth["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    remapped_gt = {
        "info": ground_truth.get("info", {}),
        "licenses": ground_truth.get("licenses", []),
        "categories": ground_truth["categories"],
        "images": [],
        "annotations": [],
    }
    remapped_predictions = {name: [] for name in predictions_by_group}
    annotation_id = 1
    for new_image_id, old_image_id in enumerate(sampled_image_ids, start=1):
        image = copy.copy(image_by_id[int(old_image_id)])
        image["id"] = new_image_id
        remapped_gt["images"].append(image)
        for source in annotations_by_image[int(old_image_id)]:
            annotation = copy.copy(source)
            annotation["id"] = annotation_id
            annotation["image_id"] = new_image_id
            remapped_gt["annotations"].append(annotation)
            annotation_id += 1
        for group, by_image in predictions_by_group.items():
            for source in by_image[int(old_image_id)]:
                prediction = copy.copy(source)
                prediction["image_id"] = new_image_id
                remapped_predictions[group].append(prediction)
    return remapped_gt, remapped_predictions


def _prediction_file(result_root, group, split):
    metrics_path = result_root / group / f"validation_{split}" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    prediction_path = Path(metrics["meta"]["predictions_file"])
    if not prediction_path.is_file():
        prediction_path = metrics_path.with_name("metrics.predictions.jsonl")
    return metrics, prediction_path


def _analyze_split(result_root, ground_truth, annotations_by_image, image_ids, split, score_threshold, iou_threshold):
    positive_ids = [image_id for image_id in image_ids if annotations_by_image[image_id]]
    empty_ids = [image_id for image_id in image_ids if not annotations_by_image[image_id]]
    predictions = {}
    official = {}
    for group in GROUPS:
        metrics, path = _prediction_file(result_root, group, split)
        if int(metrics["meta"]["n_images"]) != len(image_ids):
            raise ValueError(f"{group}/{split} expected {len(image_ids)} images")
        official[group] = metrics
        predictions[group] = load_predictions(path, image_ids)
    matches = {group: {} for group in GROUPS}
    fixed = {}
    positive_coco = {}
    for group in GROUPS:
        for image_id in image_ids:
            matches[group][image_id] = fixed_threshold_match(
                annotations_by_image[image_id], predictions[group][image_id],
                score_threshold, iou_threshold,
            )
        fixed[group] = {
            "full": _fixed_metrics(matches[group], image_ids),
            "positive_images": _fixed_metrics(matches[group], positive_ids),
            "empty_images": _fixed_metrics(matches[group], empty_ids),
        }
        positive_coco[group] = coco_metrics(
            ground_truth, _flatten_for_coco(predictions[group], positive_ids), positive_ids
        )
    point_deltas = {}
    transitions = {}
    representatives = {}
    annotations_by_class = {
        category_id: [
            item for item in ground_truth["annotations"]
            if int(item["category_id"]) == category_id
        ] for category_id in FOCUS_CLASSES
    }
    for name, (left, right) in CONTRASTS.items():
        left_metrics = official[left]["metrics"]
        right_metrics = official[right]["metrics"]
        point_deltas[name] = {
            metric: float(right_metrics[metric] - left_metrics[metric])
            for metric in ("ap", "ap50", "ap75", "ar100", "precision", "recall", "f1")
        }
        point_deltas[name]["duct_ap50"] = float(
            official[right]["per_class_ap50"]["3"]["ap50"]
            - official[left]["per_class_ap50"]["3"]["ap50"]
        )
        point_deltas[name]["lymph_ap50"] = float(
            official[right]["per_class_ap50"]["2"]["ap50"]
            - official[left]["per_class_ap50"]["2"]["ap50"]
        )
        point_deltas[name]["fixed_f1"] = fixed[right]["full"]["f1"] - fixed[left]["full"]["f1"]
        point_deltas[name]["empty_fp_per_image"] = (
            fixed[right]["empty_images"]["fp_per_image"]
            - fixed[left]["empty_images"]["fp_per_image"]
        )
        transitions[name] = {}
        for category_id, annotations in annotations_by_class.items():
            left_map = {
                int(item["id"]): matches[left][int(item["image_id"])]["gt_matches"].get(int(item["id"]))
                for item in annotations
            }
            right_map = {
                int(item["id"]): matches[right][int(item["image_id"])]["gt_matches"].get(int(item["id"]))
                for item in annotations
            }
            transitions[name][str(category_id)] = classify_gt_transitions(left_map, right_map)
        fp_change = sorted(
            (matches[left][image_id]["fp"] - matches[right][image_id]["fp"], image_id)
            for image_id in image_ids
        )
        representatives[name] = {
            "fp_reduced_image_ids": [image_id for change, image_id in reversed(fp_change) if change > 0][:5],
            "fp_increased_image_ids": [image_id for change, image_id in fp_change if change < 0][:5],
        }
    return {
        "official": official,
        "positive_image_coco": positive_coco,
        "fixed": fixed,
        "point_deltas": point_deltas,
        "focus_class_transitions": transitions,
        "representatives": representatives,
    }, predictions


def analyze(args):
    result_root = Path(args.result_root)
    ground_truth = json.loads(Path(args.ann_file).read_text(encoding="utf-8"))
    image_ids = [int(item["id"]) for item in ground_truth["images"]]
    annotations_by_image = defaultdict(list)
    for annotation in ground_truth["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    if len(image_ids) != args.expected_images:
        raise ValueError(f"validation has {len(image_ids)} images, expected {args.expected_images}")
    split_results = {}
    best_predictions = None
    for split in ("best", "final"):
        split_results[split], predictions = _analyze_split(
            result_root, ground_truth, annotations_by_image, image_ids, split,
            args.score_threshold, args.iou_threshold,
        )
        if split == "best":
            best_predictions = predictions

    coco_predictions = {
        group: {
            image_id: sorted(best_predictions[group][image_id], key=lambda item: item["score"], reverse=True)[:100]
            for image_id in image_ids
        } for group in GROUPS
    }
    rng = np.random.default_rng(args.seed)
    metrics = ("ap50", "class_2_ap50", "class_3_ap50")
    samples = {name: {metric: [] for metric in metrics} for name in CONTRASTS}
    for repetition in range(args.bootstrap_repetitions):
        draw = rng.choice(image_ids, size=len(image_ids), replace=True)
        sample_gt, arms = remap_bootstrap_sample(ground_truth, coco_predictions, draw)
        scored = {group: coco_ap50_focus(sample_gt, arm) for group, arm in arms.items()}
        for name, (left, right) in CONTRASTS.items():
            for metric in metrics:
                samples[name][metric].append(scored[right][metric] - scored[left][metric])
        if (repetition + 1) % 10 == 0:
            print(f"bootstrap {repetition + 1}/{args.bootstrap_repetitions}", flush=True)
    intervals = {}
    for name, by_metric in samples.items():
        intervals[name] = {}
        for metric, values in by_metric.items():
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            intervals[name][metric] = {
                "ci95_percentile": np.quantile(values, [0.025, 0.975]).tolist(),
                "mean": float(values.mean()),
                "positive_fraction": float((values > 0).mean()),
                "valid_repetitions": int(values.size),
            }

    payload = {
        "scope": "cmp5L KQ0-KQ3 train/validation only; best/final EMA",
        "split": {
            "images": len(image_ids),
            "positive_images": sum(bool(annotations_by_image[item]) for item in image_ids),
            "empty_images": sum(not annotations_by_image[item] for item in image_ids),
        },
        "fixed_rule": {
            "score_threshold": args.score_threshold,
            "iou_threshold": args.iou_threshold,
            "matching": "score-ordered class-constrained one-to-one",
        },
        "results": split_results,
        "bootstrap": {
            "checkpoint": "best EMA",
            "method": "paired image resampling with shared draws across KQ0-KQ3; COCO AP50 recomputed",
            "repetitions": args.bootstrap_repetitions,
            "seed": args.seed,
            "limitation": "single seed and image-level rather than patient-level resampling",
            "intervals": intervals,
        },
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "best_point_deltas": split_results["best"]["point_deltas"],
        "bootstrap": intervals,
    }, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-images", type=int, default=2975)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-repetitions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260921)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
