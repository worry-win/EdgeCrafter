"""Offline stratified metrics, transitions, and paired bootstrap for cmp5L A–F."""

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


CONDITIONS = tuple("ABCDEF")


def _iou_xywh(left, right):
    lx0, ly0, lw, lh = left
    rx0, ry0, rw, rh = right
    lx1, ly1, rx1, ry1 = lx0 + lw, ly0 + lh, rx0 + rw, ry0 + rh
    ix0, iy0 = max(lx0, rx0), max(ly0, ry0)
    ix1, iy1 = min(lx1, rx1), min(ly1, ry1)
    intersection = max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


def fixed_threshold_match(ground_truth, predictions, score_threshold=0.5, iou_threshold=0.5):
    """Greedy score-order, class-constrained, one-to-one image matching."""
    eligible = sorted(
        (prediction for prediction in predictions if prediction["score"] >= score_threshold),
        key=lambda prediction: prediction["score"],
        reverse=True,
    )
    unmatched = {int(annotation["id"]): annotation for annotation in ground_truth}
    matches, false_positives = {}, []
    for prediction in eligible:
        candidates = [
            (annotation_id, _iou_xywh(prediction["bbox"], annotation["bbox"]))
            for annotation_id, annotation in unmatched.items()
            if int(annotation["category_id"]) == int(prediction["category_id"])
        ]
        annotation_id, iou = max(candidates, key=lambda pair: pair[1], default=(None, -1.0))
        if annotation_id is not None and iou >= iou_threshold:
            matches[annotation_id] = int(prediction["query_id"])
            del unmatched[annotation_id]
        else:
            false_positives.append(prediction)
    return {
        "tp": len(matches),
        "fp": len(false_positives),
        "fn": len(unmatched),
        "gt_matches": matches,
        "false_positives": false_positives,
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


def remap_bootstrap_sample(ground_truth, predictions_by_arm, sampled_image_ids):
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
    remapped_predictions = {name: [] for name in predictions_by_arm}
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
        for name, by_image in predictions_by_arm.items():
            for source in by_image[int(old_image_id)]:
                prediction = copy.copy(source)
                prediction["image_id"] = new_image_id
                remapped_predictions[name].append(prediction)
    return remapped_gt, remapped_predictions


def load_predictions(path, expected_ids):
    by_image = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            image_id = int(record["image_id"])
            boxes, scores, labels, query_ids = (
                record["boxes_xyxy"], record["scores"], record["labels"], record["query_ids"]
            )
            if not (len(boxes) == len(scores) == len(labels) == len(query_ids)):
                raise ValueError(f"prediction field length mismatch: {path}, image {image_id}")
            predictions = []
            for box, score, label, query_id in zip(boxes, scores, labels, query_ids):
                x0, y0, x1, y1 = map(float, box)
                values = (x0, y0, x1, y1, float(score))
                if not all(map(math.isfinite, values)):
                    raise ValueError(f"non-finite prediction: {path}, image {image_id}")
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "score": float(score),
                    "query_id": int(query_id),
                })
            by_image[image_id] = predictions
    if set(by_image) != set(expected_ids):
        raise ValueError(f"prediction coverage mismatch for {path}")
    return by_image


def coco_metrics(ground_truth, detections, image_ids=None):
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
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    precision = evaluator.eval["precision"]
    iou50 = int(np.argmin(np.abs(evaluator.params.iouThrs - 0.5)))
    per_class = {}
    for class_index, category_id in enumerate(evaluator.params.catIds):
        curve = precision[iou50, :, class_index, 0, -1]
        per_class[str(category_id)] = float(curve[curve > -1].mean()) if (curve > -1).any() else None
    return {
        "ap": float(evaluator.stats[0]),
        "ap50": float(evaluator.stats[1]),
        "ap75": float(evaluator.stats[2]),
        "ar100": float(evaluator.stats[8]),
        "per_class_ap50": per_class,
    }


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


def _average_precision_binary(target, score):
    target = np.asarray(target, dtype=np.int8)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    ordered = target[order]
    positives = int(target.sum())
    if positives == 0:
        return None
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered == 1].sum() / positives)


def gate_metrics(query_file):
    arrays = np.load(query_file)
    protect = ~arrays["g_gt"].astype(bool)
    predicted_protect = ~arrays["g_pred"].astype(bool)
    p_protect = 1.0 - arrays["p_suppress"].astype(float)
    image_ids = arrays["image_id"].astype(np.int64)
    result = {}
    for name, mask in (
        ("all", np.ones(protect.shape, dtype=bool)),
        ("positive_images", np.isin(image_ids, np.unique(image_ids[~np.isnan(arrays["gt_class_logit"])]))),
    ):
        y, pred, score = protect[mask], predicted_protect[mask], p_protect[mask]
        tp = int((y & pred).sum())
        fp = int((~y & pred).sum())
        fn = int((y & ~pred).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        result[name] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "ap": _average_precision_binary(y, score),
            "protect_count": int(y.sum()),
            "predicted_protect_count": int(pred.sum()),
            "false_suppress_count": int((y & ~pred).sum()),
            "missed_suppress_count": int((~y & pred).sum()),
        }
    return result


def coco_ap50(ground_truth, detections):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = ground_truth
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(detections)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.iouThrs = np.asarray([0.5])
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"][0, :, :, 0, -1]
    valid = precision > -1
    overall = float(precision[valid].mean()) if valid.any() else float("nan")
    category_ids = list(evaluator.params.catIds)
    duct_index = category_ids.index(3)
    duct = precision[:, duct_index]
    return {
        "ap50": overall,
        "duct_ap50": float(duct[duct > -1].mean()) if (duct > -1).any() else float("nan"),
    }


def _flatten_for_coco(predictions, image_ids, max_per_image=100):
    return [
        {key: value for key, value in prediction.items() if key != "query_id"}
        for image_id in image_ids
        for prediction in sorted(
            predictions[image_id], key=lambda item: item["score"], reverse=True
        )[:max_per_image]
    ]


def analyze(args):
    result_dir = Path(args.result_dir)
    ground_truth = json.loads(Path(args.ann_file).read_text(encoding="utf-8"))
    image_ids = [int(image["id"]) for image in ground_truth["images"]]
    annotations_by_image = defaultdict(list)
    for annotation in ground_truth["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    positive_ids = [image_id for image_id in image_ids if annotations_by_image[image_id]]
    empty_ids = [image_id for image_id in image_ids if not annotations_by_image[image_id]]
    image_by_id = {int(image["id"]): image for image in ground_truth["images"]}
    image_manifest = result_dir / "image_manifest.jsonl"
    with image_manifest.open("x", encoding="utf-8") as handle:
        for image_id in image_ids:
            image = image_by_id[image_id]
            handle.write(json.dumps({
                "split": "validation",
                "image_id": image_id,
                "file_name": image.get("file_name"),
                "width": image.get("width"),
                "height": image.get("height"),
                "gt_count": len(annotations_by_image[image_id]),
                "annotations": [
                    {
                        "annotation_id": int(annotation["id"]),
                        "category_id": int(annotation["category_id"]),
                        "bbox_xywh": annotation["bbox"],
                    }
                    for annotation in annotations_by_image[image_id]
                ],
            }, ensure_ascii=False) + "\n")
    predictions = {
        condition: load_predictions(result_dir / f"predictions_{condition}.jsonl", image_ids)
        for condition in CONDITIONS
    }
    coco_by_condition = {
        condition: {
            "full": coco_metrics(
                ground_truth, _flatten_for_coco(predictions[condition], image_ids)
            ),
            "positive_images": coco_metrics(
                ground_truth,
                _flatten_for_coco(predictions[condition], positive_ids),
                positive_ids,
            ),
        }
        for condition in CONDITIONS
    }
    matches = {condition: {} for condition in CONDITIONS}
    fixed = {}
    for condition in CONDITIONS:
        for image_id in image_ids:
            matches[condition][image_id] = fixed_threshold_match(
                annotations_by_image[image_id],
                predictions[condition][image_id],
                args.score_threshold,
                args.iou_threshold,
            )
        fixed[condition] = {
            "full": _fixed_metrics(matches[condition], image_ids),
            "positive_images": _fixed_metrics(matches[condition], positive_ids),
            "empty_images": _fixed_metrics(matches[condition], empty_ids),
        }

    transition_pairs = {
        "B_minus_A": ("A", "B"),
        "C_minus_B": ("B", "C"),
        "D_minus_C": ("C", "D"),
        "E_minus_C": ("C", "E"),
        "F_minus_D": ("D", "F"),
        "F_minus_E": ("E", "F"),
    }
    duct_annotations = [
        annotation for annotation in ground_truth["annotations"]
        if int(annotation["category_id"]) == 3
    ]
    duct_by_condition = {}
    for condition in CONDITIONS:
        duct_by_condition[condition] = {
            int(annotation["id"]): matches[condition][int(annotation["image_id"])]["gt_matches"].get(int(annotation["id"]))
            for annotation in duct_annotations
        }
    duct_transitions = {
        name: classify_gt_transitions(duct_by_condition[left], duct_by_condition[right])
        for name, (left, right) in transition_pairs.items()
    }

    point_deltas = {}
    for name, (left, right) in transition_pairs.items():
        point_deltas[name] = {
            metric: coco_by_condition[right]["full"][metric] - coco_by_condition[left]["full"][metric]
            for metric in ("ap", "ap50", "ap75", "ar100")
        }
        point_deltas[name]["fixed_f1"] = fixed[right]["full"]["f1"] - fixed[left]["full"]["f1"]
        point_deltas[name]["empty_fp_per_image"] = (
            fixed[right]["empty_images"]["fp_per_image"] - fixed[left]["empty_images"]["fp_per_image"]
        )
    point_deltas["interaction_F_minus_D_minus_E_plus_C"] = {
        metric: (
            coco_by_condition["F"]["full"][metric]
            - coco_by_condition["D"]["full"][metric]
            - coco_by_condition["E"]["full"][metric]
            + coco_by_condition["C"]["full"][metric]
        )
        for metric in ("ap", "ap50", "ap75", "ar100")
    }

    representative_pairs = {}
    annotation_to_image = {int(annotation["id"]): int(annotation["image_id"]) for annotation in duct_annotations}
    for name, (left, right) in transition_pairs.items():
        rescued = [
            annotation_to_image[annotation_id]
            for annotation_id in duct_by_condition[left]
            if duct_by_condition[left][annotation_id] is None and duct_by_condition[right][annotation_id] is not None
        ]
        destroyed = [
            annotation_to_image[annotation_id]
            for annotation_id in duct_by_condition[left]
            if duct_by_condition[left][annotation_id] is not None and duct_by_condition[right][annotation_id] is None
        ]
        fp_changes = sorted(
            (
                matches[left][image_id]["fp"] - matches[right][image_id]["fp"],
                image_id,
            )
            for image_id in image_ids
        )
        representative_pairs[name] = {
            "left": left,
            "right": right,
            "duct_rescued_image_ids": rescued[:3],
            "duct_destroyed_image_ids": destroyed[:3],
            "fp_reduced_image_ids": [image_id for change, image_id in reversed(fp_changes) if change > 0][:3],
            "fp_increased_image_ids": [image_id for change, image_id in fp_changes if change < 0][:3],
        }

    coco_predictions = {
        condition: {
            image_id: [
                {key: value for key, value in prediction.items() if key != "query_id"}
                for prediction in sorted(
                    predictions[condition][image_id], key=lambda item: item["score"], reverse=True
                )[:100]
            ]
            for image_id in image_ids
        }
        for condition in CONDITIONS
    }
    bootstrap_contrasts = {**transition_pairs}
    rng = np.random.default_rng(args.seed)
    samples = {
        name: {"ap50": [], "duct_ap50": []}
        for name in [*bootstrap_contrasts, "interaction_F_minus_D_minus_E_plus_C"]
    }
    for repetition in range(args.bootstrap_repetitions):
        draw = rng.choice(image_ids, size=len(image_ids), replace=True)
        resampled_gt, arms = remap_bootstrap_sample(ground_truth, coco_predictions, draw)
        scored = {condition: coco_ap50(resampled_gt, detections) for condition, detections in arms.items()}
        for name, (left, right) in bootstrap_contrasts.items():
            for metric in ("ap50", "duct_ap50"):
                samples[name][metric].append(scored[right][metric] - scored[left][metric])
        for metric in ("ap50", "duct_ap50"):
            samples["interaction_F_minus_D_minus_E_plus_C"][metric].append(
                scored["F"][metric] - scored["D"][metric] - scored["E"][metric] + scored["C"][metric]
            )
        if (repetition + 1) % 10 == 0:
            print(f"bootstrap {repetition + 1}/{args.bootstrap_repetitions}", flush=True)
    intervals = {}
    for name, metrics in samples.items():
        intervals[name] = {}
        for metric, values in metrics.items():
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            intervals[name][metric] = {
                "ci95_percentile": np.quantile(values, [0.025, 0.975]).tolist(),
                "positive_fraction": float((values > 0).mean()),
                "valid_repetitions": int(values.size),
            }

    payload = {
        "scope": "cmp5L frozen-original-EMA A-F validation only",
        "split": {
            "images": len(image_ids),
            "positive_images": len(positive_ids),
            "empty_images": len(empty_ids),
        },
        "fixed_rule": {"score_threshold": args.score_threshold, "iou_threshold": args.iou_threshold},
        "coco_metrics": coco_by_condition,
        "fixed_metrics": fixed,
        "gate_metrics": gate_metrics(result_dir / "query_diagnostics.npz"),
        "pairwise_point_deltas": point_deltas,
        "duct_transitions": duct_transitions,
        "bootstrap": {
            "method": "paired image resampling with shared draws across A-F; COCO AP50 recomputed",
            "repetitions": args.bootstrap_repetitions,
            "seed": args.seed,
            "limitation": "patient IDs unavailable for most images; image-level intervals may be optimistic; one model seed",
            "intervals": intervals,
        },
        "representatives": representative_pairs,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pairwise": point_deltas, "duct": duct_transitions}, ensure_ascii=False, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-repetitions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
