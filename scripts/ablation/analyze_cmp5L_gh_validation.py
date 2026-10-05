"""Offline paired analysis for cmp5L G/H validation-only experiments."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.ablation.analyze_cmp5L_af_validation import (
    _average_precision_binary,
    _fixed_metrics,
    _flatten_for_coco,
    _iou_xywh,
    coco_ap50,
    coco_metrics,
    fixed_threshold_match,
    load_predictions,
    remap_bootstrap_sample,
)


G_CONDITIONS = ("A", "G0", "G1", "G2", "G3")
H_CONDITIONS = ("A", "H005", "H020", "H050", "H080", "H095")
H_THRESHOLDS = {
    "H005": 0.05,
    "H020": 0.20,
    "H050": 0.50,
    "H080": 0.80,
    "H095": 0.95,
}


def select_conservative_threshold(baseline, candidates):
    """Apply the pre-locked H calibration rule without relaxing any floor."""
    required = ("ap50", "fixed_f1", "duct_ap50")
    eligible = [
        name
        for name, metrics in candidates.items()
        if all(float(metrics[key]) >= float(baseline[key]) for key in required)
    ]
    eligible.sort(key=lambda name: float(candidates[name]["threshold"]))
    if not eligible:
        return {
            "selected": None,
            "eligible": [],
            "reason": "no pre-locked threshold passed all floors",
        }
    selected = max(
        eligible,
        key=lambda name: (
            float(candidates[name]["ap50"]),
            float(candidates[name]["threshold"]),
        ),
    )
    return {
        "selected": selected,
        "eligible": eligible,
        "reason": "highest AP50; ties choose higher threshold",
    }


def query_transition_summary(left, right):
    """Count GT transitions and distinguish stable-query from reassignment."""
    result = {
        "stable_same_query": 0,
        "stable_reassigned": 0,
        "rescued": 0,
        "destroyed": 0,
        "stable_missed": 0,
    }
    for annotation_id in sorted(set(left) | set(right)):
        left_query = left.get(annotation_id)
        right_query = right.get(annotation_id)
        if left_query is not None and right_query is not None:
            key = (
                "stable_same_query"
                if int(left_query) == int(right_query)
                else "stable_reassigned"
            )
        elif left_query is None and right_query is not None:
            key = "rescued"
        elif left_query is not None and right_query is None:
            key = "destroyed"
        else:
            key = "stable_missed"
        result[key] += 1
    return result


def build_h_evaluation_contrasts(selected):
    """Keep the historical H005/A confirmation even if calibration selects none."""
    contrasts = {"H005_minus_A": ("A", "H005")}
    if selected and selected not in ("A", "H005"):
        contrasts = {
            f"{selected}_minus_A": ("A", selected),
            f"{selected}_minus_H005": ("H005", selected),
            **contrasts,
        }
    return contrasts


def _load_split(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    calibration = [int(item["image_id"]) for item in payload["calibration"]]
    evaluation = [int(item["image_id"]) for item in payload["evaluation"]]
    if set(calibration) & set(evaluation):
        raise ValueError("locked calibration/evaluation split overlaps")
    return payload, calibration, evaluation


def _ground_truth_index(ground_truth):
    image_ids = [int(image["id"]) for image in ground_truth["images"]]
    image_by_id = {int(image["id"]): image for image in ground_truth["images"]}
    annotations_by_image = defaultdict(list)
    annotation_by_id = {}
    for annotation in ground_truth["annotations"]:
        image_id = int(annotation["image_id"])
        annotations_by_image[image_id].append(annotation)
        annotation_by_id[int(annotation["id"])] = annotation
    positive = [image_id for image_id in image_ids if annotations_by_image[image_id]]
    empty = [image_id for image_id in image_ids if not annotations_by_image[image_id]]
    duct = sorted(
        {
            int(annotation["image_id"])
            for annotation in ground_truth["annotations"]
            if int(annotation["category_id"]) == 3
        }
    )
    return {
        "image_ids": image_ids,
        "image_by_id": image_by_id,
        "annotations_by_image": annotations_by_image,
        "annotation_by_id": annotation_by_id,
        "positive_ids": positive,
        "empty_ids": empty,
        "duct_ids": duct,
    }


def _matches(conditions, predictions, annotations_by_image, image_ids, score, iou):
    matches = {condition: {} for condition in conditions}
    for condition in conditions:
        for image_id in image_ids:
            matches[condition][image_id] = fixed_threshold_match(
                annotations_by_image[image_id],
                predictions[condition][image_id],
                score_threshold=score,
                iou_threshold=iou,
            )
    return matches


def _fixed_by_subset(matches, subsets):
    return {
        condition: {
            subset: _fixed_metrics(by_image, image_ids)
            for subset, image_ids in subsets.items()
        }
        for condition, by_image in matches.items()
    }


def _coco_by_subset(ground_truth, conditions, predictions, subsets):
    return {
        condition: {
            subset: coco_metrics(
                ground_truth,
                _flatten_for_coco(predictions[condition], image_ids),
                None if subset == "full" else image_ids,
            )
            for subset, image_ids in subsets.items()
        }
        for condition in conditions
    }


def _gt_query_map(matches, condition, annotation_ids):
    by_image = matches[condition]
    return {
        annotation_id: by_image[image_id]["gt_matches"].get(annotation_id)
        for annotation_id, image_id in annotation_ids.items()
    }


def _transition_details(matches, left, right, annotation_by_id, annotation_filter=None):
    annotations = [
        annotation
        for annotation in annotation_by_id.values()
        if annotation_filter is None or annotation_filter(annotation)
    ]
    annotation_to_image = {
        int(annotation["id"]): int(annotation["image_id"])
        for annotation in annotations
    }
    left_map = _gt_query_map(matches, left, annotation_to_image)
    right_map = _gt_query_map(matches, right, annotation_to_image)
    summary = query_transition_summary(left_map, right_map)
    summary["rescued_image_ids"] = [
        annotation_to_image[annotation_id]
        for annotation_id in annotation_to_image
        if left_map[annotation_id] is None and right_map[annotation_id] is not None
    ][:20]
    summary["destroyed_image_ids"] = [
        annotation_to_image[annotation_id]
        for annotation_id in annotation_to_image
        if left_map[annotation_id] is not None and right_map[annotation_id] is None
    ][:20]
    return summary


def _prediction_rank_arrays(predictions, condition, image_ids, queries):
    rank = np.full((len(image_ids), queries), -1, dtype=np.int16)
    score = np.full((len(image_ids), queries), np.nan, dtype=np.float32)
    label = np.full((len(image_ids), queries), -1, dtype=np.int16)
    for row, image_id in enumerate(image_ids):
        for position, prediction in enumerate(predictions[condition][image_id], start=1):
            query = int(prediction["query_id"])
            if rank[row, query] < 0:
                rank[row, query] = position
                score[row, query] = float(prediction["score"])
                label[row, query] = int(prediction["category_id"])
    return rank, score, label


def _query_importance(result_dir, query_out, conditions, predictions, matches, index):
    query = np.load(result_dir / "query_diagnostics.npz")
    raw = np.load(result_dir / "raw_query_outputs.npz")
    image_ids = raw["image_ids"].astype(np.int64)
    if not np.array_equal(image_ids, query["image_id"][:, 0].astype(np.int64)):
        raise ValueError("query diagnostics and raw output image order differ")
    boxes = raw["A_boxes"].astype(np.float32)
    top1_class = query["top1_class"].astype(np.int16)
    rows, queries = boxes.shape[:2]
    best_annotation_id = np.full((rows, queries), -1, dtype=np.int64)
    best_annotation_category = np.full((rows, queries), -1, dtype=np.int16)
    class_consistent_iou = np.zeros((rows, queries), dtype=np.float32)
    agnostic_iou = np.zeros((rows, queries), dtype=np.float32)
    for row, image_id in enumerate(image_ids):
        image = index["image_by_id"][int(image_id)]
        width, height = float(image["width"]), float(image["height"])
        annotations = index["annotations_by_image"][int(image_id)]
        for query_index, (cx, cy, w, h) in enumerate(boxes[row]):
            predicted = [
                (float(cx) - float(w) / 2.0) * width,
                (float(cy) - float(h) / 2.0) * height,
                float(w) * width,
                float(h) * height,
            ]
            overlaps = [_iou_xywh(predicted, ann["bbox"]) for ann in annotations]
            if overlaps:
                best = int(np.argmax(overlaps))
                annotation = annotations[best]
                best_annotation_id[row, query_index] = int(annotation["id"])
                best_annotation_category[row, query_index] = int(annotation["category_id"])
                agnostic_iou[row, query_index] = float(overlaps[best])
                consistent = [
                    overlap
                    for overlap, candidate in zip(overlaps, annotations)
                    if int(candidate["category_id"]) == int(top1_class[row, query_index])
                ]
                class_consistent_iou[row, query_index] = max(consistent, default=0.0)
    payload = {key: query[key] for key in query.files}
    payload.update(
        {
            "best_annotation_id": best_annotation_id,
            "best_annotation_category": best_annotation_category,
            "class_agnostic_iou_recomputed": agnostic_iou,
            "class_consistent_iou": class_consistent_iou,
        }
    )
    for condition in conditions:
        rank, final_score, final_label = _prediction_rank_arrays(
            predictions, condition, image_ids, queries
        )
        actual_tp = np.zeros((rows, queries), dtype=np.uint8)
        for row, image_id in enumerate(image_ids):
            for query_id in matches[condition][int(image_id)]["gt_matches"].values():
                actual_tp[row, int(query_id)] = 1
        payload[f"{condition}_final_rank"] = rank
        payload[f"{condition}_final_score"] = final_score
        payload[f"{condition}_final_label"] = final_label
        payload[f"{condition}_actual_tp"] = actual_tp
    np.savez_compressed(query_out, **payload)
    return payload


def _query_error_summary(query_payload, conditions):
    positive = query_payload["best_annotation_id"] >= 0
    fs = positive & query_payload["false_suppress"].astype(bool)
    ms = positive & query_payload["missed_suppress"].astype(bool)
    high_iou = positive & (query_payload["class_agnostic_iou_recomputed"] >= 0.5)
    class_iou = positive & (query_payload["class_consistent_iou"] >= 0.5)
    duct_related = positive & (query_payload["best_annotation_category"] == 3)
    result = {
        "all_positive_queries": int(positive.sum()),
        "high_iou_queries": int(high_iou.sum()),
        "class_consistent_high_iou_queries": int(class_iou.sum()),
        "duct_related_queries": int(duct_related.sum()),
        "false_suppress_total": int(fs.sum()),
        "missed_suppress_total": int(ms.sum()),
        "by_condition": {},
    }
    for condition in conditions:
        tp = query_payload[f"{condition}_actual_tp"].astype(bool)
        result["by_condition"][condition] = {
            "actual_tp_queries": int(tp.sum()),
            "false_suppress_actual_tp_queries": int((fs & tp).sum()),
            "missed_suppress_actual_tp_queries": int((ms & tp).sum()),
            "high_iou_actual_tp_queries": int((high_iou & tp).sum()),
            "duct_related_actual_tp_queries": int((duct_related & tp).sum()),
        }
    return result


def _protect_metrics(query_file, gates, image_subset=None):
    arrays = np.load(query_file)
    target = ~arrays["g_gt"].astype(bool)
    image_ids = arrays["image_id"].astype(np.int64)
    mask = np.ones(target.shape, dtype=bool)
    if image_subset is not None:
        mask &= np.isin(image_ids, np.asarray(image_subset, dtype=np.int64))
    score = 1.0 - arrays["p_suppress"].astype(float)
    result = {}
    for name, gate in gates.items():
        predicted = ~gate.astype(bool)
        y, pred = target[mask], predicted[mask]
        tp = int((y & pred).sum())
        fp = int((~y & pred).sum())
        fn = int((y & ~pred).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        result[name] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0,
            "ap": _average_precision_binary(y, score[mask]),
            "protect_count": int(y.sum()),
            "predicted_protect_count": int(pred.sum()),
            "false_suppress_count": int((y & ~pred).sum()),
            "missed_suppress_count": int((~y & pred).sum()),
        }
    return result


def _contrast_value(scored, contrast, metric):
    if len(contrast) == 2:
        left, right = contrast
        return scored[right][metric] - scored[left][metric]
    base, first, second, both = contrast
    return scored[both][metric] - scored[first][metric] - scored[second][metric] + scored[base][metric]


def _paired_bootstrap(ground_truth, predictions, matches, image_ids, contrasts, repetitions, seed):
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
        for condition in predictions
    }
    samples = {name: {"ap50": [], "duct_ap50": [], "fixed_f1": []} for name in contrasts}
    rng = np.random.default_rng(seed)
    image_ids = np.asarray(image_ids, dtype=np.int64)
    for repetition in range(repetitions):
        draw = rng.choice(image_ids, size=len(image_ids), replace=True)
        resampled_gt, arms = remap_bootstrap_sample(ground_truth, coco_predictions, draw)
        scored = {condition: coco_ap50(resampled_gt, detections) for condition, detections in arms.items()}
        fixed_scored = {}
        for condition in predictions:
            tp = sum(matches[condition][int(image_id)]["tp"] for image_id in draw)
            fp = sum(matches[condition][int(image_id)]["fp"] for image_id in draw)
            fn = sum(matches[condition][int(image_id)]["fn"] for image_id in draw)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            fixed_scored[condition] = {
                "fixed_f1": 2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            }
        for name, contrast in contrasts.items():
            for metric in ("ap50", "duct_ap50"):
                samples[name][metric].append(_contrast_value(scored, contrast, metric))
            samples[name]["fixed_f1"].append(_contrast_value(fixed_scored, contrast, "fixed_f1"))
        if (repetition + 1) % 25 == 0:
            print(f"bootstrap {repetition + 1}/{repetitions}", flush=True)
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
    return {
        "method": "paired image resampling with shared draws; COCO AP50 recomputed",
        "repetitions": repetitions,
        "seed": seed,
        "limitation": "patient IDs unavailable for many images; image-level intervals may be optimistic; one model seed",
        "intervals": intervals,
    }


def _metric_delta(coco, fixed, subset, contrast):
    if len(contrast) == 2:
        left, right = contrast
        terms = ((right, 1.0), (left, -1.0))
    else:
        base, first, second, both = contrast
        terms = ((both, 1.0), (first, -1.0), (second, -1.0), (base, 1.0))
    result = {
        metric: sum(weight * coco[name][subset][metric] for name, weight in terms)
        for metric in ("ap", "ap50", "ap75", "ar100")
    }
    result["fixed_f1"] = sum(weight * fixed[name][subset]["f1"] for name, weight in terms)
    return result


def analyze_g(args, ground_truth, index, split_payload):
    result_dir = Path(args.result_dir)
    image_ids = index["image_ids"]
    predictions = {
        condition: load_predictions(result_dir / f"predictions_{condition}.jsonl", image_ids)
        for condition in G_CONDITIONS
    }
    matches = _matches(
        G_CONDITIONS, predictions, index["annotations_by_image"], image_ids,
        args.score_threshold, args.iou_threshold,
    )
    subsets = {
        "full": image_ids,
        "positive_images": index["positive_ids"],
        "empty_images": index["empty_ids"],
    }
    coco = _coco_by_subset(
        ground_truth, G_CONDITIONS, predictions,
        {key: value for key, value in subsets.items() if key != "empty_images"},
    )
    fixed = _fixed_by_subset(matches, subsets)
    query_payload = _query_importance(
        result_dir, Path(args.query_out), G_CONDITIONS, predictions, matches, index
    )
    arrays = np.load(result_dir / "query_diagnostics.npz")
    g_gates = {condition: arrays[f"gate_{condition}"] for condition in G_CONDITIONS if condition != "A"}
    contrasts = {
        "G1_minus_G0_correct_false_suppress": ("G0", "G1"),
        "G2_minus_G0_correct_missed_suppress": ("G0", "G2"),
        "G3_minus_G0_correct_both": ("G0", "G3"),
        "interaction_G3_minus_G1_minus_G2_plus_G0": ("G0", "G1", "G2", "G3"),
    }
    pair_contrasts = {name: value for name, value in contrasts.items() if len(value) == 2}
    payload = {
        "scope": "cmp5L G frozen-original-EMA validation-only; main attribution on positive images",
        "split": {
            "images": len(image_ids),
            "positive_images": len(index["positive_ids"]),
            "empty_images": len(index["empty_ids"]),
            "h_locked_split_seed": split_payload.get("seed"),
        },
        "official_full_metrics": json.loads((result_dir / "metrics_full.json").read_text()),
        "coco_metrics": coco,
        "fixed_metrics": fixed,
        "fixed_rule": {
            "score_threshold": args.score_threshold,
            "iou_threshold": args.iou_threshold,
            "class_consistent": True,
            "one_to_one": True,
        },
        "positive_image_point_deltas": {
            name: _metric_delta(coco, fixed, "positive_images", contrast)
            for name, contrast in contrasts.items()
        },
        "query_error_summary": _query_error_summary(query_payload, G_CONDITIONS),
        "protect_metrics_positive": _protect_metrics(
            result_dir / "query_diagnostics.npz", g_gates, index["positive_ids"]
        ),
        "gt_transitions": {
            name: _transition_details(matches, pair[0], pair[1], index["annotation_by_id"])
            for name, pair in pair_contrasts.items()
        },
        "duct_transitions": {
            name: _transition_details(
                matches, pair[0], pair[1], index["annotation_by_id"],
                lambda annotation: int(annotation["category_id"]) == 3,
            )
            for name, pair in pair_contrasts.items()
        },
        "bootstrap_positive_images": _paired_bootstrap(
            ground_truth, predictions, matches, index["positive_ids"], contrasts,
            args.bootstrap_repetitions, args.seed,
        ),
        "technical_audit": json.loads((result_dir / "audit.json").read_text()),
        "interpretation_boundary": "AP is nonlinear; G1/G2/interaction are contrasts, not additive causal shares or deployable methods",
    }
    return payload


def analyze_h(args, ground_truth, index, split_payload, calibration_ids, evaluation_ids):
    result_dir = Path(args.result_dir)
    image_ids = index["image_ids"]
    predictions = {
        condition: load_predictions(result_dir / f"predictions_{condition}.jsonl", image_ids)
        for condition in H_CONDITIONS
    }
    matches = _matches(
        H_CONDITIONS, predictions, index["annotations_by_image"], image_ids,
        args.score_threshold, args.iou_threshold,
    )
    subsets = {
        "full": image_ids,
        "calibration": calibration_ids,
        "evaluation": evaluation_ids,
        "positive_images": index["positive_ids"],
        "empty_images": index["empty_ids"],
    }
    coco = _coco_by_subset(
        ground_truth, H_CONDITIONS, predictions,
        {key: value for key, value in subsets.items() if key != "empty_images"},
    )
    fixed = _fixed_by_subset(matches, subsets)
    baseline = {
        "ap50": coco["A"]["calibration"]["ap50"],
        "fixed_f1": fixed["A"]["calibration"]["f1"],
        "duct_ap50": coco["A"]["calibration"]["per_class_ap50"]["3"],
    }
    candidates = {
        condition: {
            "threshold": threshold,
            "ap50": coco[condition]["calibration"]["ap50"],
            "fixed_f1": fixed[condition]["calibration"]["f1"],
            "duct_ap50": coco[condition]["calibration"]["per_class_ap50"]["3"],
        }
        for condition, threshold in H_THRESHOLDS.items()
    }
    selection = select_conservative_threshold(baseline, candidates)
    confirm = ["A", "H005"]
    if selection["selected"] and selection["selected"] not in confirm:
        confirm.append(selection["selected"])
    evaluation_confirmation = {
        condition: {"coco": coco[condition]["evaluation"], "fixed": fixed[condition]["evaluation"]}
        for condition in confirm
    }
    contrasts = build_h_evaluation_contrasts(selection["selected"])
    used = sorted({condition for pair in contrasts.values() for condition in pair})
    bootstrap = _paired_bootstrap(
        ground_truth,
        {condition: predictions[condition] for condition in used},
        {condition: matches[condition] for condition in used},
        evaluation_ids, contrasts, args.bootstrap_repetitions, args.seed,
    )
    evaluation_deltas = {
        name: _metric_delta(coco, fixed, "evaluation", contrast)
        for name, contrast in contrasts.items()
    }
    query_payload = _query_importance(
        result_dir, Path(args.query_out), H_CONDITIONS, predictions, matches, index
    )
    arrays = np.load(result_dir / "query_diagnostics.npz")
    h_gates = {condition: arrays[f"gate_{condition}"] for condition in H_THRESHOLDS}
    payload = {
        "scope": "cmp5L H frozen-predictor conservative thresholds on locked validation split",
        "split": {
            "seed": split_payload["seed"],
            "unit": split_payload["unit"],
            "calibration_images": len(calibration_ids),
            "evaluation_images": len(evaluation_ids),
            "limitation": "development-data split; predictor and historical threshold already used validation",
        },
        "official_full_metrics_descriptive_only": json.loads((result_dir / "metrics_full.json").read_text()),
        "calibration": {
            "baseline_floors": baseline,
            "candidates": candidates,
            "selection": selection,
            "all_coco_metrics": {condition: coco[condition]["calibration"] for condition in H_CONDITIONS},
            "all_fixed_metrics": {condition: fixed[condition]["calibration"] for condition in H_CONDITIONS},
        },
        "evaluation_confirmation_only": evaluation_confirmation,
        "evaluation_point_deltas": evaluation_deltas,
        "bootstrap_evaluation": bootstrap,
        "fixed_metrics_full_positive_empty": {
            condition: {
                subset: fixed[condition][subset]
                for subset in ("full", "positive_images", "empty_images")
            }
            for condition in H_CONDITIONS
        },
        "protect_metrics": {
            subset: _protect_metrics(result_dir / "query_diagnostics.npz", h_gates, ids)
            for subset, ids in (("full", None), ("calibration", calibration_ids), ("evaluation", evaluation_ids))
        },
        "query_error_summary": _query_error_summary(query_payload, H_CONDITIONS),
        "historical_to_selected_transitions": (
            _transition_details(matches, "H005", selection["selected"], index["annotation_by_id"])
            if selection["selected"]
            else None
        ),
        "technical_audit": json.loads((result_dir / "audit.json").read_text()),
    }
    return payload


def analyze(args):
    if args.bootstrap_repetitions < 500:
        raise ValueError("the locked protocol requires at least 500 bootstrap repetitions")
    result_dir = Path(args.result_dir)
    if not (result_dir / "COMPLETED.txt").exists():
        raise RuntimeError("inference result is not marked complete")
    ground_truth = json.loads(Path(args.ann_file).read_text(encoding="utf-8"))
    index = _ground_truth_index(ground_truth)
    split_payload, calibration_ids, evaluation_ids = _load_split(args.split_file)
    if set(calibration_ids) | set(evaluation_ids) != set(index["image_ids"]):
        raise ValueError("locked split does not cover validation exactly")
    payload = (
        analyze_g(args, ground_truth, index, split_payload)
        if args.mode == "G"
        else analyze_h(args, ground_truth, index, split_payload, calibration_ids, evaluation_ids)
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False)[:20000], flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("G", "H"), required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--query-out", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-repetitions", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260919)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
