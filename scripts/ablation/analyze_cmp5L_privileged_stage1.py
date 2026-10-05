"""Offline target-quality analysis and counterfactual case locking for Stage 1."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.ablation.analyze_cmp5L_af_validation import (
    _fixed_metrics,
    fixed_threshold_match,
    load_predictions,
)
from scripts.ablation.cmp5L_privileged_decision_kd import select_counterfactual_cases


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cxcywh_iou(left, right):
    lx, ly, lw, lh = map(float, left)
    rx, ry, rw, rh = map(float, right)
    lx0, ly0, lx1, ly1 = lx - lw / 2, ly - lh / 2, lx + lw / 2, ly + lh / 2
    rx0, ry0, rx1, ry1 = rx - rw / 2, ry - rh / 2, rx + rw / 2, ry + rh / 2
    ix0, iy0, ix1, iy1 = max(lx0, rx0), max(ly0, ry0), min(lx1, rx1), min(ly1, ry1)
    intersection = max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


def _safe_recovery(normal, privileged, mixed):
    denominator = float(privileged) - float(normal)
    return (float(mixed) - float(normal)) / denominator if abs(denominator) > 1e-12 else None


def _transition_counts(left, right, annotation_ids):
    result = Counter()
    for image_id, current in annotation_ids.items():
        left_map = left[image_id]["gt_matches"]
        right_map = right[image_id]["gt_matches"]
        for annotation_id in current:
            left_hit = annotation_id in left_map
            right_hit = annotation_id in right_map
            if left_hit and right_hit:
                result["stable_detected"] += 1
                result[
                    "stable_same_query"
                    if left_map[annotation_id] == right_map[annotation_id]
                    else "stable_reassigned"
                ] += 1
            elif right_hit:
                result["rescued"] += 1
            elif left_hit:
                result["destroyed"] += 1
            else:
                result["stable_missed"] += 1
    return dict(result)


def analyze(args):
    result_dir = Path(args.result_dir)
    output = Path(args.output)
    case_manifest_path = Path(args.counterfactual_manifest)
    case_coco_path = Path(args.counterfactual_coco)
    for path in (output, case_manifest_path, case_coco_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    ground_truth = json.loads(Path(args.ann_file).read_text(encoding="utf-8"))
    image_by_id = {int(image["id"]): image for image in ground_truth["images"]}
    annotations_by_image = defaultdict(list)
    for annotation in ground_truth["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    image_ids = [int(image["id"]) for image in ground_truth["images"]]
    annotation_ids = {
        image_id: [int(annotation["id"]) for annotation in annotations_by_image[image_id]]
        for image_id in image_ids
    }
    predictions = {
        condition: load_predictions(result_dir / f"predictions_{condition}.jsonl", image_ids)
        for condition in ("N", "P", "M")
    }
    matches = {
        condition: {
            image_id: fixed_threshold_match(
                annotations_by_image[image_id], predictions[condition][image_id], 0.5, 0.5
            )
            for image_id in image_ids
        }
        for condition in predictions
    }
    fixed = {
        condition: _fixed_metrics(matches[condition], image_ids)
        for condition in matches
    }
    rows = [
        json.loads(line)
        for line in (result_dir / "selected_targets.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if len(rows) != len(image_ids) or {int(row["image_id"]) for row in rows} != set(image_ids):
        raise RuntimeError("selected-target row coverage differs from subset")
    row_by_image = {int(row["image_id"]): row for row in rows}
    teacher = np.load(result_dir / "teacher_outputs.npz")
    array_image_ids = teacher["image_ids"].astype(np.int64)
    if set(map(int, array_image_ids)) != set(image_ids):
        raise RuntimeError("teacher output image coverage differs from subset")
    array_row = {int(image_id): index for index, image_id in enumerate(array_image_ids)}

    positive_detail, negative_detail = [], []
    layer_summary = defaultdict(lambda: defaultdict(list))
    boundary = Counter()
    class_counts = defaultdict(Counter)
    for row in rows:
        image_id = int(row["image_id"])
        index = array_row[image_id]
        annotations = annotations_by_image[image_id]
        for item in row["positive"]:
            gt_index = int(item["gt_index"])
            query = int(item["teacher_query"])
            annotation_id = int(annotations[gt_index]["id"])
            n_box = teacher["N_final_boxes"][index, query].astype(float)
            p_box = teacher["P_final_boxes"][index, query].astype(float)
            gt_box = row["gt_boxes_cxcywh"][gt_index]
            record = {
                "image_id": image_id,
                "annotation_id": annotation_id,
                **item,
                "normal_box_iou": _cxcywh_iou(n_box, gt_box),
                "privileged_box_iou": _cxcywh_iou(p_box, gt_box),
                "mixed_detected": annotation_id in matches["M"][image_id]["gt_matches"],
                "normal_detected": annotation_id in matches["N"][image_id]["gt_matches"],
                "privileged_detected": annotation_id in matches["P"][image_id]["gt_matches"],
            }
            positive_detail.append(record)
            class_counts[item["kind"]][str(item["gt_label"])] += 1
            if item["threshold_boundary"]:
                boundary[item["kind"]] += 1
            for layer, values in enumerate(item["layers"]):
                for key in ("gt_score", "gt_margin"):
                    layer_summary[item["kind"]][f"L{layer}_{key}"].append(float(values[key]))
        for item in row["negative"]:
            delta = float(item["normal_score"]) - float(item["privileged_score"])
            record = {"image_id": image_id, **item, "score_drop": delta}
            negative_detail.append(record)
            class_counts["suppress"][str(item["class_index"])] += 1
            if item["threshold_boundary"]:
                boundary["suppress"] += 1
            for branch in ("normal", "privileged"):
                for layer, values in enumerate(item[f"{branch}_layers"]):
                    for key in ("gt_score", "gt_margin"):
                        layer_summary["suppress"][f"{branch}_L{layer}_{key}"].append(float(values[key]))

    metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    audit = json.loads((result_dir / "audit.json").read_text(encoding="utf-8"))
    repair_detail = [item for item in positive_detail if item["kind"] == "repair"]
    protect_detail = [item for item in positive_detail if item["kind"] == "protect"]
    negative_drops = [item["score_drop"] for item in negative_detail]
    summary = {
        "scope": "enriched fixed train subset; not a train-population estimate",
        "technical_gate_passed": bool(
            audit["model_load"]["missing_keys"] == []
            and audit["model_load"]["unexpected_keys"] == []
            and max(audit["normal_N_max_abs"].values()) == 0
            and max(audit["captured_final_max_abs"].values()) == 0
            and max(audit["empty_P_identity_max_abs"].values()) == 0
            and audit["location_input_mutation_max_abs"] == 0
            and audit["attention_input_mutation_max_abs"] == 0
            and audit["teacher_buffer_max_abs_change"] == 0
            and audit["n_images"] == 640
        ),
        "metrics": metrics,
        "metric_deltas_pp": {
            contrast: {
                metric: 100 * (metrics[right][metric] - metrics[left][metric])
                for metric in ("ap", "ap50", "ap75", "precision", "recall", "f1", "ar100")
            }
            for contrast, left, right in (("P_minus_N", "N", "P"), ("M_minus_N", "N", "M"), ("M_minus_P", "P", "M"))
        },
        "classification_recovery_fraction": {
            metric: _safe_recovery(metrics["N"][metric], metrics["P"][metric], metrics["M"][metric])
            for metric in ("ap", "ap50", "ap75", "precision", "recall", "f1", "ar100")
        },
        "fixed_score05_iou05": fixed,
        "transitions": {
            contrast: _transition_counts(matches[left], matches[right], annotation_ids)
            for contrast, left, right in (("N_to_P", "N", "P"), ("N_to_M", "N", "M"), ("P_to_M", "P", "M"))
        },
        "targets": {
            "counts": audit["target_counts"],
            "by_class": {key: dict(value) for key, value in class_counts.items()},
            "boundary_counts": dict(boundary),
            "repairs_recovered_by_classification_only_M": sum(item["mixed_detected"] for item in repair_detail),
            "repairs_with_N_box_iou_ge_05": sum(item["normal_box_iou"] >= 0.5 for item in repair_detail),
            "protect_targets_preserved_by_M": sum(item["mixed_detected"] for item in protect_detail),
            "suppress_targets_crossing_below_score05": sum(float(item["privileged_score"]) < 0.5 for item in negative_detail),
            "suppress_tiny_drop_le_001": sum(item["score_drop"] <= 0.01 for item in negative_detail),
            "suppress_score_drop_mean": float(np.mean(negative_drops)) if negative_drops else None,
            "suppress_score_drop_median": float(np.median(negative_drops)) if negative_drops else None,
        },
        "layer_target_means": {
            kind: {key: float(np.mean(values)) for key, values in fields.items() if values}
            for kind, fields in layer_summary.items()
        },
        "positive_detail": positive_detail,
        "negative_detail": negative_detail,
    }

    cases = select_counterfactual_cases(rows, per_category=args.cases_per_category)
    selected_case_ids = sorted({item["image_id"] for values in cases.values() for item in values})
    case_coco = {
        **{key: value for key, value in ground_truth.items() if key not in ("images", "annotations")},
        "images": [image_by_id[image_id] for image_id in selected_case_ids],
        "annotations": [
            annotation for image_id in selected_case_ids for annotation in annotations_by_image[image_id]
        ],
    }
    case_manifest = {
        "schema_version": 1,
        "selection_rule": "first sorted unique-image cases; no counterfactual result reroll",
        "per_category_limit": args.cases_per_category,
        "cases": cases,
        "image_ids": selected_case_ids,
        "source_hashes": {
            "selected_targets": sha256_path(result_dir / "selected_targets.jsonl"),
            "subset_annotations": sha256_path(args.ann_file),
            "teacher_outputs": sha256_path(result_dir / "teacher_outputs.npz"),
        },
    }
    canonical = json.dumps(case_manifest, sort_keys=True, separators=(",", ":"))
    case_manifest["selection_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    summary["counterfactual_case_manifest"] = {
        "path": str(case_manifest_path),
        "selection_sha256": case_manifest["selection_sha256"],
        "counts": {key: len(value) for key, value in cases.items()},
        "unique_images": len(selected_case_ids),
    }
    summary["preliminary_decision"] = {
        "classification_signal_present": bool(
            metrics["M"]["ap50"] > metrics["N"]["ap50"]
            and metrics["M"]["f1"] >= metrics["N"]["f1"]
            and metrics["M"]["per_class_ap50"]["3"]["ap50"] >= metrics["N"]["per_class_ap50"]["3"]["ap50"]
        ),
        "training_authorized": False,
        "remaining_gates": ["counterfactual attribution", "augmented-train online target coverage"],
    }
    for path, payload in ((output, summary), (case_manifest_path, case_manifest), (case_coco_path, case_coco)):
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "analysis": str(output),
        "counterfactual_manifest": str(case_manifest_path),
        "case_counts": {key: len(value) for key, value in cases.items()},
        "unique_case_images": len(selected_case_ids),
        "preliminary_decision": summary["preliminary_decision"],
    }, ensure_ascii=False, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--counterfactual-manifest", required=True)
    parser.add_argument("--counterfactual-coco", required=True)
    parser.add_argument("--cases-per-category", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
