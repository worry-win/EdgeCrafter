"""Image-paired COCO AP50 bootstrap for frozen cmp5L audit predictions."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
from collections import defaultdict
from pathlib import Path


def remap_sample(ground_truth, predictions_by_arm, sampled_image_ids):
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


def load_postprocessed_predictions(path, expected_ids, category_ids):
    by_image = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            image_id = int(record["image_id"])
            if image_id in by_image:
                raise ValueError(f"duplicate image {image_id} in {path}")
            boxes, scores, labels = (
                record["boxes_xyxy"], record["scores"], record["labels"]
            )
            if not (len(boxes) == len(scores) == len(labels)):
                raise ValueError(f"mismatched prediction fields in image {image_id}")
            indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:100]
            predictions = []
            for index in indices:
                x0, y0, x1, y1 = boxes[index]
                score = float(scores[index])
                category = int(labels[index])
                if category not in category_ids or not all(map(math.isfinite, (x0, y0, x1, y1, score))):
                    raise ValueError(f"invalid prediction in image {image_id}")
                predictions.append({
                    "image_id": image_id,
                    "category_id": category,
                    "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                    "score": score,
                })
            by_image[image_id] = predictions
    if set(by_image) != set(expected_ids):
        raise ValueError(f"prediction image IDs differ in {path}")
    return by_image


def coco_ap50(ground_truth, detections):
    import io
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = ground_truth
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(detections)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.iouThrs = np.array([0.50])
        evaluator.params.imgIds = [image["id"] for image in ground_truth["images"]]
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"][0, :, :, 0, -1]
    valid = precision > -1
    ap50 = float(precision[valid].mean()) if valid.any() else float("nan")
    duct_index = list(evaluator.params.catIds).index(3)
    duct = precision[:, duct_index]
    duct_ap50 = float(duct[duct > -1].mean()) if (duct > -1).any() else float("nan")
    return {"ap50": ap50, "duct_ap50": duct_ap50}


def main():
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--arm", action="append", required=True, help="NAME=metrics.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ground_truth = json.loads(Path(args.ann_file).read_text(encoding="utf-8"))
    image_ids = [int(image["id"]) for image in ground_truth["images"]]
    category_ids = {int(category["id"]) for category in ground_truth["categories"]}
    metrics, predictions = {}, {}
    for entry in args.arm:
        name, metric_file = entry.split("=", 1)
        metric = json.loads(Path(metric_file).read_text(encoding="utf-8"))
        if metric["meta"]["n_images"] != len(image_ids):
            raise ValueError(f"{name}: wrong evaluated image count")
        predictions[name] = load_postprocessed_predictions(
            metric["meta"]["predictions_file"], image_ids, category_ids
        )
        metrics[name] = metric["metrics"]
    required = {"original_ema", "S0_ema", "S0_model", "S1_on_model", "S1_off_model", "S1_off_ema"}
    if set(predictions) != required:
        raise ValueError(f"expected arms {required}, got {set(predictions)}")
    point = {}
    for name, by_image in predictions.items():
        detections = [prediction for image_id in image_ids for prediction in by_image[image_id]]
        calculated = coco_ap50(ground_truth, detections)
        reported = float(metrics[name]["ap50"])
        if abs(calculated["ap50"] - reported) > 1e-4:
            raise RuntimeError(f"{name}: prediction AP50 {calculated['ap50']} != metric {reported}")
        point[name] = calculated
        print(f"verified {name}: AP50={reported:.6f}", flush=True)

    contrasts = {
        "S0_ema_minus_original_ema": ("S0_ema", "original_ema"),
        "S1_on_minus_off_same_Student": ("S1_on_model", "S1_off_model"),
        "S1_on_Student_minus_S0_Student": ("S1_on_model", "S0_model"),
        "S1_off_ema_minus_S0_ema": ("S1_off_ema", "S0_ema"),
    }
    rng = np.random.default_rng(args.seed)
    samples = {name: {"ap50": [], "duct_ap50": []} for name in contrasts}
    for rep in range(args.repetitions):
        draw = rng.choice(image_ids, size=len(image_ids), replace=True)
        gt, arms = remap_sample(ground_truth, predictions, draw)
        scored = {name: coco_ap50(gt, arm) for name, arm in arms.items()}
        for name, (positive, negative) in contrasts.items():
            for metric in ("ap50", "duct_ap50"):
                samples[name][metric].append(scored[positive][metric] - scored[negative][metric])
        if (rep + 1) % 10 == 0:
            print(f"bootstrap {rep + 1}/{args.repetitions}", flush=True)

    intervals = {}
    for name, (positive, negative) in contrasts.items():
        intervals[name] = {}
        for metric, draws in samples[name].items():
            values = np.asarray(draws, dtype=float)
            values = values[np.isfinite(values)]
            intervals[name][metric] = {
                "point_delta": point[positive][metric] - point[negative][metric],
                "ci95_percentile": np.quantile(values, [0.025, 0.975]).tolist(),
                "bootstrap_positive_fraction": float((values > 0).mean()),
                "valid_repetitions": int(values.size),
            }
    payload = {
        "method": "paired image-resampling; identical draws for every arm; COCO AP50 recomputed",
        "limitation": "patient IDs unavailable for most images; image-level CI may be optimistic with within-patient correlation; one training seed",
        "seed": args.seed,
        "repetitions": args.repetitions,
        "images": len(image_ids),
        "point": point,
        "contrasts": intervals,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(intervals, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
