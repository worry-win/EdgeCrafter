"""Uniform held-out test evaluation for cmp5L query-behavior pilots."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "ecdetseg"))
sys.path.insert(0, str(PROJECT_ROOT))

from engine.solver.ec_engine import (  # noqa: E402
    summarize_pr_curve_f1,
    summarize_yolo_pr_curve_metrics,
)
from scripts.ablation.probe_cmp5L_decoder_internal_tensors import _build  # noqa: E402


def prediction_record(image_id, result):
    return {
        "image_id": int(image_id),
        "boxes_xyxy": result["boxes"].detach().cpu().tolist(),
        "scores": result["scores"].detach().cpu().tolist(),
        "labels": result["labels"].detach().cpu().tolist(),
    }


def evaluate(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    args.device = device
    cfg, solver, model = _build(args)
    evaluator = cfg.evaluator
    evaluator.cleanup()
    seen = 0
    started = time.time()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions_out = output.with_suffix(".predictions.jsonl")
    prediction_stream = predictions_out.open("x", encoding="utf-8")

    try:
        with torch.no_grad():
            for samples, targets in cfg.val_dataloader:
                samples = samples.to(device)
                outputs = model(samples)
                original_sizes = torch.stack(
                    [target["orig_size"] for target in targets]
                ).to(device)
                results = solver.postprocessor(outputs, original_sizes)
                predictions = {
                    int(target["image_id"].item()): result
                    for target, result in zip(targets, results)
                }
                evaluator.update(predictions)
                for image_id, result in predictions.items():
                    prediction_stream.write(
                        json.dumps(prediction_record(image_id, result)) + "\n"
                    )
                seen += len(targets)
                if args.log_every and seen % args.log_every < len(targets):
                    print(
                        f"[{seen}/{len(cfg.val_dataloader.dataset)}] "
                        f"{time.time() - started:.1f}s",
                        flush=True,
                    )
    finally:
        prediction_stream.close()

    if args.expected_images and seen != args.expected_images:
        raise RuntimeError(f"evaluated {seen}, expected {args.expected_images}")
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    coco_eval = evaluator.coco_eval["bbox"]
    macro = summarize_pr_curve_f1(coco_eval)
    yolo = summarize_yolo_pr_curve_metrics(coco_eval, evaluator.coco_gt) or {}
    overall = yolo.get("yolo_f1_iou50", {})

    precision = coco_eval.eval["precision"]
    iou50 = int(np.argmin(np.abs(coco_eval.params.iouThrs - 0.50)))
    categories = evaluator.coco_gt.loadCats(coco_eval.params.catIds)
    per_class_ap50 = {}
    for class_index, category in enumerate(categories):
        curve = precision[iou50, :, class_index, 0, -1]
        valid = curve > -1
        per_class_ap50[str(category["id"])] = {
            "name": category.get("name", str(category["id"])),
            "ap50": float(curve[valid].mean()) if valid.any() else -1.0,
        }

    stats = coco_eval.stats.tolist()
    payload = {
        "meta": {
            "checkpoint": args.checkpoint,
            "weights": args.weights,
            "ann_file": args.ann_file,
            "n_images": seen,
            "seconds": round(time.time() - started, 2),
            "predictions_file": str(predictions_out),
        },
        "metrics": {
            "ap": float(stats[0]),
            "ap50": float(stats[1]),
            "ap75": float(stats[2]),
            "precision": float(overall.get("precision", 0.0)),
            "recall": float(overall.get("recall", 0.0)),
            "f1": float(overall.get("f1", macro["f1_iou50"])),
            "ar100": float(stats[8]),
            "small_ap": float(stats[3]),
            "medium_ap": float(stats[4]),
            "large_ap": float(stats[5]),
        },
        "per_class_ap50": per_class_ap50,
        "yolo": yolo,
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload["metrics"], ensure_ascii=False, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--img-folder", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--expected-images", type=int, default=2011)
    parser.add_argument("--log-every", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
