"""Full Oracle evaluation with GT spatial masks after HybridEncoder."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
import sys

from typing import Optional, Sequence, Tuple

import torch
import torchvision


EC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EC_ROOT / "ecdetseg"))
sys.path.insert(0, str(EC_ROOT))

from scripts.ablation.probe_cmp5L_privileged_ema_oracle import (  # noqa: E402
    _boxes_tensor,
    _condition_specs,
    _iou_matrix,
    privilege_features,
)
from scripts.ablation.probe_cmp5L_decoder_internal_tensors import (  # noqa: E402
    _build,
    _normalise_targets,
)
from scripts.ablation.dump_cmp5L_internal_behavior import _target_index  # noqa: E402
from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy  # noqa: E402
from engine.solver.ec_engine import (  # noqa: E402
    summarize_pr_curve_f1,
    summarize_yolo_pr_curve_metrics,
)


def mask_encoder_outputs(
    encoded: Sequence[torch.Tensor],
    targets,
    image_hw: Tuple[int, int],
    background_weight: float,
    context_weight: Optional[float] = None,
    context_scale: float = 1.5,
):
    """Mask cloned HybridEncoder outputs using each output's actual H×W."""
    return privilege_features(
        encoded,
        targets,
        image_hw,
        background_weight,
        context_weight,
        context_scale,
    )


def compact_binding_metrics(boxes_xyxy, logits, gt_xyxy, gt_class):
    """Return only the binding endpoints requested for location comparison."""
    gt_xyxy = torch.as_tensor(gt_xyxy, device=boxes_xyxy.device, dtype=boxes_xyxy.dtype)
    iou = _iou_matrix(boxes_xyxy, gt_xyxy[None])[:, 0]
    gt_logits = logits[:, gt_class]
    final_scores = logits.sigmoid().max(-1).values
    q_iou = int(iou.argmax())
    q_cls = int(gt_logits.argmax())
    order = torch.argsort(final_scores, descending=True)
    top1_iou = iou[order[:1]].max()
    top5_iou = iou[order[:min(5, len(order))]].max()
    return {
        "q_iou_equals_q_cls": q_iou == q_cls,
        "gt_logit_q_iou": float(gt_logits[q_iou]),
        "iou_q_cls": float(iou[q_cls]),
        "rank_q_iou_final_score": int((final_scores > final_scores[q_iou]).sum()) + 1,
        "top1_top5_iou_gap": float(top5_iou - top1_iou),
    }


def full_metrics_row(condition, stats, n_images):
    coco = stats["coco_eval_bbox"]
    pr = stats["yolo_f1_iou50"]
    return {
        "condition": condition,
        "n_images": int(n_images),
        "map": float(coco[0]),
        "map50": float(coco[1]),
        "map75": float(coco[2]),
        "precision": float(pr["precision"]),
        "recall": float(pr["recall"]),
        "f1": float(pr["f1"]),
        "ar100": float(coco[8]),
        "macro_f1_iou50": float(stats["macro_f1_iou50"]),
    }


def _raster_bounds(box, image_hw, feature_hw, scale=1.0):
    image_h, image_w = image_hw
    feature_h, feature_w = feature_hw
    x0, y0, x1, y1 = [float(value) for value in box]
    if scale != 1.0:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half_w, half_h = (x1 - x0) * scale / 2, (y1 - y0) * scale / 2
        x0, x1, y0, y1 = cx - half_w, cx + half_w, cy - half_h, cy + half_h
    return [
        max(0, min(feature_w, int(torch.floor(torch.tensor(x0 * feature_w / image_w))))),
        max(0, min(feature_h, int(torch.floor(torch.tensor(y0 * feature_h / image_h))))),
        max(0, min(feature_w, int(torch.ceil(torch.tensor(x1 * feature_w / image_w))))),
        max(0, min(feature_h, int(torch.ceil(torch.tensor(y1 * feature_h / image_h))))),
    ]


def _audit_condition(encoded, masks, targets, image_hw, spec, batch_index):
    boxes = _boxes_tensor(targets[batch_index], encoded[0].device).detach().cpu()
    levels = []
    for level, (feature, mask) in enumerate(zip(encoded, masks)):
        feature_hw = feature.shape[-2:]
        item_mask = mask[batch_index]
        levels.append({
            "level": level,
            "feature_shape": list(feature.shape),
            "spatial_hw": list(feature_hw),
            "inferred_stride_hw": [
                image_hw[0] / feature_hw[0], image_hw[1] / feature_hw[1]
            ],
            "mask_shape": list(mask.shape),
            "mask_unique": [float(value) for value in torch.unique(item_mask)],
            "mask_mean": float(item_mask.mean()),
            "foreground_cells": int((item_mask == 1).sum()),
            "gt_mapping": [
                {
                    "box_xyxy_input": [float(value) for value in box],
                    "gt_bounds_xyxy": _raster_bounds(box, image_hw, feature_hw),
                    "context_bounds_xyxy": (
                        _raster_bounds(box, image_hw, feature_hw, spec["context_scale"])
                        if spec["context_weight"] is not None else None
                    ),
                }
                for box in boxes
            ],
        })
    return {"condition": spec["name"], "sample_index": batch_index, "levels": levels}


def _mean_binding(records, condition):
    keys = (
        "q_iou_equals_q_cls", "gt_logit_q_iou", "iou_q_cls",
        "rank_q_iou_final_score", "top1_top5_iou_gap",
    )
    return {
        key: sum(float(record["conditions"][condition][key]) for record in records) / len(records)
        for key in keys
    }


def summarize_binding(records, conditions):
    groups = sorted({record["group"] for record in records})
    return {
        condition: {
            "overall": _mean_binding(records, condition),
            "groups": {
                group: _mean_binding(
                    [record for record in records if record["group"] == group], condition
                )
                for group in groups
            },
        }
        for condition in conditions
    }


def render_markdown(after_rows, before_rows):
    before = {row["condition"]: row for row in before_rows}
    lines = [
        "# Mask Location Oracle Comparison",
        "",
        "| Condition | Before HE AP50 | After HE AP50 | Δ After-Before | Before F1 | After F1 | Δ F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in after_rows:
        base = before[row["condition"]]
        lines.append(
            f"| {row['condition']} | {base['map50']:.4f} | {row['map50']:.4f} | "
            f"{row['map50'] - base['map50']:+.4f} | {base['f1']:.4f} | "
            f"{row['f1']:.4f} | {row['f1'] - base['f1']:+.4f} |"
        )
    lines.extend([
        "",
        "| After-HE condition | mAP | mAP50 | mAP75 | Precision | Recall | F1 | AR@100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in after_rows:
        lines.append(
            f"| {row['condition']} | {row['map']:.4f} | {row['map50']:.4f} | "
            f"{row['map75']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['ar100']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def evaluate(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; refusing silent CPU fallback")
    args.device = device
    manifest_rows = [
        json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()
    ]
    rows_by_image = {}
    binding_records = {}
    for row in manifest_rows:
        rows_by_image.setdefault(int(row["image_id"]), []).append(row)
        binding_records[int(row["ann_id"])] = {
            "ann_id": int(row["ann_id"]),
            "image_id": int(row["image_id"]),
            "gt_class": int(row["gt_class"]),
            "class_name": row["class_name"],
            "group": row["group"],
            "conditions": {},
        }

    cfg, solver, model = _build(args)
    specs = _condition_specs(args.context_alpha, args.context_beta, args.context_scale)
    evaluators = {spec["name"]: copy.deepcopy(cfg.evaluator) for spec in specs}
    for evaluator in evaluators.values():
        evaluator.cleanup()

    seen = 0
    parity = None
    shape_audit = []
    started = time.time()
    with torch.no_grad():
        for batch_number, (samples, targets) in enumerate(cfg.val_dataloader):
            if args.max_batches and batch_number >= args.max_batches:
                break
            samples = samples.to(device)
            base_features = model.backbone(samples)
            encoded = model.encoder([feature.clone() for feature in base_features])
            positive_indices = [index for index, target in enumerate(targets) if len(target["boxes"])]
            audit_index = positive_indices[0] if positive_indices and not shape_audit else None
            matcher_targets = None
            if any(int(target["image_id"].item()) in rows_by_image for target in targets):
                matcher_targets = _normalise_targets(
                    targets, samples.shape[-2], samples.shape[-1], device
                )

            for spec in specs:
                masked, masks = mask_encoder_outputs(
                    encoded,
                    targets,
                    samples.shape[-2:],
                    spec["background_weight"],
                    spec["context_weight"],
                    spec["context_scale"],
                )
                if audit_index is not None:
                    shape_audit.append(_audit_condition(
                        encoded, masks, targets, samples.shape[-2:], spec, audit_index
                    ))
                outputs = model.decoder(masked)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                if spec["name"] == "normal" and parity is None:
                    baseline = model(samples)
                    parity = {
                        "logits": float((outputs["pred_logits"] - baseline["pred_logits"]).abs().max()),
                        "boxes": float((outputs["pred_boxes"] - baseline["pred_boxes"]).abs().max()),
                    }
                    if max(parity.values()) > 2e-6:
                        raise RuntimeError(f"normal after-HE parity failed: {parity}")

                original_sizes = torch.stack([target["orig_size"] for target in targets]).to(device)
                results = solver.postprocessor(outputs, original_sizes)
                predictions = {
                    int(target["image_id"].item()): result
                    for target, result in zip(targets, results)
                }
                evaluators[spec["name"]].update(predictions)

                outputs_cpu = {
                    key: value.detach().cpu() for key, value in outputs.items()
                    if torch.is_tensor(value)
                }
                for batch_index, target in enumerate(targets):
                    image_id = int(target["image_id"].item())
                    if image_id not in rows_by_image:
                        continue
                    target_xyxy = box_cxcywh_to_xyxy(
                        matcher_targets[batch_index]["boxes"]
                    ).cpu()
                    boxes_xyxy = box_cxcywh_to_xyxy(
                        outputs_cpu["pred_boxes"][batch_index]
                    )
                    logits = outputs_cpu["pred_logits"][batch_index]
                    original_width, original_height = [int(value) for value in target["orig_size"]]
                    target_device = {
                        key: value.to(device) if torch.is_tensor(value) else value
                        for key, value in target.items()
                    }
                    for row in rows_by_image[image_id]:
                        gt_index = _target_index(
                            row, target_device, original_width, original_height
                        )
                        binding_records[int(row["ann_id"])]["conditions"][spec["name"]] = (
                            compact_binding_metrics(
                                boxes_xyxy,
                                logits,
                                target_xyxy[gt_index],
                                int(row["gt_class"]),
                            )
                        )

            seen += len(targets)
            if args.log_every and seen % args.log_every < len(targets):
                print(f"[{seen}/{len(cfg.val_dataloader.dataset)}] {time.time() - started:.1f}s", flush=True)

    if args.max_batches:
        smoke_payload = {
            "meta": {
                "checkpoint": args.checkpoint,
                "weights": args.weights,
                "n_images": seen,
                "normal_parity": parity,
                "shape_audit": shape_audit,
                "seconds": round(time.time() - started, 2),
            }
        }
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(smoke_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("SHAPE_AUDIT=" + json.dumps(shape_audit, ensure_ascii=False))
        return

    if args.expected_images and seen != args.expected_images:
        raise RuntimeError(f"evaluated {seen} images, expected {args.expected_images}")
    if any(len(record["conditions"]) != len(specs) for record in binding_records.values()):
        raise RuntimeError("not every manifest lesion has all after-HE conditions")

    after_rows = []
    details = {}
    for spec in specs:
        name = spec["name"]
        evaluator = evaluators[name]
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()
        coco_eval = evaluator.coco_eval["bbox"]
        macro = summarize_pr_curve_f1(coco_eval)
        yolo = summarize_yolo_pr_curve_metrics(coco_eval, evaluator.coco_gt)
        stats = {
            "coco_eval_bbox": coco_eval.stats.tolist(),
            "macro_f1_iou50": macro["f1_iou50"],
            **yolo,
        }
        after_rows.append(full_metrics_row(name, stats, seen))
        details[name] = stats

    before_payload = json.loads(Path(args.before_full_json).read_text())
    before_rows = [
        full_metrics_row(name, before_payload["details"][name], seen)
        for name in [spec["name"] for spec in specs]
    ]
    before_binding_payload = json.loads(Path(args.before_binding_json).read_text())
    after_binding_list = [binding_records[int(row["ann_id"])] for row in manifest_rows]
    condition_names = [spec["name"] for spec in specs]
    payload = {
        "meta": {
            "checkpoint": args.checkpoint,
            "weights": args.weights,
            "ann_file": args.ann_file,
            "n_images": seen,
            "n_binding_lesions": len(after_binding_list),
            "conditions": specs,
            "normal_parity": parity,
            "shape_audit": shape_audit,
            "seconds": round(time.time() - started, 2),
        },
        "after_rows": after_rows,
        "before_rows": before_rows,
        "details": details,
        "binding": {
            "before": summarize_binding(before_binding_payload["lesions"], condition_names),
            "after": summarize_binding(after_binding_list, condition_names),
            "after_records": after_binding_list,
        },
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_markdown(after_rows, before_rows)
    if args.markdown_out:
        Path(args.markdown_out).write_text(markdown, encoding="utf-8")
    print("SHAPE_AUDIT=" + json.dumps(shape_audit, ensure_ascii=False))
    print(markdown)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--img-folder", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--before-full-json", required=True)
    parser.add_argument("--before-binding-json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--markdown-out")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--context-alpha", type=float, default=0.5)
    parser.add_argument("--context-beta", type=float, default=0.2)
    parser.add_argument("--context-scale", type=float, default=1.5)
    parser.add_argument("--expected-images", type=int, default=2011)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--run-unit-tests", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.run_unit_tests:
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromName(
            "ecdetseg.tests.test_cmp5l_mask_after_hybridencoder"
        )
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)
    evaluate(cli_args)
