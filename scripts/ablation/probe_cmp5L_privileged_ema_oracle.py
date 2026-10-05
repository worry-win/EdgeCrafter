"""Oracle probe for GT-conditioned feature privilege before HybridEncoder."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision


EC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EC_ROOT / "ecdetseg"))
sys.path.insert(0, str(EC_ROOT))

from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy  # noqa: E402
from scripts.ablation.dump_cmp5L_internal_behavior import _target_index  # noqa: E402
from scripts.ablation.probe_cmp5L_decoder_internal_tensors import (  # noqa: E402
    DecoderInternalCapture,
    _build,
    _normalise_targets,
)


def _iou_matrix(boxes_a, boxes_b):
    if not len(boxes_a) or not len(boxes_b):
        return boxes_a.new_zeros((len(boxes_a), len(boxes_b)))
    left_top = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    right_bottom = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection = (right_bottom - left_top).clamp(min=0).prod(-1)
    area_a = (boxes_a[:, 2:] - boxes_a[:, :2]).clamp(min=0).prod(-1)
    area_b = (boxes_b[:, 2:] - boxes_b[:, :2]).clamp(min=0).prod(-1)
    return intersection / (area_a[:, None] + area_b[None, :] - intersection).clamp(min=1e-9)


def image_score_metrics(boxes_xyxy, logits, gt_boxes_xyxy):
    """Score-inflation and false-positive summaries for one image."""
    gt_boxes_xyxy = torch.as_tensor(
        gt_boxes_xyxy, device=boxes_xyxy.device, dtype=boxes_xyxy.dtype
    ).reshape(-1, 4)
    query_scores = logits.sigmoid().max(-1).values
    if len(gt_boxes_xyxy):
        max_iou = _iou_matrix(boxes_xyxy, gt_boxes_xyxy).max(-1).values
        positive = max_iou >= 0.5
        background = max_iou < 0.1
    else:
        positive = torch.zeros(len(boxes_xyxy), dtype=torch.bool, device=boxes_xyxy.device)
        background = torch.ones(len(boxes_xyxy), dtype=torch.bool, device=boxes_xyxy.device)

    def masked_mean(mask):
        return float(query_scores[mask].mean()) if bool(mask.any()) else None

    return {
        "all_logit_mean": float(logits.mean()),
        "all_logit_max": float(logits.max()),
        "all_query_score_mean": float(query_scores.mean()),
        "all_query_score_p95": float(torch.quantile(query_scores.float(), 0.95)),
        "top1_query_score": float(query_scores.max()),
        "positive_query_count": int(positive.sum()),
        "background_query_count": int(background.sum()),
        "positive_query_score_mean": masked_mean(positive),
        "background_query_score_mean": masked_mean(background),
        "background_fp_count_05": int((query_scores[background] >= 0.5).sum()),
        "background_fp_count_01": int((query_scores[background] >= 0.1).sum()),
    }


def lesion_query_metrics(boxes_xyxy, logits, gt_xyxy, gt_class, hungarian_query):
    """Compute binding and ranking endpoints for one GT lesion."""
    gt_xyxy = torch.as_tensor(gt_xyxy, device=boxes_xyxy.device, dtype=boxes_xyxy.dtype)
    iou = _iou_matrix(boxes_xyxy, gt_xyxy[None])[:, 0]
    gt_logits = logits[:, gt_class]
    final_scores = logits.sigmoid().max(-1).values
    q_iou = int(iou.argmax())
    q_cls = int(gt_logits.argmax())
    q_hungarian = int(hungarian_query)
    class_mask = torch.arange(logits.shape[-1], device=logits.device) != gt_class
    wrong_logit = logits[q_iou, class_mask].max()
    localized = iou >= 0.5
    best_localized_gt_logit = (
        gt_logits[localized].max() if bool(localized.any()) else gt_logits.new_tensor(float("-inf"))
    )
    final_order = torch.argsort(final_scores, descending=True)
    top1_iou = iou[final_order[:1]].max()
    top5_iou = iou[final_order[:min(5, len(final_order))]].max()
    return {
        "q_iou": q_iou,
        "q_cls": q_cls,
        "q_hungarian": q_hungarian,
        "q_iou_equals_q_cls": q_iou == q_cls,
        "q_iou_equals_q_hungarian": q_iou == q_hungarian,
        "q_cls_equals_q_hungarian": q_cls == q_hungarian,
        "max_iou": float(iou[q_iou]),
        "iou_q_cls": float(iou[q_cls]),
        "iou_q_hungarian": float(iou[q_hungarian]),
        "gt_logit_q_iou": float(gt_logits[q_iou]),
        "wrong_logit_q_iou": float(wrong_logit),
        "class_margin_q_iou": float(gt_logits[q_iou] - wrong_logit),
        "has_iou_05_candidate": bool(localized.any()),
        "best_localized_gt_logit": float(best_localized_gt_logit),
        "joint_success_iou05_score05": bool(best_localized_gt_logit >= 0.0),
        "final_score_q_iou": float(final_scores[q_iou]),
        "rank_q_iou_final_score": int((final_scores > final_scores[q_iou]).sum()) + 1,
        "rank_q_iou_gt_class": int((gt_logits > gt_logits[q_iou]).sum()) + 1,
        "top1_best_iou": float(top1_iou),
        "top5_best_iou": float(top5_iou),
        "top1_top5_iou_gap": float(top5_iou - top1_iou),
    }


def _boxes_tensor(target, device):
    boxes = target.get("boxes")
    if boxes is None:
        return torch.empty(0, 4, device=device)
    if hasattr(boxes, "as_subclass"):
        boxes = boxes.as_subclass(torch.Tensor)
    return torch.as_tensor(boxes, device=device)


def _paint_boxes(mask, boxes, image_hw, value, scale=1.0):
    image_h, image_w = image_hw
    feat_h, feat_w = mask.shape[-2:]
    for box in boxes:
        x0, y0, x1, y1 = box.float()
        if scale != 1.0:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            half_w, half_h = (x1 - x0) * scale / 2, (y1 - y0) * scale / 2
            x0, x1 = cx - half_w, cx + half_w
            y0, y1 = cy - half_h, cy + half_h
        ix0 = int(torch.floor(x0 * feat_w / image_w).clamp(0, feat_w).item())
        iy0 = int(torch.floor(y0 * feat_h / image_h).clamp(0, feat_h).item())
        ix1 = int(torch.ceil(x1 * feat_w / image_w).clamp(0, feat_w).item())
        iy1 = int(torch.ceil(y1 * feat_h / image_h).clamp(0, feat_h).item())
        if ix1 > ix0 and iy1 > iy0:
            mask[..., iy0:iy1, ix0:ix1] = torch.maximum(
                mask[..., iy0:iy1, ix0:ix1],
                mask.new_tensor(value),
            )


def privilege_features(
    features: Sequence[torch.Tensor],
    targets,
    image_hw: Tuple[int, int],
    background_weight: float,
    context_weight: Optional[float] = None,
    context_scale: float = 1.5,
):
    """Return masked feature clones and their `[B,1,H,W]` weight maps.

    Images without scored GT boxes always receive an all-one mask.
    """
    privileged, masks = [], []
    for feature in features:
        mask = feature.new_full(
            (feature.shape[0], 1, feature.shape[-2], feature.shape[-1]),
            background_weight,
        )
        for batch_index, target in enumerate(targets):
            boxes = _boxes_tensor(target, feature.device)
            if not len(boxes):
                mask[batch_index].fill_(1.0)
                continue
            if context_weight is not None:
                _paint_boxes(
                    mask[batch_index], boxes, image_hw,
                    context_weight, context_scale,
                )
            _paint_boxes(mask[batch_index], boxes, image_hw, 1.0)
        masks.append(mask)
        privileged.append(feature.clone() * mask)
    return privileged, masks


def _expanded_box(box, scale):
    center = (box[:2] + box[2:]) / 2
    half_size = (box[2:] - box[:2]) * scale / 2
    return torch.cat((center - half_size, center + half_size)).clamp(0, 1)


def auxiliary_query_metrics(layer_records, batch_index, query_id, gt_box, context_scale=1.5):
    """Compact per-layer sampling, attention, query, and FFN summaries."""
    gt_box = torch.as_tensor(gt_box, dtype=torch.float32)
    expanded = _expanded_box(gt_box, context_scale)
    result = {
        "sampling_inside": [],
        "attention_foreground_mass": [],
        "attention_context_mass": [],
        "query_update_norm": [],
        "query_norm": [],
        "ffn_activation_norm": [],
        "ffn_activation_entropy": [],
        "ffn_top10pct_energy": [],
    }
    for layer in layer_records:
        locations = layer["sampling_locations"][batch_index, query_id].float()
        weights = layer["attention_weights"][batch_index, query_id].float()
        x, y = locations[..., 0], locations[..., 1]
        inside = (
            (x >= gt_box[0]) & (x <= gt_box[2])
            & (y >= gt_box[1]) & (y <= gt_box[3])
        )
        in_expanded = (
            (x >= expanded[0]) & (x <= expanded[2])
            & (y >= expanded[1]) & (y <= expanded[3])
        )
        context = in_expanded & ~inside
        activation = layer["ffn_activation"][batch_index, query_id].float().abs()
        probability = activation / activation.sum().clamp(min=1e-9)
        entropy = -(probability * probability.clamp(min=1e-12).log()).sum()
        entropy = entropy / np.log(max(2, activation.numel()))
        energy = activation.square()
        top_count = max(1, activation.numel() // 10)
        result["sampling_inside"].append(float(inside.float().mean()))
        result["attention_foreground_mass"].append(float((weights * inside).sum(-1).mean()))
        result["attention_context_mass"].append(float((weights * context).sum(-1).mean()))
        result["query_update_norm"].append(float(torch.linalg.vector_norm(
            layer["query_out"][batch_index, query_id].float()
            - layer["query_in"][batch_index, query_id].float()
        )))
        result["query_norm"].append(float(torch.linalg.vector_norm(
            layer["query_out"][batch_index, query_id].float()
        )))
        result["ffn_activation_norm"].append(float(torch.linalg.vector_norm(activation)))
        result["ffn_activation_entropy"].append(float(entropy))
        result["ffn_top10pct_energy"].append(float(
            torch.topk(energy, top_count).values.sum() / energy.sum().clamp(min=1e-9)
        ))
    return result


def _query_at_id_metrics(boxes_xyxy, logits, gt_box, gt_class, query_id):
    iou = _iou_matrix(boxes_xyxy, gt_box[None])[:, 0]
    gt_logits = logits[:, gt_class]
    final_scores = logits.sigmoid().max(-1).values
    class_mask = torch.arange(logits.shape[-1], device=logits.device) != gt_class
    wrong_logit = logits[query_id, class_mask].max()
    return {
        "normal_q_iou_id": int(query_id),
        "normal_q_iou_current_iou": float(iou[query_id]),
        "normal_q_iou_gt_logit": float(gt_logits[query_id]),
        "normal_q_iou_wrong_logit": float(wrong_logit),
        "normal_q_iou_class_margin": float(gt_logits[query_id] - wrong_logit),
        "normal_q_iou_final_rank": int((final_scores > final_scores[query_id]).sum()) + 1,
        "normal_q_iou_gt_class_rank": int((gt_logits > gt_logits[query_id]).sum()) + 1,
    }


def _condition_specs(context_alpha, context_beta, context_scale):
    return [
        {"name": "normal", "background_weight": 1.0, "context_weight": None, "context_scale": context_scale},
        {"name": "mild", "background_weight": 0.5, "context_weight": None, "context_scale": context_scale},
        {"name": "strong", "background_weight": 0.2, "context_weight": None, "context_scale": context_scale},
        {"name": "hard", "background_weight": 0.0, "context_weight": None, "context_scale": context_scale},
        {"name": "context", "background_weight": context_beta, "context_weight": context_alpha, "context_scale": context_scale},
    ]


def _forward_condition(model, base_features, targets, image_hw, spec):
    features, masks = privilege_features(
        base_features,
        targets,
        image_hw,
        spec["background_weight"],
        spec["context_weight"],
        spec["context_scale"],
    )
    encoded = model.encoder(features)
    with DecoderInternalCapture(model.decoder) as capture:
        outputs = model.decoder(encoded)
    if isinstance(outputs, (tuple, list)):
        outputs = outputs[0]
    return outputs, capture.cpu_snapshot(), [mask.detach().cpu() for mask in masks]


def collect(args):
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; refusing silent CPU fallback")
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    if args.lesion_limit:
        rows = rows[:args.lesion_limit]
    rows_by_image = {}
    lesion_records = {}
    for row in rows:
        rows_by_image.setdefault(int(row["image_id"]), []).append(row)
        lesion_records[int(row["ann_id"])] = {
            "ann_id": int(row["ann_id"]),
            "image_id": int(row["image_id"]),
            "gt_class": int(row["gt_class"]),
            "class_name": row["class_name"],
            "group": row["group"],
            "conditions": {},
        }

    cfg, solver, model = _build(args)
    model.eval()
    specs = _condition_specs(args.context_alpha, args.context_beta, args.context_scale)
    normal_qids = {}
    image_records = []
    negative_ids = set()
    normal_parity = None
    negative_parity = {spec["name"]: {"logits": 0.0, "boxes": 0.0} for spec in specs[1:]}
    started = time.time()

    for samples, targets in cfg.val_dataloader:
        image_ids = [int(target["image_id"].item()) for target in targets]
        negative_batch = [
            len(target["boxes"]) == 0 and args.expected_negatives > 0
            for target in targets
        ]
        if not any(image_id in rows_by_image for image_id in image_ids) and not any(negative_batch):
            continue
        samples = samples.to(args.device)
        matcher_targets = _normalise_targets(
            targets, samples.shape[-2], samples.shape[-1], args.device
        )
        with torch.no_grad():
            base_features = model.backbone(samples)
            baseline = model(samples) if normal_parity is None else None
            normal_outputs_cpu = None
            for spec in specs:
                outputs, snapshot, masks = _forward_condition(
                    model, base_features, targets, samples.shape[-2:], spec
                )
                outputs_cpu = {
                    key: value.detach().cpu() for key, value in outputs.items()
                    if torch.is_tensor(value)
                }
                if spec["name"] == "normal":
                    normal_outputs_cpu = outputs_cpu
                    if baseline is not None:
                        normal_parity = {
                            "logits": float((baseline["pred_logits"].cpu() - outputs_cpu["pred_logits"]).abs().max()),
                            "boxes": float((baseline["pred_boxes"].cpu() - outputs_cpu["pred_boxes"]).abs().max()),
                        }
                        if max(normal_parity.values()) > 2e-6:
                            raise RuntimeError(f"normal composed forward parity failed: {normal_parity}")
                matches = solver.criterion.matcher(outputs, matcher_targets)["indices"]

                for batch_index, (image_id, target, is_negative) in enumerate(
                    zip(image_ids, targets, negative_batch)
                ):
                    if image_id not in rows_by_image and not is_negative:
                        continue
                    target_xyxy = box_cxcywh_to_xyxy(matcher_targets[batch_index]["boxes"]).cpu()
                    boxes_xyxy = box_cxcywh_to_xyxy(outputs_cpu["pred_boxes"][batch_index])
                    logits = outputs_cpu["pred_logits"][batch_index]
                    image_metric = image_score_metrics(boxes_xyxy, logits, target_xyxy)
                    image_metric.update({
                        "image_id": image_id,
                        "condition": spec["name"],
                        "is_negative": bool(is_negative),
                        "feature_mask_mean": [float(mask[batch_index].mean()) for mask in masks],
                        "feature_foreground_fraction": [float((mask[batch_index] == 1).float().mean()) for mask in masks],
                    })
                    image_records.append(image_metric)
                    if is_negative:
                        negative_ids.add(image_id)
                        if spec["name"] != "normal":
                            negative_parity[spec["name"]]["logits"] = max(
                                negative_parity[spec["name"]]["logits"],
                                float((logits - normal_outputs_cpu["pred_logits"][batch_index]).abs().max()),
                            )
                            negative_parity[spec["name"]]["boxes"] = max(
                                negative_parity[spec["name"]]["boxes"],
                                float((outputs_cpu["pred_boxes"][batch_index]
                                       - normal_outputs_cpu["pred_boxes"][batch_index]).abs().max()),
                            )
                        continue

                    original_width, original_height = [int(v) for v in target["orig_size"].tolist()]
                    target_device = {
                        key: value.to(args.device) if torch.is_tensor(value) else value
                        for key, value in target.items()
                    }
                    hungarian = {int(gt): int(query) for query, gt in zip(*matches[batch_index])}
                    for row in rows_by_image.get(image_id, []):
                        gt_index = _target_index(row, target_device, original_width, original_height)
                        gt_box = target_xyxy[gt_index]
                        metrics = lesion_query_metrics(
                            boxes_xyxy, logits, gt_box, int(row["gt_class"]), hungarian[gt_index]
                        )
                        ann_id = int(row["ann_id"])
                        if spec["name"] == "normal":
                            normal_qids[ann_id] = metrics["q_iou"]
                        metrics.update(_query_at_id_metrics(
                            boxes_xyxy, logits, gt_box, int(row["gt_class"]), normal_qids[ann_id]
                        ))
                        metrics["auxiliary_q_iou"] = auxiliary_query_metrics(
                            snapshot["layers"], batch_index, metrics["q_iou"], gt_box,
                            args.context_scale,
                        )
                        lesion_records[ann_id]["conditions"][spec["name"]] = metrics

    if len(negative_ids) != args.expected_negatives:
        raise RuntimeError(f"collected {len(negative_ids)} negative images, expected {args.expected_negatives}")
    if any(len(record["conditions"]) != len(specs) for record in lesion_records.values()):
        raise RuntimeError("not every lesion has every privilege condition")
    if any(max(values.values()) > 2e-6 for values in negative_parity.values()):
        raise RuntimeError(f"negative-image privilege changed outputs: {negative_parity}")

    payload = {
        "meta": {
            "checkpoint": args.checkpoint,
            "split": args.split,
            "n_lesions": len(lesion_records),
            "n_negative_images": len(negative_ids),
            "negative_image_ids": sorted(negative_ids),
            "conditions": specs,
            "normal_parity": normal_parity,
            "negative_parity": negative_parity,
            "seconds": round(time.time() - started, 2),
        },
        "lesions": [lesion_records[int(row["ann_id"])] for row in rows],
        "images": image_records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["meta"], ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--img-folder", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", default="ema")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--context-alpha", type=float, default=0.5)
    parser.add_argument("--context-beta", type=float, default=0.2)
    parser.add_argument("--context-scale", type=float, default=1.5)
    parser.add_argument("--expected-negatives", type=int, default=8)
    parser.add_argument("--lesion-limit", type=int, default=0)
    return parser.parse_args()


def main():
    collect(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
