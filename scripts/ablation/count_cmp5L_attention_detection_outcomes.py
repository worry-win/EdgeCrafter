"""Count fixed-threshold TP/FP/FN for selected attention interventions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchvision


EC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EC_ROOT / "ecdetseg"))
sys.path.insert(0, str(EC_ROOT))

from scripts.ablation.probe_cmp5L_decoder_internal_tensors import _build, _normalise_targets  # noqa: E402
from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy  # noqa: E402


def _pairwise_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((len(boxes1), len(boxes2)))
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(-1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0).prod(-1)[:, None]
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0).prod(-1)[None, :]
    return intersection / (area1 + area2 - intersection).clamp_min(1e-9)


def greedy_detection_counts(
    pred_boxes_xyxy,
    pred_logits,
    gt_boxes_xyxy,
    gt_labels,
    *,
    score_threshold=0.5,
    iou_threshold=0.5,
):
    scores, labels = pred_logits.sigmoid().max(-1)
    keep = scores >= score_threshold
    boxes = pred_boxes_xyxy[keep]
    scores = scores[keep]
    labels = labels[keep]
    order = torch.argsort(scores, descending=True)
    matched = torch.zeros(len(gt_boxes_xyxy), dtype=torch.bool, device=gt_boxes_xyxy.device)
    tp = fp = 0
    for prediction in order:
        candidates = torch.nonzero((gt_labels == labels[prediction]) & ~matched, as_tuple=False).flatten()
        if len(candidates) == 0:
            fp += 1
            continue
        iou = _pairwise_iou(boxes[prediction:prediction + 1], gt_boxes_xyxy[candidates])[0]
        best_value, best_offset = iou.max(0)
        if float(best_value) >= iou_threshold:
            matched[candidates[best_offset]] = True
            tp += 1
        else:
            fp += 1
    return {"tp": tp, "fp": fp, "fn": int((~matched).sum())}


def evaluate(args):
    from scripts.ablation.evaluate_cmp5L_gt_guided_attention import GTUnionAttentionController
    from scripts.ablation.evaluate_cmp5L_l1_box_guided_attention import L1BoxAttentionController
    from scripts.ablation.evaluate_cmp5L_oracle_gated_l1_attention import OracleGatedL1Controller

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; refusing silent CPU fallback")
    args.device = device
    cfg, _solver, model = _build(args)
    totals = {"tp": 0, "fp": 0, "fn": 0}
    seen = 0
    with torch.no_grad():
        for samples, targets in cfg.val_dataloader:
            samples = samples.to(device)
            normalized = _normalise_targets(targets, samples.shape[-2], samples.shape[-1], device)
            gt_boxes = [box_cxcywh_to_xyxy(target["boxes"]) for target in normalized]
            gt_labels = [target["labels"] for target in normalized]
            if args.mode == "baseline":
                outputs = model(samples)
            elif args.mode == "ungated_hard_l2_l3":
                controller = L1BoxAttentionController(
                    model.decoder, (1, 2), box_weight=1.0, ring_weight=0.0,
                    background_weight=0.0, ring_scale=1.5,
                )
                with controller:
                    outputs = model(samples)
            elif args.mode == "gt_soft_l2_l3":
                controller = GTUnionAttentionController(
                    model.decoder, (1, 2), box_weight=1.0, ring_weight=0.5,
                    background_weight=0.2, ring_scale=1.5,
                )
                controller.set_gt_boxes(gt_boxes)
                with controller:
                    outputs = model(samples)
            elif args.mode in {"gateA_hard_l2_l3", "gateB_hard_l2_l3"}:
                gate = "localization" if args.mode.startswith("gateA") else "localization_class"
                controller = OracleGatedL1Controller(
                    model.decoder, (1, 2), gate=gate, box_weight=1.0,
                    ring_weight=0.0, background_weight=0.0, ring_scale=1.5,
                )
                controller.set_targets(gt_boxes, gt_labels)
                with controller:
                    outputs = model(samples)
            else:
                raise ValueError(args.mode)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            pred_boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"])
            for batch_index in range(len(targets)):
                counts = greedy_detection_counts(
                    pred_boxes[batch_index],
                    outputs["pred_logits"][batch_index],
                    gt_boxes[batch_index],
                    gt_labels[batch_index],
                    score_threshold=args.score_threshold,
                    iou_threshold=0.5,
                )
                for key in totals:
                    totals[key] += counts[key]
            seen += len(targets)

    precision = totals["tp"] / max(1, totals["tp"] + totals["fp"])
    recall = totals["tp"] / max(1, totals["tp"] + totals["fn"])
    result = {
        "mode": args.mode,
        "n_images": seen,
        "score_threshold": args.score_threshold,
        "iou_threshold": 0.5,
        **totals,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "checkpoint": args.checkpoint,
        "checkpoint_weights": args.weights,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
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
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument(
        "--mode",
        choices=("baseline", "ungated_hard_l2_l3", "gt_soft_l2_l3", "gateA_hard_l2_l3", "gateB_hard_l2_l3"),
        required=True,
    )
    evaluate(parser.parse_args())
