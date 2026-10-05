"""Inference-only L1-box guidance for later decoder deformable attention.

The intervention is intentionally isolated in this evaluator.  It patches only
the post-softmax attention weights of actual decoder layers 1 and/or 2 (user
L2/L3), while using the normalized box entering actual layer 1 as the fixed L1
prediction for every query.  References, offsets, values, anchors, spatial
shapes, positional inputs, and the final L4 layer are unchanged.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import types
from contextlib import AbstractContextManager
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision


EC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EC_ROOT / "ecdetseg"))
sys.path.insert(0, str(EC_ROOT))

from scripts.ablation.dump_cmp5L_internal_behavior import _target_index  # noqa: E402
from scripts.ablation.probe_cmp5L_decoder_internal_tensors import (  # noqa: E402
    _build,
    _normalise_targets,
)
from scripts.ablation.probe_cmp5L_privileged_ema_oracle import _iou_matrix  # noqa: E402
from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy  # noqa: E402
from engine.solver.ec_engine import (  # noqa: E402
    summarize_pr_curve_f1,
    summarize_yolo_pr_curve_metrics,
)


def layer_indices(name):
    """Map the paper-facing layer names onto actual zero-based indices."""
    mapping = {"l2": (1,), "l3": (2,), "l2_l3": (1, 2)}
    if name not in mapping:
        raise ValueError(f"unknown layer selection: {name}")
    return mapping[name]


def _cxcywh_to_xyxy(boxes):
    center = boxes[..., :2]
    half = boxes[..., 2:] / 2
    return torch.cat((center - half, center + half), dim=-1)


def apply_box_attention_prior(
    attention_weights,
    sampling_locations,
    l1_boxes_cxcywh,
    *,
    box_weight,
    ring_weight,
    background_weight,
    ring_scale=1.5,
):
    """Apply a per-query box/ring/background prior and renormalize per head.

    Hard priors can encounter a head with no point inside the predicted box.
    Such heads fall back to their original normalized weights rather than
    producing an all-zero/NaN attention output.
    """
    if box_weight == ring_weight == background_weight == 1.0:
        return attention_weights, torch.ones_like(attention_weights)

    boxes = _cxcywh_to_xyxy(l1_boxes_cxcywh).clamp(0, 1)
    centers = l1_boxes_cxcywh[..., :2]
    half = l1_boxes_cxcywh[..., 2:] * ring_scale / 2
    expanded = torch.cat((centers - half, centers + half), dim=-1).clamp(0, 1)
    x = sampling_locations[..., 0]
    y = sampling_locations[..., 1]
    box = boxes[:, :, None, None, :]
    ring_box = expanded[:, :, None, None, :]
    inside = (x >= box[..., 0]) & (x <= box[..., 2]) & (y >= box[..., 1]) & (y <= box[..., 3])
    in_expanded = (
        (x >= ring_box[..., 0]) & (x <= ring_box[..., 2])
        & (y >= ring_box[..., 1]) & (y <= ring_box[..., 3])
    )
    prior = torch.full_like(attention_weights, float(background_weight))
    prior = torch.where(in_expanded, prior.new_tensor(float(ring_weight)), prior)
    prior = torch.where(inside, prior.new_tensor(float(box_weight)), prior)
    numerator = attention_weights * prior
    denominator = numerator.sum(-1, keepdim=True)
    guided = numerator / denominator.clamp_min(torch.finfo(numerator.dtype).tiny)
    guided = torch.where(denominator > 0, guided, attention_weights)
    return guided, prior


class L1BoxAttentionController(AbstractContextManager):
    """Temporarily patch only L2/L3 deformable-attention weight consumption."""

    def __init__(
        self,
        ec_transformer,
        selected_layers,
        *,
        box_weight,
        ring_weight,
        background_weight,
        ring_scale=1.5,
    ):
        self.decoder = ec_transformer.decoder
        self.selected_layers = set(selected_layers)
        if not self.selected_layers.issubset({1, 2}):
            raise ValueError("only actual layers 1/2 (user L2/L3) may be guided")
        self.box_weight = box_weight
        self.ring_weight = ring_weight
        self.background_weight = background_weight
        self.ring_scale = ring_scale
        self.l1_boxes = None
        self.records = {}
        self._originals = {}
        self._pre_handle = None

    def _reset(self, _module, _inputs):
        self.l1_boxes = None
        self.records.clear()

    def _patched_forward(self, index, module):
        def forward(_module, query, reference_points, value, value_spatial_shapes):
            batch, queries = query.shape[:2]
            offsets = module.sampling_offsets(query).reshape(
                batch, queries, module.num_heads, sum(module.num_points_list), 2
            )
            weight_logits = module.attention_weights(query).reshape(
                batch, queries, module.num_heads, sum(module.num_points_list)
            )
            weights = F.softmax(weight_logits, dim=-1)
            if reference_points.shape[-1] == 4:
                point_scale = module.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
                offset = offsets * point_scale * reference_points[:, :, None, :, 2:] * module.offset_scale
                locations = reference_points[:, :, None, :, :2] + offset
            elif reference_points.shape[-1] == 2:
                normalizer = torch.as_tensor(
                    value_spatial_shapes, device=query.device, dtype=query.dtype
                ).flip([1]).reshape(1, 1, 1, module.num_levels, 1, 2)
                shaped = offsets.reshape(
                    batch, queries, module.num_heads, module.num_levels, -1, 2
                )
                locations = reference_points.reshape(
                    batch, queries, 1, module.num_levels, 1, 2
                ) + shaped / normalizer
                locations = locations.flatten(3, 4)
            else:
                raise ValueError("reference point width must be 2 or 4")

            # Layer 1 receives exactly the box predicted after layer 0 (user L1).
            if index == 1:
                self.l1_boxes = reference_points[:, :, 0, :].detach()
            if self.l1_boxes is None:
                raise RuntimeError("L1 boxes were not captured before guided attention")

            if index in self.selected_layers:
                used_weights, prior = apply_box_attention_prior(
                    weights,
                    locations,
                    self.l1_boxes,
                    box_weight=self.box_weight,
                    ring_weight=self.ring_weight,
                    background_weight=self.background_weight,
                    ring_scale=self.ring_scale,
                )
            else:
                used_weights, prior = weights, torch.ones_like(weights)
            self.records[index] = {
                "sampling_locations": locations.detach(),
                "attention_weights": used_weights.detach(),
                "prior": prior.detach(),
                "reference_points": reference_points.detach(),
            }
            return module.ms_deformable_attn_core(
                value,
                value_spatial_shapes,
                locations,
                used_weights,
                module.num_points_list,
            )
        return types.MethodType(forward, module)

    def __enter__(self):
        if self.decoder.training:
            raise RuntimeError("L1BoxAttentionController requires eval mode")
        self._pre_handle = self.decoder.register_forward_pre_hook(self._reset)
        # Patch L2 for L1-box capture even when only L3 is guided; patch L3 for
        # requested analysis. L1 and L4 remain byte-for-byte on their class forward.
        for index in (1, 2):
            module = self.decoder.layers[index].cross_attn
            self._originals[index] = ("forward" in module.__dict__, module.__dict__.get("forward"))
            module.forward = self._patched_forward(index, module)
        return self

    def __exit__(self, exc_type, exc, traceback):
        for index, (had_instance_forward, original) in self._originals.items():
            module = self.decoder.layers[index].cross_attn
            if had_instance_forward:
                module.forward = original
            else:
                del module.forward
        self._originals.clear()
        if self._pre_handle is not None:
            self._pre_handle.remove()
            self._pre_handle = None
        return False


def _binding_metrics(boxes_xyxy, logits, gt_xyxy, gt_class):
    gt_xyxy = torch.as_tensor(gt_xyxy, dtype=boxes_xyxy.dtype)
    iou = _iou_matrix(boxes_xyxy, gt_xyxy[None])[:, 0]
    gt_logits = logits[:, gt_class]
    class_mask = torch.arange(logits.shape[-1]) != gt_class
    wrong_logits = logits[:, class_mask].max(-1).values
    final_scores = logits.sigmoid().max(-1).values
    q_iou = int(iou.argmax())
    q_cls = int(gt_logits.argmax())
    localized = iou >= 0.5
    q_detection = int(torch.where(localized, gt_logits, gt_logits.new_full((), float("-inf"))).argmax()) if bool(localized.any()) else -1
    best_localized = float(gt_logits[q_detection]) if q_detection >= 0 else float("-inf")
    order = torch.argsort(final_scores, descending=True)
    return {
        "q_iou": q_iou,
        "q_cls": q_cls,
        "q_detection": q_detection,
        "q_iou_equals_q_cls": q_iou == q_cls,
        "gt_logit_q_iou": float(gt_logits[q_iou]),
        "iou_q_cls": float(iou[q_cls]),
        "rank_q_iou_final_score": int((final_scores > final_scores[q_iou]).sum()) + 1,
        "top1_top5_iou_gap": float(iou[order[:min(5, len(order))]].max() - iou[order[0]]),
        "class_margin_q_iou": float(gt_logits[q_iou] - wrong_logits[q_iou]),
        "joint_success_iou05_score05": best_localized >= 0.0,
        "query_iou": [float(value) for value in iou],
        "query_gt_logit": [float(value) for value in gt_logits],
        "query_wrong_logit": [float(value) for value in wrong_logits],
        "query_final_score": [float(value) for value in final_scores],
    }


def _attention_mass(records, batch_index, gt_xyxy, context_scale=1.5):
    box = torch.as_tensor(gt_xyxy, device=next(iter(records.values()))["sampling_locations"].device)
    center = (box[:2] + box[2:]) / 2
    half = (box[2:] - box[:2]) * context_scale / 2
    expanded = torch.cat((center - half, center + half)).clamp(0, 1)
    result = {}
    for index in (1, 2):
        locations = records[index]["sampling_locations"][batch_index]
        weights = records[index]["attention_weights"][batch_index]
        x, y = locations[..., 0], locations[..., 1]
        inside = (x >= box[0]) & (x <= box[2]) & (y >= box[1]) & (y <= box[3])
        in_expanded = (
            (x >= expanded[0]) & (x <= expanded[2])
            & (y >= expanded[1]) & (y <= expanded[3])
        )
        context = in_expanded & ~inside
        result[f"l{index + 1}_foreground_mass"] = [
            float(value) for value in (weights * inside).sum(-1).mean(-1)
        ]
        result[f"l{index + 1}_context_mass"] = [
            float(value) for value in (weights * context).sum(-1).mean(-1)
        ]
    return result


def _full_metrics_row(condition, stats, n_images):
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
    }


def evaluate(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; refusing silent CPU fallback")
    args.device = device
    manifest_rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    rows_by_image = {}
    lesion_records = {}
    for row in manifest_rows:
        rows_by_image.setdefault(int(row["image_id"]), []).append(row)
        lesion_records[int(row["ann_id"])] = {
            key: row[key] for key in ("ann_id", "image_id", "gt_class", "class_name", "group")
        }

    cfg, solver, model = _build(args)
    evaluator = copy.deepcopy(cfg.evaluator)
    evaluator.cleanup()
    selected = layer_indices(args.layers)
    prior_values = {
        "baseline": (1.0, 1.0, 1.0, ()),
        "identity": (1.0, 1.0, 1.0, selected),
        "soft": (1.0, args.ring_weight, args.background_weight, selected),
        "hard": (1.0, 0.0, 0.0, selected),
    }
    box_weight, ring_weight, background_weight, guided_layers = prior_values[args.prior]
    condition = "baseline" if args.prior == "baseline" else f"{args.prior}_{args.layers}"
    controller = L1BoxAttentionController(
        model.decoder,
        guided_layers,
        box_weight=box_weight,
        ring_weight=ring_weight,
        background_weight=background_weight,
        ring_scale=args.ring_scale,
    )

    parity = None
    seen = 0
    started = time.time()
    with torch.no_grad():
        for batch_number, (samples, targets) in enumerate(cfg.val_dataloader):
            if args.max_batches and batch_number >= args.max_batches:
                break
            samples = samples.to(device)
            if args.parity:
                baseline = model(samples)
                if isinstance(baseline, (tuple, list)):
                    baseline = baseline[0]
            with controller:
                outputs = model(samples)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            if args.parity and parity is None:
                parity = {
                    "logits": float((outputs["pred_logits"] - baseline["pred_logits"]).abs().max()),
                    "boxes": float((outputs["pred_boxes"] - baseline["pred_boxes"]).abs().max()),
                }
                if max(parity.values()) > 2e-6:
                    raise RuntimeError(f"all-one attention parity failed: {parity}")

            original_sizes = torch.stack([target["orig_size"] for target in targets]).to(device)
            results = solver.postprocessor(outputs, original_sizes)
            evaluator.update({
                int(target["image_id"].item()): result
                for target, result in zip(targets, results)
            })

            if any(int(target["image_id"].item()) in rows_by_image for target in targets):
                matcher_targets = _normalise_targets(
                    targets, samples.shape[-2], samples.shape[-1], device
                )
                boxes_cpu = box_cxcywh_to_xyxy(outputs["pred_boxes"].detach().cpu())
                logits_cpu = outputs["pred_logits"].detach().cpu()
                for batch_index, target in enumerate(targets):
                    image_id = int(target["image_id"].item())
                    if image_id not in rows_by_image:
                        continue
                    target_xyxy = box_cxcywh_to_xyxy(matcher_targets[batch_index]["boxes"]).cpu()
                    original_width, original_height = [int(value) for value in target["orig_size"]]
                    target_device = {
                        key: value.to(device) if torch.is_tensor(value) else value
                        for key, value in target.items()
                    }
                    for row in rows_by_image[image_id]:
                        gt_index = _target_index(row, target_device, original_width, original_height)
                        gt_box = target_xyxy[gt_index]
                        record = _binding_metrics(
                            boxes_cpu[batch_index], logits_cpu[batch_index], gt_box,
                            int(row["gt_class"]),
                        )
                        record.update(_attention_mass(controller.records, batch_index, gt_box))
                        lesion_records[int(row["ann_id"])]["metrics"] = record

            seen += len(targets)
            if args.log_every and seen % args.log_every < len(targets):
                print(f"[{condition} {seen}/{len(cfg.val_dataloader.dataset)}] {time.time() - started:.1f}s", flush=True)

    if args.expected_images and not args.max_batches and seen != args.expected_images:
        raise RuntimeError(f"evaluated {seen} images, expected {args.expected_images}")
    if not args.max_batches and any("metrics" not in record for record in lesion_records.values()):
        raise RuntimeError("not every manifest lesion was evaluated")

    payload = {
        "meta": {
            "condition": condition,
            "prior": args.prior,
            "layers": args.layers,
            "actual_layer_indices": list(guided_layers),
            "l1_source": "actual layer-0 output box entering actual layer 1",
            "ring_scale": args.ring_scale,
            "weights": {"box": box_weight, "ring": ring_weight, "background": background_weight},
            "checkpoint": args.checkpoint,
            "checkpoint_weights": args.weights,
            "n_images": seen,
            "n_lesions": len(lesion_records),
            "parity": parity,
            "hard_empty_head_policy": "fall back to original normalized weights",
            "seconds": round(time.time() - started, 2),
        },
        "lesions": list(lesion_records.values()),
    }
    if not args.max_batches:
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()
        coco_eval = evaluator.coco_eval["bbox"]
        macro = summarize_pr_curve_f1(coco_eval)
        yolo = summarize_yolo_pr_curve_metrics(coco_eval, evaluator.coco_gt)
        details = {
            "coco_eval_bbox": coco_eval.stats.tolist(),
            "macro_f1_iou50": macro["f1_iou50"],
            **yolo,
        }
        payload["row"] = _full_metrics_row(condition, details, seen)
        payload["details"] = details
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"meta": payload["meta"], "row": payload.get("row")}, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--img-folder", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prior", choices=("baseline", "identity", "soft", "hard"), required=True)
    parser.add_argument("--layers", choices=("l2", "l3", "l2_l3"), default="l2_l3")
    parser.add_argument("--ring-weight", type=float, default=0.5)
    parser.add_argument("--background-weight", type=float, default=0.2)
    parser.add_argument("--ring-scale", type=float, default=1.5)
    parser.add_argument("--expected-images", type=int, default=2011)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--run-unit-tests", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.run_unit_tests:
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromName(
            "ecdetseg.tests.test_cmp5l_l1_box_guided_attention"
        )
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)
    evaluate(cli_args)
