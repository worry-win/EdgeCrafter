"""Inference-only GT-union guidance for decoder deformable attention."""

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
from scripts.ablation.evaluate_cmp5L_l1_box_guided_attention import (  # noqa: E402
    _attention_mass,
    _binding_metrics,
    _full_metrics_row,
    layer_indices,
)
from scripts.ablation.probe_cmp5L_decoder_internal_tensors import (  # noqa: E402
    _build,
    _normalise_targets,
)
from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy  # noqa: E402
from engine.solver.ec_engine import (  # noqa: E402
    summarize_pr_curve_f1,
    summarize_yolo_pr_curve_metrics,
)


def apply_gt_union_attention_prior(
    attention_weights,
    sampling_locations,
    gt_boxes_xyxy,
    *,
    box_weight,
    ring_weight,
    background_weight,
    ring_scale=1.5,
):
    """Reweight each query/head using the union of every GT region per image."""
    if box_weight == ring_weight == background_weight == 1.0:
        return attention_weights, torch.ones_like(attention_weights)

    prior = torch.empty_like(attention_weights)
    for batch_index, boxes in enumerate(gt_boxes_xyxy):
        boxes = torch.as_tensor(
            boxes, device=sampling_locations.device, dtype=sampling_locations.dtype
        )
        if boxes.numel() == 0:
            prior[batch_index].fill_(1.0)
            continue
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        half = (boxes[:, 2:] - boxes[:, :2]) * ring_scale / 2
        expanded = torch.cat((centers - half, centers + half), dim=-1).clamp(0, 1)
        locations = sampling_locations[batch_index]
        x = locations[..., 0, None]
        y = locations[..., 1, None]
        boxes_view = boxes[None, None, None, :, :]
        expanded_view = expanded[None, None, None, :, :]
        inside = (
            (x >= boxes_view[..., 0]) & (x <= boxes_view[..., 2])
            & (y >= boxes_view[..., 1]) & (y <= boxes_view[..., 3])
        ).any(-1)
        in_expanded = (
            (x >= expanded_view[..., 0]) & (x <= expanded_view[..., 2])
            & (y >= expanded_view[..., 1]) & (y <= expanded_view[..., 3])
        ).any(-1)
        item_prior = torch.full_like(attention_weights[batch_index], float(background_weight))
        item_prior = torch.where(in_expanded, item_prior.new_tensor(float(ring_weight)), item_prior)
        prior[batch_index] = torch.where(inside, item_prior.new_tensor(float(box_weight)), item_prior)

    numerator = attention_weights * prior
    denominator = numerator.sum(-1, keepdim=True)
    guided = numerator / denominator.clamp_min(torch.finfo(numerator.dtype).tiny)
    guided = torch.where(denominator > 0, guided, attention_weights)
    return guided, prior


class GTUnionAttentionController(AbstractContextManager):
    """Patch only L2/L3 attention weights using per-image GT-union regions."""

    def __init__(
        self,
        ec_transformer,
        selected_layers,
        *,
        box_weight,
        ring_weight,
        background_weight,
        ring_scale=1.5,
        capture_invariants=False,
    ):
        self.decoder = ec_transformer.decoder
        self.selected_layers = set(selected_layers)
        if not self.selected_layers.issubset({1, 2}):
            raise ValueError("only actual layers 1/2 (user L2/L3) may be guided")
        self.box_weight = box_weight
        self.ring_weight = ring_weight
        self.background_weight = background_weight
        self.ring_scale = ring_scale
        self.capture_invariants = capture_invariants
        self.gt_boxes = None
        self.records = {}
        self._originals = {}
        self._pre_handle = None

    def set_gt_boxes(self, gt_boxes):
        self.gt_boxes = gt_boxes

    def _reset(self, _module, _inputs):
        self.records.clear()
        if self.gt_boxes is None:
            raise RuntimeError("GT boxes must be set before decoder forward")

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

            if index in self.selected_layers:
                used_weights, prior = apply_gt_union_attention_prior(
                    weights,
                    locations,
                    self.gt_boxes,
                    box_weight=self.box_weight,
                    ring_weight=self.ring_weight,
                    background_weight=self.background_weight,
                    ring_scale=self.ring_scale,
                )
            else:
                used_weights, prior = weights, torch.ones_like(weights)
            record = {
                "sampling_locations": locations.detach(),
                "attention_weights": used_weights.detach(),
                "prior": prior.detach(),
                "reference_points": reference_points.detach(),
            }
            if self.capture_invariants:
                record["sampling_offsets"] = offsets.detach().clone()
                record["value"] = tuple(item.detach().clone() for item in value)
            self.records[index] = record
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
            raise RuntimeError("GTUnionAttentionController requires eval mode")
        self._pre_handle = self.decoder.register_forward_pre_hook(self._reset)
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


def _max_abs(left, right):
    return float((left - right).abs().max())


def _invariant_audit(baseline_records, identity_records):
    result = {}
    for index in (1, 2):
        base = baseline_records[index]
        identity = identity_records[index]
        result[f"l{index + 1}"] = {
            "sampling_offsets": _max_abs(base["sampling_offsets"], identity["sampling_offsets"]),
            "sampling_locations": _max_abs(base["sampling_locations"], identity["sampling_locations"]),
            "reference_points": _max_abs(base["reference_points"], identity["reference_points"]),
            "feature_value": max(
                _max_abs(a, b) for a, b in zip(base["value"], identity["value"])
            ),
            "attention_weights": _max_abs(base["attention_weights"], identity["attention_weights"]),
        }
    return result


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
    condition = "baseline" if args.prior == "baseline" else f"gt_{args.prior}_{args.layers}"
    controller = GTUnionAttentionController(
        model.decoder,
        guided_layers,
        box_weight=box_weight,
        ring_weight=ring_weight,
        background_weight=background_weight,
        ring_scale=args.ring_scale,
        capture_invariants=args.parity,
    )
    baseline_controller = GTUnionAttentionController(
        model.decoder,
        (),
        box_weight=1.0,
        ring_weight=1.0,
        background_weight=1.0,
        ring_scale=args.ring_scale,
        capture_invariants=True,
    ) if args.parity else None

    parity = None
    safety_audit = None
    seen = 0
    started = time.time()
    with torch.no_grad():
        for batch_number, (samples, targets) in enumerate(cfg.val_dataloader):
            if args.max_batches and batch_number >= args.max_batches:
                break
            samples = samples.to(device)
            matcher_targets = _normalise_targets(
                targets, samples.shape[-2], samples.shape[-1], device
            )
            gt_boxes = [box_cxcywh_to_xyxy(target["boxes"]) for target in matcher_targets]
            controller.set_gt_boxes(gt_boxes)
            if args.parity:
                original = model(samples)
                if isinstance(original, (tuple, list)):
                    original = original[0]
                baseline_controller.set_gt_boxes(gt_boxes)
                with baseline_controller:
                    wrapped_baseline = model(samples)
                if isinstance(wrapped_baseline, (tuple, list)):
                    wrapped_baseline = wrapped_baseline[0]
            with controller:
                outputs = model(samples)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            if args.parity and parity is None:
                parity = {
                    "wrapped_baseline_logits": _max_abs(wrapped_baseline["pred_logits"], original["pred_logits"]),
                    "wrapped_baseline_boxes": _max_abs(wrapped_baseline["pred_boxes"], original["pred_boxes"]),
                    "identity_logits": _max_abs(outputs["pred_logits"], original["pred_logits"]),
                    "identity_boxes": _max_abs(outputs["pred_boxes"], original["pred_boxes"]),
                }
                safety_audit = _invariant_audit(baseline_controller.records, controller.records)
                invariant_max = max(value for layer in safety_audit.values() for value in layer.values())
                if max(parity.values()) > 2e-6 or invariant_max > 0:
                    raise RuntimeError(f"GT attention identity safety gate failed: {parity}, {safety_audit}")

            original_sizes = torch.stack([target["orig_size"] for target in targets]).to(device)
            results = solver.postprocessor(outputs, original_sizes)
            evaluator.update({
                int(target["image_id"].item()): result
                for target, result in zip(targets, results)
            })

            boxes_cpu = box_cxcywh_to_xyxy(outputs["pred_boxes"].detach().cpu())
            logits_cpu = outputs["pred_logits"].detach().cpu()
            for batch_index, target in enumerate(targets):
                image_id = int(target["image_id"].item())
                if image_id not in rows_by_image:
                    continue
                target_xyxy = gt_boxes[batch_index].detach().cpu()
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
            "prior_source": "union of all GT boxes/rings per image",
            "prior": args.prior,
            "layers": args.layers,
            "actual_layer_indices": list(guided_layers),
            "ring_scale": args.ring_scale,
            "weights": {"box": box_weight, "ring": ring_weight, "background": background_weight},
            "checkpoint": args.checkpoint,
            "checkpoint_weights": args.weights,
            "n_images": seen,
            "n_lesions": len(lesion_records),
            "parity": parity,
            "safety_audit": safety_audit,
            "negative_image_policy": "all-one prior",
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
            "ecdetseg.tests.test_cmp5l_gt_guided_attention"
        )
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)
    evaluate(cli_args)
