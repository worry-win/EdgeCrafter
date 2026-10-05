"""Oracle-quality gated L1 predicted-box guidance for decoder attention."""

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
    apply_box_attention_prior,
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


def _cxcywh_to_xyxy(boxes):
    center = boxes[..., :2]
    half = boxes[..., 2:] / 2
    return torch.cat((center - half, center + half), dim=-1)


def _pairwise_iou(boxes1, boxes2):
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(-1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0).prod(-1)[:, None]
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0).prod(-1)[None, :]
    return intersection / (area1 + area2 - intersection).clamp_min(1e-9)


def oracle_query_gate(l1_boxes_cxcywh, l1_logits, gt_boxes_xyxy, gt_labels, *, gate):
    """Return per-query Oracle reliability and its best-IoU GT assignment."""
    if gate not in {"localization", "localization_class"}:
        raise ValueError(f"unknown gate: {gate}")
    gates, matches = [], []
    boxes_xyxy = _cxcywh_to_xyxy(l1_boxes_cxcywh)
    for batch_index, boxes in enumerate(gt_boxes_xyxy):
        boxes = torch.as_tensor(boxes, device=boxes_xyxy.device, dtype=boxes_xyxy.dtype)
        if boxes.numel() == 0:
            gates.append(torch.zeros(boxes_xyxy.shape[1], dtype=torch.bool, device=boxes_xyxy.device))
            matches.append(torch.full((boxes_xyxy.shape[1],), -1, dtype=torch.long, device=boxes_xyxy.device))
            continue
        best_iou, best_gt = _pairwise_iou(boxes_xyxy[batch_index], boxes).max(-1)
        reliable = best_iou >= 0.5
        if gate == "localization_class":
            labels = torch.as_tensor(gt_labels[batch_index], device=l1_logits.device, dtype=torch.long)
            reliable = reliable & (l1_logits[batch_index].argmax(-1) == labels[best_gt])
        gates.append(reliable)
        matches.append(best_gt)
    return torch.stack(gates), torch.stack(matches)


def mix_guided_attention(original, guided, gate):
    """Use guided weights only for Oracle-reliable queries."""
    return torch.where(gate[:, :, None, None], guided, original)


class OracleGatedL1Controller(AbstractContextManager):
    """Apply L1-box attention guidance only to GT-certified queries."""

    def __init__(
        self,
        ec_transformer,
        selected_layers,
        *,
        gate,
        box_weight,
        ring_weight,
        background_weight,
        ring_scale=1.5,
        capture_invariants=False,
    ):
        self.ec = ec_transformer
        self.decoder = ec_transformer.decoder
        self.selected_layers = set(selected_layers)
        if not self.selected_layers.issubset({1, 2}):
            raise ValueError("only actual layers 1/2 (user L2/L3) may be guided")
        self.gate_kind = gate
        self.box_weight = box_weight
        self.ring_weight = ring_weight
        self.background_weight = background_weight
        self.ring_scale = ring_scale
        self.capture_invariants = capture_invariants
        self.gt_boxes = None
        self.gt_labels = None
        self.l1_boxes = None
        self.l1_logits = None
        self.gate = None
        self.best_gt = None
        self.records = {}
        self._originals = {}
        self._pre_handle = None
        self._score_handle = None

    def set_targets(self, gt_boxes, gt_labels):
        self.gt_boxes = gt_boxes
        self.gt_labels = gt_labels

    def _reset(self, _module, _inputs):
        self.l1_boxes = None
        self.l1_logits = None
        self.gate = None
        self.best_gt = None
        self.records.clear()
        if self.gt_boxes is None or self.gt_labels is None:
            raise RuntimeError("GT boxes/labels must be set before decoder forward")

    def _capture_l1_logits(self, _module, _inputs, output):
        self.l1_logits = output.detach()

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

            if index == 1:
                self.l1_boxes = reference_points[:, :, 0, :].detach()
                if self.l1_logits is None:
                    raise RuntimeError("L1 class logits were not observed before L2 cross-attention")
                self.gate, self.best_gt = oracle_query_gate(
                    self.l1_boxes,
                    self.l1_logits,
                    self.gt_boxes,
                    self.gt_labels,
                    gate=self.gate_kind,
                )
            if self.l1_boxes is None or self.gate is None:
                raise RuntimeError("L1 reliability gate was not established before guidance")

            if index in self.selected_layers:
                guided, prior = apply_box_attention_prior(
                    weights,
                    locations,
                    self.l1_boxes,
                    box_weight=self.box_weight,
                    ring_weight=self.ring_weight,
                    background_weight=self.background_weight,
                    ring_scale=self.ring_scale,
                )
                used_weights = mix_guided_attention(weights, guided, self.gate)
                effective_prior = torch.where(
                    self.gate[:, :, None, None], prior, torch.ones_like(prior)
                )
            else:
                used_weights = weights
                effective_prior = torch.ones_like(weights)
            record = {
                "sampling_locations": locations.detach(),
                "attention_weights": used_weights.detach(),
                "prior": effective_prior.detach(),
                "reference_points": reference_points.detach(),
                "gate": self.gate.detach(),
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
            raise RuntimeError("OracleGatedL1Controller requires eval mode")
        self._pre_handle = self.decoder.register_forward_pre_hook(self._reset)
        self._score_handle = self.ec.dec_score_head[0].register_forward_hook(self._capture_l1_logits)
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
        for handle_name in ("_pre_handle", "_score_handle"):
            handle = getattr(self, handle_name)
            if handle is not None:
                handle.remove()
                setattr(self, handle_name, None)
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
            "feature_value": max(_max_abs(a, b) for a, b in zip(base["value"], identity["value"])),
            "attention_weights": _max_abs(base["attention_weights"], identity["attention_weights"]),
        }
    return result


def _lesion_l1_metrics(controller, batch_index, gt_box, gt_class):
    l1_xyxy = box_cxcywh_to_xyxy(controller.l1_boxes[batch_index]).detach().cpu()
    gt_box = torch.as_tensor(gt_box, dtype=l1_xyxy.dtype)
    iou = _pairwise_iou(l1_xyxy, gt_box[None])[:, 0]
    logits = controller.l1_logits[batch_index].detach().cpu()
    class_mask = torch.arange(logits.shape[-1]) != gt_class
    margin = logits[:, gt_class] - logits[:, class_mask].max(-1).values
    return {
        "oracle_gate": [bool(value) for value in controller.gate[batch_index].detach().cpu()],
        "l1_iou_to_gt": [float(value) for value in iou],
        "l1_top_class": [int(value) for value in logits.argmax(-1)],
        "l1_gt_class_margin": [float(value) for value in margin],
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
    condition = "baseline" if args.prior == "baseline" else f"gate_{args.gate}_{args.prior}_{args.layers}"
    controller = OracleGatedL1Controller(
        model.decoder,
        guided_layers,
        gate=args.gate,
        box_weight=box_weight,
        ring_weight=ring_weight,
        background_weight=background_weight,
        ring_scale=args.ring_scale,
        capture_invariants=args.parity,
    )
    baseline_controller = OracleGatedL1Controller(
        model.decoder,
        (),
        gate=args.gate,
        box_weight=1.0,
        ring_weight=1.0,
        background_weight=1.0,
        ring_scale=args.ring_scale,
        capture_invariants=True,
    ) if args.parity else None

    parity = None
    safety_audit = None
    gate_count = 0
    query_count = 0
    seen = 0
    started = time.time()
    with torch.no_grad():
        for batch_number, (samples, targets) in enumerate(cfg.val_dataloader):
            if args.max_batches and batch_number >= args.max_batches:
                break
            samples = samples.to(device)
            matcher_targets = _normalise_targets(targets, samples.shape[-2], samples.shape[-1], device)
            gt_boxes = [box_cxcywh_to_xyxy(target["boxes"]) for target in matcher_targets]
            gt_labels = [target["labels"] for target in matcher_targets]
            controller.set_targets(gt_boxes, gt_labels)
            if args.parity:
                original = model(samples)
                if isinstance(original, (tuple, list)):
                    original = original[0]
                baseline_controller.set_targets(gt_boxes, gt_labels)
                with baseline_controller:
                    wrapped_baseline = model(samples)
                if isinstance(wrapped_baseline, (tuple, list)):
                    wrapped_baseline = wrapped_baseline[0]
            with controller:
                outputs = model(samples)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            gate_count += int(controller.gate.sum())
            query_count += controller.gate.numel()
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
                    raise RuntimeError(f"gated L1 identity safety gate failed: {parity}, {safety_audit}")

            original_sizes = torch.stack([target["orig_size"] for target in targets]).to(device)
            results = solver.postprocessor(outputs, original_sizes)
            evaluator.update({int(target["image_id"].item()): result for target, result in zip(targets, results)})

            boxes_cpu = box_cxcywh_to_xyxy(outputs["pred_boxes"].detach().cpu())
            logits_cpu = outputs["pred_logits"].detach().cpu()
            for batch_index, target in enumerate(targets):
                image_id = int(target["image_id"].item())
                if image_id not in rows_by_image:
                    continue
                target_xyxy = gt_boxes[batch_index].detach().cpu()
                original_width, original_height = [int(value) for value in target["orig_size"]]
                target_device = {key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()}
                for row in rows_by_image[image_id]:
                    gt_index = _target_index(row, target_device, original_width, original_height)
                    gt_box = target_xyxy[gt_index]
                    gt_class = int(row["gt_class"])
                    record = _binding_metrics(boxes_cpu[batch_index], logits_cpu[batch_index], gt_box, gt_class)
                    record.update(_attention_mass(controller.records, batch_index, gt_box))
                    record.update(_lesion_l1_metrics(controller, batch_index, gt_box, gt_class))
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
            "gate": args.gate,
            "prior": args.prior,
            "layers": args.layers,
            "actual_layer_indices": list(guided_layers),
            "l1_box_source": "actual layer-0 FDR box entering actual layer 1",
            "l1_class_source": "raw dec_score_head[0] pre-score after actual layer 0",
            "ring_scale": args.ring_scale,
            "weights": {"box": box_weight, "ring": ring_weight, "background": background_weight},
            "checkpoint": args.checkpoint,
            "checkpoint_weights": args.weights,
            "n_images": seen,
            "n_lesions": len(lesion_records),
            "gated_queries": gate_count,
            "total_queries": query_count,
            "gated_fraction": gate_count / query_count,
            "parity": parity,
            "safety_audit": safety_audit,
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
        details = {"coco_eval_bbox": coco_eval.stats.tolist(), "macro_f1_iou50": macro["f1_iou50"], **yolo}
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
    parser.add_argument("--gate", choices=("localization", "localization_class"), required=True)
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
            "ecdetseg.tests.test_cmp5l_oracle_gated_l1_attention"
        )
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)
    evaluate(cli_args)
