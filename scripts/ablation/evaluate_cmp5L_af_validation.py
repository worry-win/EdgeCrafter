"""Frozen-baseline cmp5L A–F validation-only inference audit.

This module exposes the coefficient/region/postprocess contract as small public
functions so the scientific intervention can be tested independently of the
full model runner.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision


FAR_BG = 0
RING = 1
FG = 2
RING_SCALE = 1.5


def json_compatible_args(args):
    """Return CLI arguments using only JSON-native values."""
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, (Path, torch.device)):
            value = str(value)
        result[key] = value
    return result


def require_tensor_subset_coverage(requested, observed):
    requested_ids = {int(image_id) for image_id in requested}
    observed_ids = {int(image_id) for image_id in observed}
    missing = sorted(requested_ids - observed_ids)
    if missing:
        raise RuntimeError(f"missing fixed tensor subset image IDs: {missing}")
LAMBDA_MIN = 0.2


def region_mask_with_outside(locations, gt_xyxy, scale=RING_SCALE):
    """Classify normalized locations with historical FG>ring>far-BG priority.

    The out-of-image flag is orthogonal: it is recorded for diagnosis but does
    not alter the historical region coefficient.
    """
    outside = ((locations < 0) | (locations > 1)).any(dim=-1)
    flat = locations.reshape(-1, 2)
    inside = torch.zeros(flat.shape[0], dtype=torch.bool, device=locations.device)
    in_ring = torch.zeros_like(inside)
    for box in gt_xyxy:
        x0, y0, x1, y1 = box
        x, y = flat[:, 0], flat[:, 1]
        inside |= (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half_width = (x1 - x0) * scale / 2
        half_height = (y1 - y0) * scale / 2
        in_ring |= (
            (x >= cx - half_width)
            & (x <= cx + half_width)
            & (y >= cy - half_height)
            & (y <= cy + half_height)
        )
    region = torch.where(
        inside,
        torch.full_like(inside, FG, dtype=torch.long),
        torch.where(
            in_ring & ~inside,
            torch.full_like(inside, RING, dtype=torch.long),
            torch.full_like(inside, FAR_BG, dtype=torch.long),
        ),
    )
    return region.reshape(locations.shape[:-1]), outside


def normal_bg_dependency(attention_by_layer, region_by_layer):
    """Historical point-sum, head-mean, layer-mean far-BG dependency."""
    if len(attention_by_layer) != len(region_by_layer) or not attention_by_layer:
        raise ValueError("matching non-empty attention/region layer lists required")
    layer_mass = []
    for attention, region in zip(attention_by_layer, region_by_layer):
        layer_mass.append(
            (attention * (region == FAR_BG).to(attention.dtype)).sum(-1).mean(-1)
        )
    return torch.stack(layer_mass).mean(0)


def build_af_coeff(
    condition,
    g_gt,
    g_pred,
    region,
    has_gt,
    lambda_min=LAMBDA_MIN,
):
    """Return `[B,Q,H,P]` coefficients for the locked A–F definition."""
    if condition not in set("ABCDEF"):
        raise ValueError(f"unknown A-F condition: {condition}")
    if condition == "A":
        return torch.ones_like(region, dtype=torch.float32)
    batch, queries, heads, points = region.shape
    if condition == "B":
        gate = torch.ones(batch, queries, device=region.device, dtype=torch.float32)
    elif condition in ("C", "E"):
        gate = g_gt.to(dtype=torch.float32)
    else:
        gate = g_pred.to(dtype=torch.float32)
    if condition in ("B", "C", "D"):
        spatial = region != FG
    else:
        spatial = torch.ones_like(region, dtype=torch.bool)
    active = gate[:, :, None, None] * spatial.to(dtype=torch.float32)
    # B-E reproduce the historical GT-empty shortcut. F is explicitly GT-free.
    if condition != "F":
        active = active * has_gt.to(dtype=torch.float32)[:, None, None, None]
    return 1.0 - (1.0 - float(lambda_min)) * active


def topk_query_predictions(output, original_sizes, topk=300):
    """Focal postprocess with query identities preserved."""
    logits, boxes = output["pred_logits"], output["pred_boxes"]
    scores = logits.sigmoid()
    top_scores, flat_index = torch.topk(scores.flatten(1), topk, dim=-1)
    labels = flat_index % logits.shape[-1]
    query_ids = flat_index // logits.shape[-1]
    boxes_xyxy = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
    boxes_xyxy = boxes_xyxy * original_sizes.repeat(1, 2).unsqueeze(1)
    selected_boxes = boxes_xyxy.gather(
        1, query_ids.unsqueeze(-1).expand(-1, -1, boxes_xyxy.shape[-1])
    )
    return [
        {
            "scores": score,
            "labels": label,
            "boxes": box,
            "query_ids": query_id,
        }
        for score, label, box, query_id in zip(
            top_scores, labels, selected_boxes, query_ids
        )
    ]


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sampled_value_tensor(value, spatial_shapes, sampling_locations, num_points_list):
    """Return bilinear samples as `[B,Q,H,P,C_head]`."""
    batch, heads, channels, _ = value[0].shape
    queries = sampling_locations.shape[1]
    grids = (2 * sampling_locations - 1).permute(0, 2, 1, 3, 4).flatten(0, 1)
    grid_by_level = grids.split(num_points_list, dim=-2)
    sampled = []
    for level, ((height, width), grid) in enumerate(zip(spatial_shapes, grid_by_level)):
        sampled.append(
            F.grid_sample(
                value[level].reshape(batch * heads, channels, height, width),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )
    joined = torch.cat(sampled, dim=-1)
    return joined.reshape(batch, heads, channels, queries, -1).permute(0, 3, 1, 4, 2)


def layer_query_statistics(attention, region, outside, coeff):
    """Compact `[B,Q,13]` spatial/coefficient diagnostics."""
    dtype = attention.dtype
    one_minus = 1.0 - coeff
    masses = [
        (attention * (region == label).to(dtype)).sum(-1).mean(-1)
        for label in (FG, RING, FAR_BG)
    ]
    outside_mass = (attention * outside.to(dtype)).sum(-1).mean(-1)
    outside_ratio = outside.to(dtype).mean(dim=(-1, -2))
    coeff_mean = coeff.mean(dim=(-1, -2))
    coeff_min = coeff.amin(dim=(-1, -2))
    coeff_max = coeff.amax(dim=(-1, -2))
    suppressed_ratio = (coeff < 1.0 - 1e-7).to(dtype).mean(dim=(-1, -2))
    weighted = (attention * one_minus).sum(-1).mean(-1)
    regional_weighted = [
        (attention * one_minus * (region == label).to(dtype)).sum(-1).mean(-1)
        for label in (FG, RING, FAR_BG)
    ]
    return torch.stack(
        [
            *masses,
            outside_mass,
            outside_ratio,
            coeff_mean,
            coeff_min,
            coeff_max,
            suppressed_ratio,
            weighted,
            *regional_weighted,
        ],
        dim=-1,
    )


LAYER_STAT_NAMES = [
    "fg_attention_mass",
    "ring_attention_mass",
    "far_bg_attention_mass",
    "outside_attention_mass",
    "outside_point_ratio",
    "coeff_mean",
    "coeff_min",
    "coeff_max",
    "suppressed_point_ratio",
    "weighted_suppression",
    "fg_weighted_suppression",
    "ring_weighted_suppression",
    "far_bg_weighted_suppression",
]


def _tensor_output(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return value[0]
    raise TypeError(f"expected tensor output, got {type(value).__name__}")


def _historical_coeff(condition, g_dependency, region):
    from scripts.ablation.diag_cmp5L_adaptive_suppression import build_suppression

    if condition == "B":
        return build_suppression("NP-fixed", None, None, region, None)
    if condition == "C":
        dependency = torch.where(
            g_dependency,
            torch.full_like(g_dependency, 0.200001, dtype=torch.float32),
            torch.full_like(g_dependency, 0.2, dtype=torch.float32),
        )
        return build_suppression("dep-gate", None, dependency, region, 0.2)
    raise ValueError(condition)


def run_af_branch(
    ec,
    decoder_input,
    condition,
    gt_xyxy,
    has_gt,
    g_gt,
    g_pred,
    image_ids,
    tensor_subset,
    audit,
):
    """Replay one A–F condition from fixed Normal decoder initialization."""
    from scripts.ablation.diag_cmp5L_adaptive_suppression import (
        run_suppressed,
        suppression_core,
    )

    modules = [ec.decoder.layers[index].cross_attn for index in range(3)]
    originals = [module.ms_deformable_attn_core for module in modules]
    handles = []
    batch_stats = [None, None, None]
    tensor_records = {}
    selected = {
        index: image_id
        for index, image_id in enumerate(image_ids)
        if int(image_id) in tensor_subset
    }

    def record_tensor(image_id, layer, name, tensor):
        tensor_records.setdefault(str(image_id), {}).setdefault(condition, {}).setdefault(
            "layers", {}
        ).setdefault(str(layer), {})[name] = tensor.detach().to("cpu", torch.float16)

    for layer_index, module in enumerate(modules):
        def pre_hook_factory(index):
            def hook(_module, inputs):
                for batch_index, image_id in selected.items():
                    record_tensor(image_id, index, "query_in", inputs[0][batch_index])
            return hook

        def post_hook_factory(index):
            def hook(_module, _inputs, output):
                value = _tensor_output(output)
                for batch_index, image_id in selected.items():
                    record_tensor(image_id, index, "query_out", value[batch_index])
            return hook

        handles.append(ec.decoder.layers[layer_index].register_forward_pre_hook(pre_hook_factory(layer_index)))
        handles.append(ec.decoder.layers[layer_index].register_forward_hook(post_hook_factory(layer_index)))

        def core_factory(index):
            def core(value, spatial_shapes, locations, attention, num_points_list):
                location_before = locations.detach().clone()
                attention_before = attention.detach().clone()
                regions, outside_flags = [], []
                for batch_index in range(locations.shape[0]):
                    region, outside = region_mask_with_outside(
                        locations[batch_index], gt_xyxy[batch_index]
                    )
                    regions.append(region)
                    outside_flags.append(outside)
                region = torch.stack(regions)
                outside = torch.stack(outside_flags)
                coeff = build_af_coeff(condition, g_gt, g_pred, region, has_gt)
                batch_stats[index] = layer_query_statistics(
                    attention, region, outside, coeff
                ).detach().cpu().to(torch.float16).numpy()

                audit["coeff_min"][condition][index] = min(
                    audit["coeff_min"][condition][index], float(coeff.min())
                )
                audit["coeff_max"][condition][index] = max(
                    audit["coeff_max"][condition][index], float(coeff.max())
                )
                audit["coeff_shapes"][condition][index] = list(coeff.shape)
                if condition in ("B", "C") and audit["historical_coeff_max_abs"][condition][index] is None:
                    differences = []
                    for batch_index in range(locations.shape[0]):
                        expected = (
                            _historical_coeff(condition, g_gt[batch_index], region[batch_index])
                            if bool(has_gt[batch_index])
                            else torch.ones_like(coeff[batch_index])
                        )
                        differences.append(float((coeff[batch_index] - expected).abs().max()))
                    audit["historical_coeff_max_abs"][condition][index] = max(differences)

                if selected:
                    sampled = sampled_value_tensor(
                        value, spatial_shapes, locations, num_points_list
                    )
                    z_pre = (sampled * attention[..., None]).sum(-2).flatten(-2)
                    z_post = (
                        sampled * attention[..., None] * coeff[..., None]
                    ).sum(-2).flatten(-2)
                    contributions = {}
                    contributions_post = {}
                    for label, name in ((FG, "fg"), (RING, "ring"), (FAR_BG, "far_bg")):
                        contributions[name] = (
                            sampled
                            * attention[..., None]
                            * (region == label)[..., None].to(sampled.dtype)
                        ).sum(-2).flatten(-2)
                        contributions_post[name] = (
                            sampled
                            * attention[..., None]
                            * coeff[..., None]
                            * (region == label)[..., None].to(sampled.dtype)
                        ).sum(-2).flatten(-2)
                    for batch_index, image_id in selected.items():
                        record_tensor(image_id, index, "sampling_locations", locations[batch_index])
                        record_tensor(image_id, index, "attention_weights", attention[batch_index])
                        record_tensor(image_id, index, "sampled_value", sampled[batch_index])
                        record_tensor(image_id, index, "coeff", coeff[batch_index])
                        record_tensor(image_id, index, "z_pre", z_pre[batch_index])
                        record_tensor(image_id, index, "z_post", z_post[batch_index])
                        for name, contribution in contributions.items():
                            record_tensor(image_id, index, f"z_{name}", contribution[batch_index])
                        for name, contribution in contributions_post.items():
                            record_tensor(image_id, index, f"z_post_{name}", contribution[batch_index])

                output = suppression_core(
                    value, spatial_shapes, locations, attention, num_points_list, coeff
                )
                audit["location_input_mutation_max_abs"] = max(
                    audit["location_input_mutation_max_abs"],
                    float((locations - location_before).abs().max()),
                )
                audit["attention_input_mutation_max_abs"] = max(
                    audit["attention_input_mutation_max_abs"],
                    float((attention - attention_before).abs().max()),
                )
                return output
            return core

        module.ms_deformable_attn_core = core_factory(layer_index)

    try:
        output = run_suppressed(
            ec,
            decoder_input["query_content"],
            decoder_input["reference_unact"],
            decoder_input["memory"],
            decoder_input["spatial_shapes"],
            set(),
            "NN",
            None,
            [],
            None,
            None,
            None,
        )
    finally:
        for handle in handles:
            handle.remove()
        for module, original in zip(modules, originals):
            module.ms_deformable_attn_core = original
    if any(module.ms_deformable_attn_core is not original for module, original in zip(modules, originals)):
        raise RuntimeError("cross-attention core restoration failed")
    return output, batch_stats, tensor_records


def _merge_tensor_records(destination, source):
    for image_id, conditions in source.items():
        destination.setdefault(image_id, {}).update(conditions)


def _normal_query_diagnostics(output, gt_xyxy, gt_labels):
    from scripts.ablation.diag_cmp5L_adaptive_suppression import _iou_matrix
    from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy

    scores = output["pred_logits"].sigmoid()
    top2 = scores.topk(2, dim=-1).values
    result = {
        "score": scores.max(-1).values,
        "top1_class": scores.argmax(-1),
        "margin": top2[..., 0] - top2[..., 1],
    }
    best_indices, best_ious, gt_logits = [], [], []
    boxes = box_cxcywh_to_xyxy(output["pred_boxes"])
    for batch_index in range(boxes.shape[0]):
        if gt_xyxy[batch_index].numel() == 0:
            best_indices.append(torch.full((boxes.shape[1],), -1, device=boxes.device, dtype=torch.long))
            best_ious.append(torch.zeros(boxes.shape[1], device=boxes.device))
            gt_logits.append(torch.full((boxes.shape[1],), float("nan"), device=boxes.device))
            continue
        iou = _iou_matrix(boxes[batch_index], gt_xyxy[batch_index])
        best_iou, best_index = iou.max(-1)
        label = gt_labels[batch_index][best_index]
        best_indices.append(best_index)
        best_ious.append(best_iou)
        gt_logits.append(output["pred_logits"][batch_index].gather(1, label[:, None]).squeeze(1))
    result.update({
        "best_gt_index": torch.stack(best_indices),
        "best_iou": torch.stack(best_ious),
        "gt_class_logit": torch.stack(gt_logits),
    })
    return result


def _summarize_evaluator(evaluator):
    from engine.solver.ec_engine import summarize_yolo_pr_curve_metrics

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    coco_eval = evaluator.coco_eval["bbox"]
    yolo = summarize_yolo_pr_curve_metrics(coco_eval, evaluator.coco_gt) or {}
    overall = yolo.get("yolo_f1_iou50", {})
    precision = coco_eval.eval["precision"]
    iou50 = int(np.argmin(np.abs(coco_eval.params.iouThrs - 0.5)))
    per_class = {}
    categories = evaluator.coco_gt.loadCats(coco_eval.params.catIds)
    for class_index, category in enumerate(categories):
        curve = precision[iou50, :, class_index, 0, -1]
        valid = curve > -1
        per_class[str(category["id"])] = {
            "name": category.get("name", str(category["id"])),
            "ap50": float(curve[valid].mean()) if valid.any() else -1.0,
        }
    stats = coco_eval.stats.tolist()
    return {
        "ap": float(stats[0]),
        "ap50": float(stats[1]),
        "ap75": float(stats[2]),
        "precision": float(overall.get("precision", 0.0)),
        "recall": float(overall.get("recall", 0.0)),
        "f1": float(overall.get("f1", 0.0)),
        "ar100": float(stats[8]),
        "per_class_ap50": per_class,
    }


def _prediction_record(image_id, result):
    return {
        "image_id": int(image_id),
        "boxes_xyxy": result["boxes"].detach().cpu().tolist(),
        "scores": result["scores"].detach().cpu().tolist(),
        "labels": result["labels"].detach().cpu().tolist(),
        "query_ids": result["query_ids"].detach().cpu().tolist(),
    }


def load_query_predictor(path, device):
    from scripts.ablation.train_cmp5L_bgdep_query_predictor import Predictor

    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("variant") != "q_l1":
        raise ValueError(f"locked predictor must be q_l1, got {checkpoint.get('variant')}")
    if float(checkpoint.get("bgdep_tau", -1)) != 0.2:
        raise ValueError("predictor bgdep_tau differs from locked 0.20")
    model = Predictor(checkpoint["input_dim"], checkpoint.get("hidden", 0)).to(device)
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    checkpoint["mean"] = checkpoint["mean"].to(device=device, dtype=torch.float32)
    checkpoint["std"] = checkpoint["std"].to(device=device, dtype=torch.float32)
    return model, checkpoint, {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "state_key_count": len(checkpoint["state_dict"]),
    }


def checkpoint_key_audit(model, checkpoint_path, weights):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if weights == "ema":
        ema = checkpoint.get("ema")
        if isinstance(ema, dict) and "module" in ema:
            state = ema["module"]
        elif hasattr(ema, "module"):
            state = ema.module.state_dict()
        else:
            raise ValueError("EMA state not found in checkpoint")
    else:
        state = checkpoint["model"]
    model_keys, state_keys = set(model.state_dict()), set(state)
    result = {
        "weight_type": weights,
        "model_key_count": len(model_keys),
        "checkpoint_key_count": len(state_keys),
        "missing_keys": sorted(model_keys - state_keys),
        "unexpected_keys": sorted(state_keys - model_keys),
        "epoch": int(checkpoint["epoch"]) if checkpoint.get("epoch") is not None else None,
    }
    del checkpoint, state
    return result


def encoder_topk_indices(ec, memory, spatial_shapes):
    if ec.training or ec.eval_spatial_size is None:
        _anchors, valid_mask = ec._generate_anchors(spatial_shapes, device=memory.device)
    else:
        valid_mask = ec.valid_mask
    masked_memory = valid_mask.to(memory.dtype) * memory
    logits = ec.enc_score_head(masked_memory)
    if ec.query_select_method == "default":
        return torch.topk(logits.max(-1).values, ec.num_queries, dim=-1).indices
    if ec.query_select_method == "one2many":
        return torch.topk(logits.flatten(1), ec.num_queries, dim=-1).indices // ec.num_classes
    if ec.query_select_method == "agnostic":
        return torch.topk(logits.squeeze(-1), ec.num_queries, dim=-1).indices
    raise ValueError(f"unknown query_select_method {ec.query_select_method}")


def evaluate(args):
    from scripts.ablation.probe_cmp5L_decoder_internal_tensors import (
        DecoderInternalCapture,
        _build,
        _normalise_targets,
    )
    from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    args.device = device
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    tensor_subset = set(json.loads(Path(args.tensor_subset).read_text())["image_ids"])
    predictor, predictor_spec, predictor_load = load_query_predictor(
        args.query_predictor, device
    )
    cfg, solver, model = _build(args)
    model.eval()
    model_load = checkpoint_key_audit(model, args.checkpoint, args.weights)
    if model_load["missing_keys"] or model_load["unexpected_keys"]:
        raise RuntimeError(f"model checkpoint key mismatch: {model_load}")
    conditions = list("ABCDEF")
    evaluators = {condition: copy.deepcopy(cfg.evaluator) for condition in conditions}
    for evaluator in evaluators.values():
        evaluator.cleanup()
    streams = {
        condition: (output_dir / f"predictions_{condition}.jsonl").open("x", encoding="utf-8")
        for condition in conditions
    }
    audit = {
        "baseline_A_max_abs": {"logits": 0.0, "boxes": 0.0},
        "A_repeat_after_branches_max_abs": None,
        "normal_hook_max_abs": {"logits": 0.0, "boxes": 0.0},
        "location_input_mutation_max_abs": 0.0,
        "attention_input_mutation_max_abs": 0.0,
        "historical_coeff_max_abs": {"B": [None] * 3, "C": [None] * 3},
        "coeff_min": {condition: [float("inf")] * 3 for condition in conditions},
        "coeff_max": {condition: [float("-inf")] * 3 for condition in conditions},
        "coeff_shapes": {condition: [None] * 3 for condition in conditions},
        "finite_outputs": {condition: True for condition in conditions},
        "predictor_load": predictor_load,
        "model_load": model_load,
        "model_eval": not model.training,
        "conditions": conditions,
        "direct_layers": [0, 1, 2],
    }
    query_batches = {key: [] for key in (
        "image_id", "d_bg", "p_suppress", "g_gt", "g_pred", "false_suppress",
        "missed_suppress", "score", "top1_class", "margin", "best_gt_index",
        "best_iou", "gt_class_logit",
    )}
    layer_batches = {
        condition: [[] for _ in range(3)] for condition in conditions
    }
    raw_batches = {
        condition: {"logits": [], "boxes": []} for condition in conditions
    }
    tensor_records = {}
    seen_ids = []
    seen = 0
    first_batch = True
    started = time.time()
    try:
        with torch.no_grad():
            for samples, targets in cfg.val_dataloader:
                if args.limit and seen >= args.limit:
                    break
                samples = samples.to(device)
                image_ids = [int(target["image_id"].item()) for target in targets]
                matcher_targets = _normalise_targets(
                    targets, samples.shape[-2], samples.shape[-1], device
                )
                gt_xyxy = [box_cxcywh_to_xyxy(target["boxes"]) for target in matcher_targets]
                gt_labels = [target["labels"] for target in matcher_targets]
                has_gt = torch.tensor(
                    [boxes.shape[0] > 0 for boxes in gt_xyxy], device=device
                )
                baseline = model(samples)
                with DecoderInternalCapture(model.decoder) as capture:
                    normal = model(samples)
                audit["normal_hook_max_abs"]["logits"] = max(
                    audit["normal_hook_max_abs"]["logits"],
                    float((baseline["pred_logits"] - normal["pred_logits"]).abs().max()),
                )
                audit["normal_hook_max_abs"]["boxes"] = max(
                    audit["normal_hook_max_abs"]["boxes"],
                    float((baseline["pred_boxes"] - normal["pred_boxes"]).abs().max()),
                )
                normal_regions = []
                normal_attention = []
                for layer_index in range(3):
                    record = capture.layers[layer_index]
                    normal_attention.append(record["attention_weights"])
                    normal_regions.append(torch.stack([
                        region_mask_with_outside(record["sampling_locations"][batch_index], gt_xyxy[batch_index])[0]
                        for batch_index in range(len(image_ids))
                    ]))
                d_bg = normal_bg_dependency(normal_attention, normal_regions)
                g_gt = d_bg > float(predictor_spec["bgdep_tau"])
                q_l1 = capture.layers[1]["query_in"].float()
                p_suppress = torch.sigmoid(
                    predictor((q_l1 - predictor_spec["mean"]) / predictor_spec["std"])
                )
                g_pred = p_suppress > float(predictor_spec["threshold"])
                diagnostic = _normal_query_diagnostics(normal, gt_xyxy, gt_labels)
                repeated_ids = torch.tensor(image_ids)[:, None].expand(-1, d_bg.shape[1])
                payloads = {
                    "image_id": repeated_ids,
                    "d_bg": d_bg,
                    "p_suppress": p_suppress,
                    "g_gt": g_gt,
                    "g_pred": g_pred,
                    "false_suppress": g_pred & ~g_gt,
                    "missed_suppress": ~g_pred & g_gt,
                    **diagnostic,
                }
                for key, value in payloads.items():
                    query_batches[key].append(value.detach().cpu().numpy())

                decoder_input = capture.decoder_input
                selected_batch_indices = {
                    index: image_id for index, image_id in enumerate(image_ids)
                    if image_id in tensor_subset
                }
                if selected_batch_indices:
                    topk_indices = encoder_topk_indices(
                        model.decoder,
                        decoder_input["memory"],
                        decoder_input["spatial_shapes"],
                    )
                    for batch_index, image_id in selected_batch_indices.items():
                        tensor_records.setdefault(str(image_id), {})["shared"] = {
                            "query_content": decoder_input["query_content"][batch_index].detach().cpu().to(torch.float16),
                            "initial_reference": decoder_input["reference_unact"][batch_index].sigmoid().detach().cpu().to(torch.float16),
                            "encoder_topk_indices": topk_indices[batch_index].detach().cpu(),
                            "g_gt": g_gt[batch_index].detach().cpu(),
                            "g_pred": g_pred[batch_index].detach().cpu(),
                            "d_bg": d_bg[batch_index].detach().cpu().to(torch.float16),
                            "p_suppress": p_suppress[batch_index].detach().cpu().to(torch.float16),
                        }
                outputs = {}
                first_a = None
                for condition in conditions:
                    branch, statistics, tensors = run_af_branch(
                        model.decoder,
                        decoder_input,
                        condition,
                        gt_xyxy,
                        has_gt,
                        g_gt,
                        g_pred,
                        image_ids,
                        tensor_subset,
                        audit,
                    )
                    outputs[condition] = branch
                    if condition == "A":
                        first_a = branch
                        audit["baseline_A_max_abs"]["logits"] = max(
                            audit["baseline_A_max_abs"]["logits"],
                            float((branch["pred_logits"] - normal["pred_logits"]).abs().max()),
                        )
                        audit["baseline_A_max_abs"]["boxes"] = max(
                            audit["baseline_A_max_abs"]["boxes"],
                            float((branch["pred_boxes"] - normal["pred_boxes"]).abs().max()),
                        )
                    for layer_index, stat in enumerate(statistics):
                        layer_batches[condition][layer_index].append(stat)
                    _merge_tensor_records(tensor_records, tensors)
                    raw_batches[condition]["logits"].append(
                        branch["pred_logits"].detach().cpu().to(torch.float16).numpy()
                    )
                    raw_batches[condition]["boxes"].append(
                        branch["pred_boxes"].detach().cpu().to(torch.float16).numpy()
                    )
                    audit["finite_outputs"][condition] &= bool(
                        torch.isfinite(branch["pred_logits"]).all()
                        and torch.isfinite(branch["pred_boxes"]).all()
                    )
                if first_batch:
                    repeated_a, _, _ = run_af_branch(
                        model.decoder, decoder_input, "A", gt_xyxy, has_gt, g_gt, g_pred,
                        image_ids, set(), audit,
                    )
                    audit["A_repeat_after_branches_max_abs"] = {
                        "logits": float((repeated_a["pred_logits"] - first_a["pred_logits"]).abs().max()),
                        "boxes": float((repeated_a["pred_boxes"] - first_a["pred_boxes"]).abs().max()),
                    }
                    first_batch = False

                original_sizes = torch.stack([target["orig_size"] for target in targets]).to(device)
                for condition, branch in outputs.items():
                    official = solver.postprocessor(branch, original_sizes)
                    identified = topk_query_predictions(branch, original_sizes, topk=300)
                    if seen == 0:
                        for left, right in zip(official, identified):
                            if not (
                                torch.equal(left["labels"], right["labels"])
                                and torch.allclose(left["scores"], right["scores"])
                                and torch.allclose(left["boxes"], right["boxes"])
                            ):
                                raise RuntimeError("query-identity postprocess differs from official postprocessor")
                    evaluators[condition].update({
                        image_id: result for image_id, result in zip(image_ids, official)
                    })
                    for image_id, result in zip(image_ids, identified):
                        streams[condition].write(
                            json.dumps(_prediction_record(image_id, result)) + "\n"
                        )
                seen_ids.extend(image_ids)
                seen += len(image_ids)
                if args.log_every and seen % args.log_every < len(image_ids):
                    print(f"[{seen}/{len(cfg.val_dataloader.dataset)}] {time.time()-started:.1f}s", flush=True)
    finally:
        for stream in streams.values():
            stream.close()

    expected = args.limit or args.expected_images
    if seen != expected or len(set(seen_ids)) != seen:
        raise RuntimeError(f"image coverage failed: seen={seen}, unique={len(set(seen_ids))}, expected={expected}")
    if not args.limit and set(map(int, tensor_records)) != tensor_subset:
        raise RuntimeError(
            f"fixed tensor subset coverage failed: observed={sorted(map(int, tensor_records))}, "
            f"expected={sorted(tensor_subset)}"
        )
    parity_values = [
        *audit["baseline_A_max_abs"].values(),
        *audit["normal_hook_max_abs"].values(),
        *audit["A_repeat_after_branches_max_abs"].values(),
        audit["location_input_mutation_max_abs"],
        audit["attention_input_mutation_max_abs"],
        *audit["historical_coeff_max_abs"]["B"],
        *audit["historical_coeff_max_abs"]["C"],
    ]
    if max(parity_values) > 2e-6 or not all(audit["finite_outputs"].values()):
        raise RuntimeError(f"technical gate failed: {audit}")

    metrics = {condition: _summarize_evaluator(evaluator) for condition, evaluator in evaluators.items()}
    np.savez_compressed(
        output_dir / "query_diagnostics.npz",
        **{key: np.concatenate(values, axis=0) for key, values in query_batches.items()},
    )
    np.savez_compressed(
        output_dir / "layer_query_stats.npz",
        stat_names=np.asarray(LAYER_STAT_NAMES),
        image_ids=np.asarray(seen_ids, dtype=np.int64),
        **{
            f"{condition}_L{layer_index}": np.concatenate(layer_batches[condition][layer_index], axis=0)
            for condition in conditions for layer_index in range(3)
        },
    )
    np.savez_compressed(
        output_dir / "raw_query_outputs.npz",
        image_ids=np.asarray(seen_ids, dtype=np.int64),
        **{
            f"{condition}_{name}": np.concatenate(raw_batches[condition][name], axis=0)
            for condition in conditions for name in ("logits", "boxes")
        },
    )
    torch.save(tensor_records, output_dir / "fixed_tensor_subset.pt")
    (output_dir / "metrics_full.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    audit.update({
        "n_images": seen,
        "unique_image_ids": len(set(seen_ids)),
        "tensor_subset_requested": sorted(tensor_subset),
        "tensor_subset_observed": sorted(map(int, tensor_records)),
        "elapsed_seconds": round(time.time() - started, 2),
    })
    (output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    require_tensor_subset_coverage(tensor_subset, tensor_records)
    manifest = {
        "scope": "validation-only A-F frozen baseline EMA",
        "args": json_compatible_args(args),
        "hashes": {
            "runner": sha256_path(__file__),
            "checkpoint": sha256_path(args.checkpoint),
            "predictor": sha256_path(args.query_predictor),
            "config": sha256_path(args.config),
            "annotations": sha256_path(args.ann_file),
        },
        "predictor": {
            "variant": predictor_spec["variant"],
            "threshold": float(predictor_spec["threshold"]),
            "threshold_source": "stored validation-label F1 optimum",
            "output_semantics": "p_suppress=P(d_BG>0.20)",
            "normal_forward_required": True,
        },
        "conditions": {
            "A": "identity",
            "B": "all queries; GT non-FG; historical empty-GT skip",
            "C": "GT dependency gate; GT non-FG; historical empty-GT skip",
            "D": "predicted dependency gate; GT non-FG; historical empty-GT skip",
            "E": "GT dependency gate; all points; historical empty-GT skip",
            "F": "predicted dependency gate; all points; no GT empty-image read",
        },
        "lambda_min": LAMBDA_MIN,
        "direct_layers": [0, 1, 2],
        "ring_scale": RING_SCALE,
        "layer_stat_names": LAYER_STAT_NAMES,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "COMPLETED.txt").write_text("status=passed\n", encoding="utf-8")
    print(json.dumps({"audit": audit, "metrics": metrics}, ensure_ascii=False)[:12000], flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--query-predictor", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--img-folder", required=True)
    parser.add_argument("--tensor-subset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", choices=("ema",), default="ema")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--expected-images", type=int, default=2975)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
