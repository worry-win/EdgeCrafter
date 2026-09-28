"""Build a stratified lesion set and dump compact cmp5L decoder behavior.

Selection reads the existing score-floor-zero prediction dumps.  Collection
then performs a fresh eval forward with :class:`DecoderInternalCapture`; it does
not trust or import conclusions from earlier routing analyses.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision


EC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EC_ROOT / "ecdetseg"))
sys.path.insert(0, str(EC_ROOT))

from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy  # noqa: E402
from scripts.ablation.probe_cmp5L_decoder_internal_tensors import (  # noqa: E402
    DecoderInternalCapture,
    _build,
    _normalise_targets,
)


ARMS = ("dinov2s", "dinov2b", "lw_xlarge", "mae_vitb", "ecvits_official")
CLASS_NAMES = ("solid", "cystic", "lymph", "duct")
QUERY_KINDS = ("q_iou", "q_cls", "q_global", "q_detection", "q_hungarian", "q_random")


def classify_lesion(success, localised):
    """Return mutually exclusive group; arm 0 is the student."""
    success = np.asarray(success, dtype=bool)
    localised = np.asarray(localised, dtype=bool)
    if success.all():
        return "agreement_easy"
    if not success.any():
        return "common_failure"
    if not success[0] and success[1:].any():
        return "rank_fixable" if localised[0] else "localization_fixable"
    if success[0] and not success[1:].all():
        return "student_correct_disagreement"
    return "other_disagreement"


def xyxy_iou(boxes, gt):
    lt = np.maximum(boxes[:, :2], gt[None, :2])
    rb = np.minimum(boxes[:, 2:], gt[None, 2:])
    inter = np.clip(rb - lt, 0, None).prod(1)
    area_a = np.clip(boxes[:, 2:] - boxes[:, :2], 0, None).prod(1)
    area_b = np.clip(gt[2:] - gt[:2], 0, None).prod()
    return inter / np.clip(area_a + area_b - inter, 1e-9, None)


def _load_prediction(path):
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _arm_lesion_metrics(data, gt_index, tau, score_threshold):
    image_id = int(data["gt_image_id"][gt_index])
    image_pos = int(np.flatnonzero(data["image_ids"] == image_id)[0])
    start, stop = int(data["rec_ptr"][image_pos]), int(data["rec_ptr"][image_pos + 1])
    boxes = data["rec_box"][start:stop].astype(np.float64)
    logits = data["rec_logit"][start:stop].astype(np.float64)
    gt_box = data["gt_box"][gt_index].astype(np.float64)
    gt_class = int(data["gt_cls"][gt_index])
    iou = xyxy_iou(boxes, gt_box)
    gt_score = 1.0 / (1.0 + np.exp(-logits[:, gt_class]))
    query_score = (1.0 / (1.0 + np.exp(-logits))).max(1)
    candidates = np.flatnonzero(iou >= tau)
    detection = int(candidates[np.argmax(gt_score[candidates])]) if len(candidates) else -1
    success = bool(len(candidates) and gt_score[detection] >= score_threshold)
    return {
        "success": success,
        "localised": bool(iou.max() >= tau),
        "max_iou": float(iou.max()),
        "best_localised_gt_score": float(gt_score[detection]) if detection >= 0 else 0.0,
        "q_iou": int(iou.argmax()),
        "q_cls": int(gt_score.argmax()),
        "q_global": int(query_score.argmax()),
        "q_detection": detection,
    }


def _balanced_take(rows, limit, seed):
    if len(rows) <= limit:
        return list(rows)
    rng = random.Random(seed)
    buckets = {}
    for row in rows:
        buckets.setdefault(int(row["gt_class"]), []).append(row)
    for values in buckets.values():
        rng.shuffle(values)
    picked = []
    while len(picked) < limit and buckets:
        for key in list(sorted(buckets)):
            if buckets[key] and len(picked) < limit:
                picked.append(buckets[key].pop())
            if not buckets[key]:
                del buckets[key]
    return picked


def _select_gradient_ann_ids(rows, limit, seed):
    """Select a deterministic group×class-stratified gradient subset."""
    if limit <= 0:
        return set()
    rng = random.Random(seed)
    buckets = {}
    for row in rows:
        buckets.setdefault((row["group"], int(row["gt_class"])), []).append(int(row["ann_id"]))
    for values in buckets.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < min(limit, len(rows)) and buckets:
        for key in list(sorted(buckets)):
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].pop())
            if not buckets[key]:
                del buckets[key]
    return set(selected)


def build_manifest(pred_dir, split, tau=0.5, score_threshold=0.5,
                   target_size=650, seed=42):
    pred_dir = Path(pred_dir)
    arms = [_load_prediction(pred_dir / f"{name}_{split}.npz") for name in ARMS]
    reference = arms[0]
    for other in arms[1:]:
        if not np.array_equal(reference["gt_ann_id"], other["gt_ann_id"]):
            raise ValueError("prediction dumps do not share the same GT order")
    rows = []
    for index, ann_id in enumerate(reference["gt_ann_id"]):
        metrics = [_arm_lesion_metrics(arm, index, tau, score_threshold) for arm in arms]
        success = [item["success"] for item in metrics]
        localised = [item["localised"] for item in metrics]
        rows.append({
            "lesion_index": index,
            "ann_id": int(ann_id),
            "image_id": int(reference["gt_image_id"][index]),
            "gt_class": int(reference["gt_cls"][index]),
            "class_name": CLASS_NAMES[int(reference["gt_cls"][index])],
            "gt_box": reference["gt_box"][index].astype(float).tolist(),
            "gt_area": float(reference["gt_area"][index]),
            "group": classify_lesion(success, localised),
            "is_disagreement": bool(any(success) and not all(success)),
            "arm_success": dict(zip(ARMS, success)),
            "arm_localised": dict(zip(ARMS, localised)),
            "arm_metrics": dict(zip(ARMS, metrics)),
        })
    groups = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    quotas = {
        "agreement_easy": 150,
        "common_failure": 150,
        "rank_fixable": 150,
        "localization_fixable": 100,
        "student_correct_disagreement": 100,
        "other_disagreement": 50,
    }
    selected = []
    for group, quota in quotas.items():
        selected.extend(_balanced_take(groups.get(group, []), min(quota, target_size), seed + len(selected)))
    selected_ids = {row["ann_id"] for row in selected}
    remainder = [row for row in rows if row["ann_id"] not in selected_ids]
    if len(selected) < target_size:
        selected.extend(_balanced_take(remainder, target_size - len(selected), seed + 991))
    elif len(selected) > target_size:
        selected = _balanced_take(selected, target_size, seed + 991)
    selected.sort(key=lambda row: (row["image_id"], row["ann_id"]))
    return selected


def _pairwise_iou_torch(boxes_a, boxes_b):
    left_top = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    right_bottom = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection = (right_bottom - left_top).clamp(min=0).prod(-1)
    area_a = (boxes_a[:, 2:] - boxes_a[:, :2]).clamp(min=0).prod(-1)
    area_b = (boxes_b[:, 2:] - boxes_b[:, :2]).clamp(min=0).prod(-1)
    return intersection / (area_a[:, None] + area_b[None, :] - intersection).clamp(min=1e-9)


def _target_index(row, target, original_width, original_height):
    boxes = target["boxes"]
    boxes = boxes.as_subclass(torch.Tensor) if hasattr(boxes, "as_subclass") else boxes
    height, width = target.get("size", torch.tensor([640, 640], device=boxes.device)).tolist()
    gt = torch.tensor(row["gt_box"], device=boxes.device, dtype=boxes.dtype)
    gt = gt * torch.tensor(
        [width / original_width, height / original_height,
         width / original_width, height / original_height],
        device=boxes.device, dtype=boxes.dtype,
    )
    candidates = torch.nonzero(
        target["labels"] == int(row["gt_class"]), as_tuple=False
    ).flatten()
    if not len(candidates):
        raise RuntimeError(f"annotation {row['ann_id']} class absent after eval transform")
    iou = _pairwise_iou_torch(boxes[candidates], gt[None])[:, 0]
    best = int(candidates[int(iou.argmax())])
    if float(iou.max()) < 0.999:
        raise RuntimeError(f"annotation {row['ann_id']} could not be mapped to transformed target")
    return best


def capture_ffn_gradient_importance(model, samples, items):
    """Return |activation * dL/dactivation| for selected final queries.

    ``items`` contains batch/query/class indices and normalized cxcywh GT boxes.
    The model must already be in eval mode; this function never changes mode.
    """
    ec = model if hasattr(model, "dec_score_head") else model.decoder
    if model.training or ec.training or ec.decoder.training:
        raise RuntimeError("FFN gradient capture requires eval mode")
    activations = {}
    handles = []
    decoder_parameters = [
        parameter for parameter in ec.decoder.parameters()
        if parameter.is_floating_point() or parameter.is_complex()
    ]
    requires_grad_state = [parameter.requires_grad for parameter in decoder_parameters]
    try:
        # EMA checkpoint modules are commonly materialized with every parameter
        # frozen.  Enabling autograd on the decoder temporarily leaves values and
        # eval behavior unchanged while making hidden-activation VJPs available.
        for parameter in decoder_parameters:
            parameter.requires_grad_(True)
        for layer_index, layer in enumerate(ec.decoder.layers):
            def save_activation(_module, _inputs, output, index=layer_index):
                output.retain_grad()
                activations[index] = output
                return None
            handles.append(layer.activation.register_forward_hook(save_activation))
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            outputs = model(samples)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            losses = []
            for item in items:
                batch_index = item["batch_index"]
                query_id = item["query_id"]
                gt_class = item["gt_class"]
                gt = torch.as_tensor(
                    item["gt_box_cxcywh"], device=outputs["pred_boxes"].device,
                    dtype=outputs["pred_boxes"].dtype,
                )
                losses.append(
                    -outputs["pred_logits"][batch_index, query_id, gt_class]
                    + 5.0 * torch.nn.functional.l1_loss(
                        outputs["pred_boxes"][batch_index, query_id], gt,
                        reduction="sum",
                    )
                )
            torch.stack(losses).sum().backward()
        result = []
        for item in items:
            per_layer = []
            for layer_index in range(len(ec.decoder.layers)):
                activation = activations[layer_index]
                value = activation[item["batch_index"], item["query_id"]]
                gradient = activation.grad[item["batch_index"], item["query_id"]]
                per_layer.append((value * gradient).abs().detach().cpu())
            result.append(torch.stack(per_layer))
        return torch.stack(result)
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        for parameter, requires_grad in zip(decoder_parameters, requires_grad_state):
            parameter.requires_grad_(requires_grad)


def collect_arm(args):
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; refusing silent CPU fallback")
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    rows_by_image = {}
    for row in rows:
        rows_by_image.setdefault(int(row["image_id"]), []).append(row)
    needed_images = set(rows_by_image)

    cfg, solver, model = _build(args)
    model.eval()
    records = []
    parity = None
    gradient_done = 0
    gradient_ann_ids = _select_gradient_ann_ids(rows, args.gradient_limit, args.seed)
    started = time.time()
    for samples, targets in cfg.val_dataloader:
        image_ids = [int(target["image_id"].item()) for target in targets]
        if not any(image_id in needed_images for image_id in image_ids):
            continue
        samples = samples.to(args.device)
        with torch.no_grad():
            if parity is None:
                baseline = model(samples)
            with DecoderInternalCapture(model.decoder) as capture:
                outputs = model(samples)
            replay = capture.replay_decoder_heads()
        if parity is None:
            parity = {
                "logits": float((baseline["pred_logits"] - outputs["pred_logits"]).abs().max()),
                "boxes": float((baseline["pred_boxes"] - outputs["pred_boxes"]).abs().max()),
                "replay_logits": float((replay["lqe_logits"][capture.td.eval_idx] - outputs["pred_logits"]).abs().max()),
                "replay_boxes": float((replay["boxes"][capture.td.eval_idx] - outputs["pred_boxes"]).abs().max()),
            }
            if max(parity.values()) > 2e-6:
                raise RuntimeError(f"capture parity failed: {parity}")

        matcher_targets = _normalise_targets(targets, samples.shape[-2], samples.shape[-1], args.device)
        matches = solver.criterion.matcher(outputs, matcher_targets)["indices"]
        snapshot = capture.cpu_snapshot()
        replay_cpu = {key: value.detach().cpu() for key, value in replay.items()}
        outputs_cpu = {key: value.detach().cpu() for key, value in outputs.items() if torch.is_tensor(value)}
        batch_records = []

        for batch_index, (image_id, target) in enumerate(zip(image_ids, targets)):
            if image_id not in needed_images:
                continue
            original_width, original_height = [int(v) for v in target["orig_size"].tolist()]
            target_device = {key: value.to(args.device) if torch.is_tensor(value) else value
                             for key, value in target.items()}
            hungarian = {int(gt): int(query) for query, gt in zip(*matches[batch_index])}
            final_boxes = box_cxcywh_to_xyxy(outputs_cpu["pred_boxes"][batch_index])
            final_logits = outputs_cpu["pred_logits"][batch_index]
            final_probs = final_logits.sigmoid()
            target_norm = matcher_targets[batch_index]
            target_xyxy = box_cxcywh_to_xyxy(target_norm["boxes"]).cpu()

            for row in rows_by_image[image_id]:
                gt_index = _target_index(row, target_device, original_width, original_height)
                gt_box = target_xyxy[gt_index]
                iou = _pairwise_iou_torch(final_boxes, gt_box[None])[:, 0]
                gt_class = int(row["gt_class"])
                gt_scores = final_probs[:, gt_class]
                query_scores = final_probs.max(-1).values
                candidates = torch.nonzero(iou >= args.tau, as_tuple=False).flatten()
                q_detection = int(candidates[gt_scores[candidates].argmax()]) if len(candidates) else -1
                query_ids = [
                    int(iou.argmax()),
                    int(gt_scores.argmax()),
                    int(query_scores.argmax()),
                    q_detection,
                    hungarian[gt_index],
                    int((int(row["ann_id"]) * 2654435761 + args.seed) % len(iou)),
                ]
                gather_ids = [query_id if query_id >= 0 else query_ids[0] for query_id in query_ids]
                layer_records = snapshot["layers"]
                query_stages = []
                ffn_preact, ffn_activation, ffn_linear2 = [], [], []
                offsets, locations, weights = [], [], []
                references = []
                topk_inside, topk_mass = [], []
                top_order = torch.argsort(query_scores, descending=True)
                for layer in layer_records:
                    query_stages.append(torch.stack([
                        layer["query_in"][batch_index, gather_ids],
                        layer["self_attn_output"][batch_index, gather_ids],
                        layer["self_attn_residual"][batch_index, gather_ids],
                        layer["cross_attn_output"][batch_index, gather_ids],
                        layer["cross_attn_residual"][batch_index, gather_ids],
                        layer["query_out"][batch_index, gather_ids],
                    ], dim=1))
                    ffn_preact.append(layer["ffn_linear1"][batch_index, gather_ids])
                    ffn_activation.append(layer["ffn_activation"][batch_index, gather_ids])
                    ffn_linear2.append(layer["ffn_linear2"][batch_index, gather_ids])
                    offsets.append(layer["sampling_offsets"][batch_index, gather_ids])
                    locations.append(layer["sampling_locations"][batch_index, gather_ids])
                    weights.append(layer["attention_weights"][batch_index, gather_ids])
                    references.append(layer["reference_in"][batch_index, gather_ids, 0])
                    all_locations = layer["sampling_locations"][batch_index]
                    all_weights = layer["attention_weights"][batch_index]
                    inside = (
                        (all_locations[..., 0] >= gt_box[0])
                        & (all_locations[..., 0] <= gt_box[2])
                        & (all_locations[..., 1] >= gt_box[1])
                        & (all_locations[..., 1] <= gt_box[3])
                    )
                    layer_inside, layer_mass = [], []
                    for topk in (1, 5, 10):
                        ids = top_order[:topk]
                        layer_inside.append(float(inside[ids].float().mean((-1, -2)).max()))
                        layer_mass.append(float((all_weights[ids] * inside[ids]).sum(-1).mean(-1).max()))
                    topk_inside.append(layer_inside)
                    topk_mass.append(layer_mass)
                record = {
                    "row": row,
                    "query_ids": np.asarray(query_ids, np.int16),
                    "query_stages": torch.stack(query_stages, dim=1).half().numpy(),
                    "ffn_preact": torch.stack(ffn_preact, dim=1).half().numpy(),
                    "ffn_activation": torch.stack(ffn_activation, dim=1).half().numpy(),
                    "ffn_linear2": torch.stack(ffn_linear2, dim=1).half().numpy(),
                    "sampling_offsets": torch.stack(offsets, dim=1).half().numpy(),
                    "sampling_locations": torch.stack(locations, dim=1).half().numpy(),
                    "attention_weights": torch.stack(weights, dim=1).half().numpy(),
                    "references_in": torch.stack(references, dim=1).numpy(),
                    "refined_boxes": replay_cpu["boxes"][:, batch_index, gather_ids].permute(1, 0, 2).numpy(),
                    "raw_logits": replay_cpu["raw_logits"][:, batch_index, gather_ids].permute(1, 0, 2).numpy(),
                    "lqe_logits": replay_cpu["lqe_logits"][:, batch_index, gather_ids].permute(1, 0, 2).numpy(),
                    "final_iou": iou[gather_ids].numpy(),
                    "score_topk_sampling_inside": np.asarray(topk_inside, np.float32).T,
                    "score_topk_attention_inside": np.asarray(topk_mass, np.float32).T,
                    "score_topk_box_iou_max": np.asarray([
                        float(iou[top_order[:topk]].max()) for topk in (1, 5, 10)
                    ], np.float32),
                    "gt_box_norm": gt_box.numpy(),
                    "ffn_grad_importance": np.full(
                        (len(layer_records), layer_records[0]["ffn_activation"].shape[-1]),
                        np.nan, dtype=np.float16,
                    ),
                }
                records.append(record)
                batch_records.append((record, batch_index, query_ids[4], gt_class, gt_box.numpy()))

        eligible = [item for item in batch_records if int(item[0]["row"]["ann_id"]) in gradient_ann_ids]
        if eligible:
            grad_items = []
            for _record, batch_index, query_id, gt_class, gt_xyxy in eligible:
                gt = np.asarray(gt_xyxy, np.float32)
                grad_items.append({
                    "batch_index": batch_index,
                    "query_id": query_id,
                    "gt_class": gt_class,
                    "gt_box_cxcywh": np.concatenate(((gt[:2] + gt[2:]) / 2, gt[2:] - gt[:2])),
                })
            importance = capture_ffn_gradient_importance(model, samples, grad_items)
            for (record, *_rest), value in zip(eligible, importance):
                record["ffn_grad_importance"] = value.half().numpy()
            gradient_done += len(eligible)
    if len(records) != len(rows):
        raise RuntimeError(f"collected {len(records)}/{len(rows)} manifest lesions")
    by_ann = {record["row"]["ann_id"]: record for record in records}
    records = [by_ann[row["ann_id"]] for row in rows]
    payload = {
        "ann_id": np.asarray([r["row"]["ann_id"] for r in records], np.int64),
        "image_id": np.asarray([r["row"]["image_id"] for r in records], np.int64),
        "gt_class": np.asarray([r["row"]["gt_class"] for r in records], np.int8),
        "gt_area": np.asarray([r["row"]["gt_area"] for r in records], np.float32),
        "group": np.asarray([r["row"]["group"] for r in records], object),
        "arm_success": np.asarray([
            [r["row"]["arm_success"][name] for name in ARMS] for r in records
        ], bool),
        "arm_localised": np.asarray([
            [r["row"]["arm_localised"][name] for name in ARMS] for r in records
        ], bool),
        "gt_box_norm": np.stack([r["gt_box_norm"] for r in records]),
    }
    for key in (
        "query_ids", "query_stages", "ffn_preact", "ffn_activation", "ffn_linear2",
        "sampling_offsets", "sampling_locations", "attention_weights", "references_in",
        "refined_boxes", "raw_logits", "lqe_logits", "final_iou",
        "score_topk_sampling_inside", "score_topk_attention_inside",
        "score_topk_box_iou_max",
        "ffn_grad_importance",
    ):
        payload[key] = np.stack([record[key] for record in records])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    meta = {
        "arm": args.arm,
        "split": args.split,
        "n_lesions": len(records),
        "n_images": len(needed_images),
        "query_kinds": QUERY_KINDS,
        "query_stage_order": (
            "input", "self_attn_raw", "self_attn_residual", "cross_attn_raw",
            "cross_attn_residual", "ffn_residual_output",
        ),
        "num_points_list": list(capture.td.layers[0].cross_attn.num_points_list),
        "parity": parity,
        "gradient_importance_lesions": gradient_done,
        "gradient_loss": "-GT-class LQE logit + 5*L1(final cxcywh, GT), summed within batch",
        "seconds": round(time.time() - started, 2),
    }
    Path(str(out).replace(".npz", ".meta.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--pred-dir", default="outputs/ablation/cmp5L_predictions")
    parser.add_argument("--split", default="test")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--target-size", type=int, default=650)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--ann-file")
    parser.add_argument("--img-folder")
    parser.add_argument("--out")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", default="ema")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gradient-limit", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = Path(args.manifest)
    if args.select_only:
        rows = build_manifest(
            args.pred_dir, args.split, args.tau, args.score_threshold,
            args.target_size, args.seed,
        )
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
        counts = {}
        classes = {}
        for row in rows:
            counts[row["group"]] = counts.get(row["group"], 0) + 1
            classes[row["class_name"]] = classes.get(row["class_name"], 0) + 1
        print(json.dumps({"n": len(rows), "groups": counts, "classes": classes}, indent=2))
        return 0
    for name in ("arm", "config", "checkpoint", "ann_file", "img_folder", "out"):
        if not getattr(args, name):
            raise SystemExit(f"--{name.replace('_', '-')} is required for collection")
    return collect_arm(args)


if __name__ == "__main__":
    raise SystemExit(main())
