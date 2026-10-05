"""Experiment 2: collect and classify harmful Far-BG sampling points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ecdetseg"))
sys.path.insert(0, str(ROOT))

from scripts.ablation.cmp5L_bgdep_proxy import binary_metrics
from scripts.ablation.diag_cmp5L_adaptive_suppression import query_tp_fp, region_mask
from scripts.ablation.probe_cmp5L_decoder_internal_tensors import (
    DecoderInternalCapture,
    _build,
    _normalise_targets,
)
from scripts.ablation.train_cmp5L_bgdep_query_predictor import Predictor, best_f1_threshold, predict, train_one
from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy


def capture_sampled_core(rec):
    def core(value, shapes, locations, attention, num_points_list):
        bs, heads, channels, _ = value[0].shape
        queries = locations.shape[1]
        grids = (2 * locations - 1).permute(0, 2, 1, 3, 4).flatten(0, 1)
        sampled_levels = []
        for level, ((height, width), grid) in enumerate(zip(shapes, grids.split(num_points_list, -2))):
            value_level = value[level].reshape(bs * heads, channels, height, width)
            sampled_levels.append(F.grid_sample(value_level, grid, mode="bilinear", padding_mode="zeros", align_corners=False))
        sampled = torch.cat(sampled_levels, -1)
        rec["sampled_values"] = sampled.reshape(bs, heads, channels, queries, -1).permute(0, 3, 1, 4, 2).detach().clone()
        weights = attention.permute(0, 2, 1, 3).reshape(bs * heads, 1, queries, -1)
        return (sampled * weights).sum(-1).reshape(bs, heads * channels, queries).permute(0, 2, 1)
    return core


def _sample_indices(region, per_region, generator):
    flat = region.reshape(-1)
    selected = []
    for region_id in (0, 1, 2):
        candidates = torch.where(flat == region_id)[0]
        if candidates.numel() > per_region:
            order = torch.randperm(candidates.numel(), generator=generator, device=candidates.device)[:per_region]
            candidates = candidates[order]
        selected.append(candidates)
    return torch.cat(selected) if selected else torch.empty(0, dtype=torch.long, device=flat.device)


def collect(args, ann_file, limit, out_file, seed):
    args.ann_file = ann_file
    args.limit = limit
    cfg, solver, model = _build(args)
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(seed)
    fields = {name: [] for name in (
        "query", "value", "geometry", "attention", "region", "layer", "level",
        "head", "query_type", "image_id", "query_idx",
    )}
    seen = 0
    for samples, targets in cfg.val_dataloader:
        if limit and seen >= limit:
            break
        samples = samples.to(device)
        norm_targets = _normalise_targets(targets, samples.shape[-2], samples.shape[-1], device)
        with torch.no_grad(), DecoderInternalCapture(model.decoder) as capture:
            for li in range(3):
                capture.layers[li]["collector_marker"] = True
                model.decoder.decoder.layers[li].cross_attn.ms_deformable_attn_core = capture_sampled_core(capture.layers[li])
            output = model(samples)

        boxes = box_cxcywh_to_xyxy(output["pred_boxes"])
        logits = output["pred_logits"]
        for bi, target in enumerate(norm_targets):
            gt_xyxy = box_cxcywh_to_xyxy(target["boxes"])
            is_tp = query_tp_fp(boxes[bi], logits[bi], gt_xyxy, target["labels"])
            image_id = int(targets[bi]["image_id"].item())
            for li in range(3):
                rec = capture.layers[li]
                locations = rec["sampling_locations"][bi]
                region = region_mask(locations, gt_xyxy)
                selected = _sample_indices(region, args.points_per_region, generator)
                if selected.numel() == 0:
                    continue
                q_count, heads, points = region.shape
                query_idx = selected // (heads * points)
                rem = selected % (heads * points)
                head_idx = rem // points
                point_idx = rem % points
                point_level = torch.empty(points, dtype=torch.long, device=device)
                offset = 0
                for level, count in enumerate(rec["num_points_list"]):
                    point_level[offset:offset + count] = level
                    offset += count
                level_idx = point_level[point_idx]

                query = rec["query_in"][bi, query_idx]
                value = rec["sampled_values"][bi, query_idx, head_idx, point_idx]
                attention = rec["attention_weights"][bi, query_idx, head_idx, point_idx, None]
                offsets = rec["sampling_offsets"][bi, query_idx, head_idx, point_idx]
                reference = rec["reference_in"][bi, query_idx]
                if reference.ndim == 3:
                    reference = reference[:, 0]
                center = reference[:, :2]
                size = reference[:, 2:].clamp_min(1e-4) if reference.shape[-1] == 4 else torch.ones_like(center)
                relative = (locations[query_idx, head_idx, point_idx] - center) / size
                distance = relative.norm(dim=-1, keepdim=True)
                geometry = torch.cat([
                    relative, distance, offsets,
                    level_idx.float().unsqueeze(-1) / max(len(rec["num_points_list"]) - 1, 1),
                    head_idx.float().unsqueeze(-1) / max(heads - 1, 1),
                ], dim=-1)

                tensors = {
                    "query": query, "value": value, "geometry": geometry,
                    "attention": attention, "region": region.reshape(-1)[selected],
                    "layer": torch.full_like(selected, li), "level": level_idx,
                    "head": head_idx, "query_type": is_tp[query_idx].long(),
                    "image_id": torch.full_like(selected, image_id), "query_idx": query_idx,
                }
                for name, tensor in tensors.items():
                    fields[name].append(tensor.detach().cpu().numpy())
        seen += len(targets)
        if seen % 200 < len(targets):
            print(f"point collection: {seen} images", flush=True)

    payload = {name: np.concatenate(chunks, axis=0) for name, chunks in fields.items()}
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_file, **payload)
    print(f"saved {len(payload['region'])} points from {seen} images to {out_file}", flush=True)


def point_features(payload, variant):
    q, value, geometry, attention = (payload[k].astype(np.float32, copy=False) for k in ("query", "value", "geometry", "attention"))
    choices = {
        "q_only": q,
        "value_only": value,
        "q_value": np.concatenate([q, value], 1),
        "q_value_geometry": np.concatenate([q, value, geometry], 1),
        "q_value_geometry_attention": np.concatenate([q, value, geometry, attention], 1),
    }
    return choices[variant]


def train_predictors(args):
    train = np.load(args.train_points)
    valid = np.load(args.valid_points)
    y_train = (train["region"] == 0).astype(np.int8)
    y_valid = (valid["region"] == 0).astype(np.int8)
    variants = ["q_only", "value_only", "q_value", "q_value_geometry", "q_value_geometry_attention"]
    results, best = {}, None
    device = torch.device(args.device)
    for variant in variants:
        xt, xv = point_features(train, variant), point_features(valid, variant)
        model, mean, std = train_one(xt, y_train, "classification", args.hidden, device, args.epochs, args.seed)
        probability = 1 / (1 + np.exp(-predict(model, xv, mean, std, device)))
        threshold = best_f1_threshold(y_valid, probability)
        overall = binary_metrics(y_valid, probability, threshold)
        ring_or_far = valid["region"] != 2
        ring_far = binary_metrics(y_valid[ring_or_far], probability[ring_or_far], threshold)
        grouped = {}
        for group_name in ("layer", "level", "query_type"):
            grouped[group_name] = {}
            for group_value in np.unique(valid[group_name]):
                mask = valid[group_name] == group_value
                grouped[group_name][str(int(group_value))] = binary_metrics(y_valid[mask], probability[mask], threshold)
        result = {"overall": overall, "ring_vs_farbg": ring_far, "grouped": grouped}
        results[variant] = result
        candidate = (ring_far["pr_auc"], overall["pr_auc"], variant)
        if best is None or candidate > best[0]:
            best = (candidate, model, mean, std, threshold, result, xt.shape[1])
        print(json.dumps({variant: {"overall": overall, "ring_vs_farbg": ring_far}}, indent=2), flush=True)

    candidate, model, mean, std, threshold, result, input_dim = best
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(), "variant": candidate[2], "input_dim": input_dim,
        "hidden": args.hidden, "mean": mean.cpu(), "std": std.cpu(),
        "threshold": threshold, "metrics": result, "seed": args.seed,
    }, out_dir / "best_point_predictor.pth")
    payload = {
        "experiment": 2,
        "split_policy": "train labels fit; validation model/threshold selection; test untouched",
        "best": {"variant": candidate[2], **result},
        "results": results,
        "counts": {"train": int(len(y_train)), "valid": int(len(y_valid))},
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "COMPLETED.txt").write_text("status=passed\n", encoding="utf-8")


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not Path(args.train_points).exists():
        collect(args, args.train_ann, args.train_limit, args.train_points, args.seed)
    if not Path(args.valid_points).exists():
        collect(args, args.valid_ann, args.valid_limit, args.valid_points, args.seed + 1)
    train_predictors(args)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--img-folder", required=True)
    p.add_argument("--train-ann", required=True)
    p.add_argument("--valid-ann", required=True)
    p.add_argument("--train-points", required=True)
    p.add_argument("--valid-points", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--train-limit", type=int, default=1200)
    p.add_argument("--valid-limit", type=int, default=1200)
    p.add_argument("--points-per-region", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--weights", default="ema")
    p.add_argument("--device", default="cuda")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
