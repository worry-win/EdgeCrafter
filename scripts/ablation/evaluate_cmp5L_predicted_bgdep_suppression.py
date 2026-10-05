"""Experiment 3: GT-free predicted adaptive evidence suppression on test."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ecdetseg"))
sys.path.insert(0, str(ROOT))

from scripts.ablation.cmp5L_bgdep_proxy import point_suppression_coeff
from scripts.ablation.diag_cmp5L_adaptive_suppression import run_suppressed, suppression_core
from scripts.ablation.diag_cmp5L_collect_bgdep import _capture_core_factory, _query_vector
from scripts.ablation.probe_cmp5L_decoder_internal_tensors import DecoderInternalCapture, _build, _normalise_targets
from scripts.ablation.train_cmp5L_bgdep_query_predictor import Predictor, feature_matrix
from engine.solver.ec_engine import summarize_yolo_pr_curve_metrics
from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy


def load_predictor(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = Predictor(checkpoint["input_dim"], checkpoint.get("hidden", 0)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    checkpoint["mean"] = checkpoint["mean"].to(device)
    checkpoint["std"] = checkpoint["std"].to(device)
    return model, checkpoint


def query_probabilities(capture, output, predictor, spec, device):
    batch, queries = output["pred_logits"].shape[:2]
    probabilities = []
    scores = output["pred_logits"].sigmoid()
    boxes = output["pred_boxes"]
    for bi in range(batch):
        rows_emb, rows_stats = [], []
        top2 = scores[bi].topk(2, -1).values
        for qi in range(queries):
            emb, stats = _query_vector(
                capture, bi, qi, float(scores[bi, qi].max()),
                float(top2[qi, 0] - top2[qi, 1]),
                float(boxes[bi, qi, 2] * boxes[bi, qi, 3]),
                float(boxes[bi, qi, 2] / (boxes[bi, qi, 3] + 1e-9)),
            )
            rows_emb.append(emb)
            rows_stats.append(stats)
        payload = {"emb": np.stack(rows_emb), "stats": np.stack(rows_stats)}
        x = torch.from_numpy(feature_matrix(payload, spec["variant"])).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            probabilities.append(torch.sigmoid(predictor((x - spec["mean"]) / spec["std"])))
    return torch.stack(probabilities)


def _torch_point_features(query, value, geometry, attention, variant):
    options = {
        "q_only": query,
        "value_only": value,
        "q_value": torch.cat([query, value], -1),
        "q_value_geometry": torch.cat([query, value, geometry], -1),
        "q_value_geometry_attention": torch.cat([query, value, geometry, attention], -1),
    }
    return options[variant]


def run_predicted(ec, content, ref_unact, memory, shapes, strategy, query_probability,
                  point_predictor, point_spec, enabled=True):
    saved, handles = [], []
    layer_runtime = [dict() for _ in ec.decoder.layers]
    for li in range(3):
        module = ec.decoder.layers[li].cross_attn
        original = module.ms_deformable_attn_core
        saved.append((module, original))

        def make_pre(index):
            def pre(_module, inputs):
                layer_runtime[index]["query"] = inputs[0]
                layer_runtime[index]["reference"] = inputs[1]
            return pre

        handles.append(module.register_forward_pre_hook(make_pre(li)))

        def make_offsets(index, heads_count, point_count):
            def hook(_module, _inputs, output):
                layer_runtime[index]["offsets"] = output.reshape(
                    output.shape[0], output.shape[1], heads_count, point_count, 2
                )
            return hook

        handles.append(module.sampling_offsets.register_forward_hook(
            make_offsets(li, module.num_heads, sum(module.num_points_list))
        ))

        def make_core(index):
            def core(value, spatial_shapes, locations, attention_weights, num_points_list):
                bs, heads, channels, _ = value[0].shape
                queries, points = locations.shape[1], locations.shape[3]
                grids = (2 * locations - 1).permute(0, 2, 1, 3, 4).flatten(0, 1)
                sampled_levels = []
                for level, ((height, width), grid) in enumerate(zip(spatial_shapes, grids.split(num_points_list, -2))):
                    sampled_levels.append(F.grid_sample(
                        value[level].reshape(bs * heads, channels, height, width), grid,
                        mode="bilinear", padding_mode="zeros", align_corners=False,
                    ))
                sampled = torch.cat(sampled_levels, -1)
                sampled_points = sampled.reshape(bs, heads, channels, queries, points).permute(0, 3, 1, 4, 2)

                if not enabled:
                    coeff = torch.ones(bs, queries, heads, points, device=locations.device)
                elif strategy == "query_gate":
                    high = (query_probability > float(point_spec.get("query_threshold", 0.5))).float()
                    coeff = 1.0 - 0.8 * high[:, :, None, None].expand(-1, -1, heads, points)
                else:
                    query = layer_runtime[index]["query"][:, :, None, None, :].expand(-1, -1, heads, points, -1)
                    reference = layer_runtime[index]["reference"]
                    if reference.ndim == 4:
                        reference = reference[:, :, 0]
                    center = reference[..., :2]
                    size = reference[..., 2:].clamp_min(1e-4) if reference.shape[-1] == 4 else torch.ones_like(center)
                    relative = (locations - center[:, :, None, None, :]) / size[:, :, None, None, :]
                    distance = relative.norm(dim=-1, keepdim=True)
                    level_id = torch.empty(points, dtype=locations.dtype, device=locations.device)
                    start = 0
                    for level, count in enumerate(num_points_list):
                        level_id[start:start + count] = level / max(len(num_points_list) - 1, 1)
                        start += count
                    level_feature = level_id[None, None, None, :, None].expand(bs, queries, heads, -1, -1)
                    head_id = torch.arange(heads, device=locations.device, dtype=locations.dtype) / max(heads - 1, 1)
                    head_feature = head_id[None, None, :, None, None].expand(bs, queries, -1, points, -1)
                    offsets = layer_runtime[index]["offsets"]
                    geometry = torch.cat([relative, distance, offsets, level_feature, head_feature], -1)
                    attention = attention_weights.unsqueeze(-1)
                    x = _torch_point_features(query, sampled_points, geometry, attention, point_spec["variant"])
                    probability = torch.sigmoid(point_predictor((x - point_spec["mean"]) / point_spec["std"]))
                    qdep = query_probability if strategy == "joint_gate" else None
                    coeff = point_suppression_coeff(probability, qdep)
                return suppression_core(value, spatial_shapes, locations, attention_weights, num_points_list, coeff)
            return core
        module.ms_deformable_attn_core = make_core(li)

    try:
        return run_suppressed(
            ec, content, ref_unact, memory, shapes, set(), "NN", None,
            [], None, None, None,
        )
    finally:
        for handle in handles:
            handle.remove()
        for module, original in saved:
            module.ms_deformable_attn_core = original


def summarize(evaluator):
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    ce = evaluator.coco_eval["bbox"]
    yolo = summarize_yolo_pr_curve_metrics(ce, evaluator.coco_gt) or {}
    overall = yolo.get("yolo_overall") or {}
    precision = ce.eval["precision"]
    iou50 = int(np.argmin(np.abs(ce.params.iouThrs - 0.50)))
    per_class = {}
    for ki, cat_id in enumerate(ce.params.catIds):
        values = precision[iou50, :, ki, 0, -1]
        per_class[str(cat_id)] = float(values[values > -1].mean()) if (values > -1).any() else -1.0
    return {
        "ap": float(ce.stats[0]), "ap50": float(ce.stats[1]), "ap75": float(ce.stats[2]),
        "precision": float(overall.get("precision", 0)), "recall": float(overall.get("recall", 0)),
        "f1": float(overall.get("f1", 0)), "ar": float(ce.stats[8]),
        "small_ap": float(ce.stats[3]), "medium_ap": float(ce.stats[4]), "large_ap": float(ce.stats[5]),
        "per_class_ap50": per_class,
    }


def main(args):
    device = torch.device(args.device)
    query_model, query_spec = load_predictor(args.query_predictor, device)
    point_model, point_spec = load_predictor(args.point_predictor, device)
    point_spec["query_threshold"] = query_spec["threshold"]
    cfg, solver, model = _build(args)
    evaluators = {name: copy.deepcopy(cfg.evaluator) for name in ("NN", "predicted_query", "predicted_point", "predicted_joint")}
    for evaluator in evaluators.values():
        evaluator.cleanup()
    parity = None
    seen = 0
    for samples, targets in cfg.val_dataloader:
        samples = samples.to(device)
        with torch.no_grad():
            baseline = model(samples)
            saved = []
            with DecoderInternalCapture(model.decoder) as capture:
                for li in range(3):
                    module = model.decoder.decoder.layers[li].cross_attn
                    saved.append((module, module.ms_deformable_attn_core))
                    module.ms_deformable_attn_core = _capture_core_factory(capture.layers[li])
                normal = model(samples)
            for module, core in saved:
                module.ms_deformable_attn_core = core
            if parity is None:
                parity = {
                    "capture_logits_max_abs": float((baseline["pred_logits"] - normal["pred_logits"]).abs().max()),
                    "capture_boxes_max_abs": float((baseline["pred_boxes"] - normal["pred_boxes"]).abs().max()),
                }
                if max(parity.values()) > 2e-6:
                    raise RuntimeError(f"normal capture parity failed: {parity}")
            qprob = query_probabilities(capture, normal, query_model, query_spec, device)
            decoder_input = capture.decoder_input
            predicted = {
                "predicted_query": run_predicted(model.decoder, decoder_input["query_content"], decoder_input["reference_unact"], decoder_input["memory"], decoder_input["spatial_shapes"], "query_gate", qprob, point_model, point_spec),
                "predicted_point": run_predicted(model.decoder, decoder_input["query_content"], decoder_input["reference_unact"], decoder_input["memory"], decoder_input["spatial_shapes"], "point_gate", qprob, point_model, point_spec),
                "predicted_joint": run_predicted(model.decoder, decoder_input["query_content"], decoder_input["reference_unact"], decoder_input["memory"], decoder_input["spatial_shapes"], "joint_gate", qprob, point_model, point_spec),
            }
            if seen == 0:
                identity = run_predicted(model.decoder, decoder_input["query_content"], decoder_input["reference_unact"], decoder_input["memory"], decoder_input["spatial_shapes"], "point_gate", qprob, point_model, point_spec, enabled=False)
                parity.update({
                    "lambda1_logits_max_abs": float((normal["pred_logits"] - identity["pred_logits"]).abs().max()),
                    "lambda1_boxes_max_abs": float((normal["pred_boxes"] - identity["pred_boxes"]).abs().max()),
                })
                if max(parity.values()) > 2e-6:
                    raise RuntimeError(f"lambda=1 parity failed: {parity}")
            outputs = {"NN": normal, **predicted}
            sizes = torch.stack([target["orig_size"] for target in targets]).to(device)
            for name, output in outputs.items():
                results = solver.postprocessor(output, sizes)
                evaluators[name].update({int(t["image_id"].item()): r for t, r in zip(targets, results)})
        seen += len(targets)
        if seen % 400 < len(targets):
            print(f"evaluated {seen} images", flush=True)

    metrics = {name: summarize(evaluator) for name, evaluator in evaluators.items()}
    payload = {
        "experiment": 3, "n_images": seen, "test_gt_used_for_predictor": False,
        "threshold_source": "validation checkpoints", "parity": parity, "metrics": metrics,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out.with_name("COMPLETED.txt").write_text("status=passed\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ann-file", required=True)
    p.add_argument("--img-folder", required=True)
    p.add_argument("--query-predictor", required=True)
    p.add_argument("--point-predictor", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--weights", default="ema")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=6)
    main(p.parse_args())
