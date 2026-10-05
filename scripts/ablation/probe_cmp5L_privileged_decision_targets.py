"""Stage-1 frozen target-quality probe for cmp5L privileged decision KD."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from scripts.ablation.cmp5L_privileged_decision_kd import (
    build_mixed_oracle_output,
    max_named_buffer_change,
    run_with_all_score_heads,
    select_outcome_targets,
)
from scripts.ablation.evaluate_cmp5L_af_validation import (
    _merge_tensor_records,
    _prediction_record,
    _summarize_evaluator,
    checkpoint_key_audit,
    json_compatible_args,
    normal_bg_dependency,
    region_mask_with_outside,
    run_af_branch,
    sha256_path,
    topk_query_predictions,
)
from scripts.ablation.probe_cmp5L_decoder_internal_tensors import (
    DecoderInternalCapture,
    _build,
    _normalise_targets,
)
from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy


CONDITIONS = ("N", "P", "M")
BGDEP_THRESHOLD = 0.20
SCORE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.50
NEGATIVE_IOU_THRESHOLD = 0.30


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fixed_tensor_ids(manifest, per_stratum=2):
    groups = defaultdict(list)
    for row in manifest["images"]:
        groups[row["stratum"]].append(int(row["image_id"]))
    return {
        image_id
        for stratum in ("empty", "duct", "lymph", "cystic", "solid", "hard")
        for image_id in groups[stratum][:per_stratum]
    }


def _layer_metrics(layer_logits, query, label):
    result = []
    for logits in layer_logits:
        scores = logits[query].sigmoid()
        gt_score = float(scores[label])
        other = torch.cat((scores[:label], scores[label + 1:]))
        margin = gt_score - (float(other.max()) if other.numel() else 0.0)
        result.append({"gt_score": gt_score, "gt_margin": margin})
    return result


def _paired_iou(box, target):
    left = box_cxcywh_to_xyxy(box[None])[0]
    right = box_cxcywh_to_xyxy(target[None])[0]
    top_left = torch.maximum(left[:2], right[:2])
    bottom_right = torch.minimum(left[2:], right[2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod()
    union = (
        (left[2:] - left[:2]).clamp(min=0).prod()
        + (right[2:] - right[:2]).clamp(min=0).prod()
        - intersection
    )
    return float(intersection / union.clamp(min=1e-12))


def _enrich_selected(selected, normal, privileged, n_layers, p_layers, targets):
    enriched = []
    for batch_index, current in enumerate(selected):
        target = targets[batch_index]
        row = {
            "positive": [],
            "negative": [],
            "counts": current["counts"],
            "normal_gt_matches": current["normal_gt_matches"],
            "privileged_gt_matches": current["privileged_gt_matches"],
        }
        for item in current["positive"]:
            gt_index = int(item["gt_index"])
            query = int(item["teacher_query"])
            label = int(target["labels"][gt_index])
            branch = normal if item["branch"] == "N" else privileged
            layers = n_layers if item["branch"] == "N" else p_layers
            score = float(branch["pred_logits"][batch_index, query, label].sigmoid())
            iou = _paired_iou(
                branch["pred_boxes"][batch_index, query], target["boxes"][gt_index]
            )
            flattened = branch["pred_logits"][batch_index].sigmoid().flatten()
            rank = 1 + int((flattened > score).sum())
            row["positive"].append({
                **item,
                "gt_label": label,
                "teacher_score": score,
                "teacher_iou": iou,
                "teacher_flat_rank": rank,
                "threshold_boundary": abs(score - SCORE_THRESHOLD) <= 0.05 or abs(iou - IOU_THRESHOLD) <= 0.05,
                "layers": _layer_metrics(
                    [layer[batch_index] for layer in layers], query, label
                ),
            })
        for item in current["negative"]:
            query, label = int(item["teacher_query"]), int(item["class_index"])
            row["negative"].append({
                **item,
                "normal_flat_rank": 1 + int(
                    (normal["pred_logits"][batch_index].sigmoid().flatten() > item["normal_score"]).sum()
                ),
                "threshold_boundary": abs(float(item["normal_score"]) - SCORE_THRESHOLD) <= 0.05,
                "normal_layers": _layer_metrics(
                    [layer[batch_index] for layer in n_layers], query, label
                ),
                "privileged_layers": _layer_metrics(
                    [layer[batch_index] for layer in p_layers], query, label
                ),
            })
        enriched.append(row)
    return enriched


def _run_branch(model, decoder_input, condition, gt_xyxy, has_gt, gate, image_ids, tensor_ids, audit):
    holder = {}

    def branch():
        output, statistics, tensors = run_af_branch(
            model.decoder,
            decoder_input,
            condition,
            gt_xyxy,
            has_gt,
            gate,
            torch.zeros_like(gate),
            image_ids,
            tensor_ids,
            audit,
        )
        holder["statistics"] = statistics
        holder["tensors"] = tensors
        return output

    output, layer_logits = run_with_all_score_heads(model.decoder, branch)
    return output, layer_logits, holder["statistics"], holder["tensors"]


def evaluate(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    args.device = device
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    subset_manifest = json.loads(Path(args.subset_manifest).read_text(encoding="utf-8"))
    tensor_ids = _fixed_tensor_ids(subset_manifest)

    cfg, solver, model = _build(args)
    model.eval().requires_grad_(False)
    model_load = checkpoint_key_audit(model, args.checkpoint, args.weights)
    if model_load["missing_keys"] or model_load["unexpected_keys"]:
        raise RuntimeError(f"model checkpoint key mismatch: {model_load}")
    buffer_before = {name: value.detach().clone() for name, value in model.named_buffers()}
    evaluators = {name: copy.deepcopy(cfg.evaluator) for name in CONDITIONS}
    for evaluator in evaluators.values():
        evaluator.cleanup()
    streams = {
        name: (output_dir / f"predictions_{name}.jsonl").open("x", encoding="utf-8")
        for name in CONDITIONS
    }
    target_stream = (output_dir / "selected_targets.jsonl").open("x", encoding="utf-8")
    audit = {
        "model_load": model_load,
        "model_eval": not model.training,
        "all_teacher_parameters_frozen": all(not value.requires_grad for value in model.parameters()),
        "normal_N_max_abs": {"logits": 0.0, "boxes": 0.0},
        "captured_final_max_abs": {"N": 0.0, "P": 0.0},
        "empty_P_identity_max_abs": {"logits": 0.0, "boxes": 0.0, "layers": 0.0},
        "location_input_mutation_max_abs": 0.0,
        "attention_input_mutation_max_abs": 0.0,
        "historical_coeff_max_abs": {"B": [None] * 3, "C": [None] * 3},
        "coeff_min": {"A": [float("inf")] * 3, "E": [float("inf")] * 3},
        "coeff_max": {"A": [float("-inf")] * 3, "E": [float("-inf")] * 3},
        "coeff_shapes": {"A": [None] * 3, "E": [None] * 3},
        "finite_outputs": {"A": True, "E": True},
        "direct_layers": [0, 1, 2],
    }
    raw = {
        branch: {f"L{layer}": [] for layer in range(4)}
        for branch in ("N", "P")
    }
    raw_boxes = {"N": [], "P": []}
    layer_stats = {branch: [[] for _ in range(3)] for branch in ("N", "P")}
    tensor_records = {}
    target_counts = Counter()
    target_class_counts = defaultdict(Counter)
    seen_ids = []
    started = time.time()
    first_batch = True
    try:
        with torch.no_grad():
            for samples, targets in cfg.val_dataloader:
                samples = samples.to(device)
                image_ids = [int(target["image_id"].item()) for target in targets]
                normalized = _normalise_targets(
                    targets, samples.shape[-2], samples.shape[-1], device
                )
                gt_xyxy = [box_cxcywh_to_xyxy(target["boxes"]) for target in normalized]
                has_gt = torch.tensor([bool(boxes.shape[0]) for boxes in gt_xyxy], device=device)
                normal_live = model(samples)
                with DecoderInternalCapture(model.decoder) as capture:
                    normal_hooked = model(samples)
                if first_batch:
                    if not torch.equal(normal_live["pred_logits"], normal_hooked["pred_logits"]):
                        raise RuntimeError("normal capture changed logits")
                    if not torch.equal(normal_live["pred_boxes"], normal_hooked["pred_boxes"]):
                        raise RuntimeError("normal capture changed boxes")
                attention, regions = [], []
                for layer in range(3):
                    record = capture.layers[layer]
                    attention.append(record["attention_weights"])
                    regions.append(torch.stack([
                        region_mask_with_outside(
                            record["sampling_locations"][batch], gt_xyxy[batch]
                        )[0]
                        for batch in range(len(image_ids))
                    ]))
                dependency = normal_bg_dependency(attention, regions)
                gate = dependency > BGDEP_THRESHOLD
                n_out, n_layers, n_stats, n_tensors = _run_branch(
                    model, capture.decoder_input, "A", gt_xyxy, has_gt, gate,
                    image_ids, tensor_ids, audit,
                )
                p_out, p_layers, p_stats, p_tensors = _run_branch(
                    model, capture.decoder_input, "E", gt_xyxy, has_gt, gate,
                    image_ids, tensor_ids, audit,
                )
                audit["normal_N_max_abs"]["logits"] = max(
                    audit["normal_N_max_abs"]["logits"],
                    float((normal_live["pred_logits"] - n_out["pred_logits"]).abs().max()),
                )
                audit["normal_N_max_abs"]["boxes"] = max(
                    audit["normal_N_max_abs"]["boxes"],
                    float((normal_live["pred_boxes"] - n_out["pred_boxes"]).abs().max()),
                )
                for name, output, layers in (("N", n_out, n_layers), ("P", p_out, p_layers)):
                    audit["captured_final_max_abs"][name] = max(
                        audit["captured_final_max_abs"][name],
                        float((layers[-1] - output["pred_logits"]).abs().max()),
                    )
                empty = ~has_gt
                if bool(empty.any()):
                    audit["empty_P_identity_max_abs"]["logits"] = max(
                        audit["empty_P_identity_max_abs"]["logits"],
                        float((p_out["pred_logits"][empty] - n_out["pred_logits"][empty]).abs().max()),
                    )
                    audit["empty_P_identity_max_abs"]["boxes"] = max(
                        audit["empty_P_identity_max_abs"]["boxes"],
                        float((p_out["pred_boxes"][empty] - n_out["pred_boxes"][empty]).abs().max()),
                    )
                    audit["empty_P_identity_max_abs"]["layers"] = max(
                        audit["empty_P_identity_max_abs"]["layers"],
                        max(float((p[empty] - n[empty]).abs().max()) for n, p in zip(n_layers, p_layers)),
                    )
                selected = select_outcome_targets(
                    n_out, p_out, normalized,
                    score_threshold=SCORE_THRESHOLD,
                    iou_threshold=IOU_THRESHOLD,
                    negative_iou_threshold=NEGATIVE_IOU_THRESHOLD,
                )
                mixed = build_mixed_oracle_output(n_out, p_out, selected)
                enriched = _enrich_selected(selected, n_out, p_out, n_layers, p_layers, normalized)
                for batch, (image_id, record) in enumerate(zip(image_ids, enriched)):
                    for key, value in record["counts"].items():
                        target_counts[key] += int(value)
                    for item in record["positive"]:
                        target_class_counts[item["kind"]][str(item["gt_label"])] += 1
                    for item in record["negative"]:
                        target_class_counts["suppress"][str(item["class_index"])] += 1
                    target_stream.write(json.dumps({
                        "image_id": image_id,
                        "gt_labels": normalized[batch]["labels"].detach().cpu().tolist(),
                        "gt_boxes_cxcywh": normalized[batch]["boxes"].detach().cpu().tolist(),
                        **record,
                    }, ensure_ascii=False) + "\n")
                for branch, layers, boxes, stats, tensors in (
                    ("N", n_layers, n_out["pred_boxes"], n_stats, n_tensors),
                    ("P", p_layers, p_out["pred_boxes"], p_stats, p_tensors),
                ):
                    for layer, logits in enumerate(layers):
                        raw[branch][f"L{layer}"].append(logits.detach().cpu().to(torch.float16).numpy())
                    raw_boxes[branch].append(boxes.detach().cpu().to(torch.float16).numpy())
                    for layer, values in enumerate(stats):
                        layer_stats[branch][layer].append(values)
                    _merge_tensor_records(tensor_records, tensors)
                original_sizes = torch.stack([target["orig_size"] for target in targets]).to(device)
                for name, output in (("N", n_out), ("P", p_out), ("M", mixed)):
                    official = solver.postprocessor(output, original_sizes)
                    identified = topk_query_predictions(output, original_sizes, topk=300)
                    if first_batch:
                        for left, right in zip(official, identified):
                            if not (
                                torch.equal(left["labels"], right["labels"])
                                and torch.allclose(left["scores"], right["scores"])
                                and torch.allclose(left["boxes"], right["boxes"])
                            ):
                                raise RuntimeError("query-aware postprocess differs from official")
                    evaluators[name].update({
                        image_id: result for image_id, result in zip(image_ids, official)
                    })
                    for image_id, result in zip(image_ids, identified):
                        streams[name].write(json.dumps(_prediction_record(image_id, result)) + "\n")
                seen_ids.extend(image_ids)
                first_batch = False
                if args.log_every and len(seen_ids) % args.log_every < len(image_ids):
                    print(f"[{len(seen_ids)}/{args.expected_images}] {time.time()-started:.1f}s", flush=True)
    finally:
        target_stream.close()
        for stream in streams.values():
            stream.close()

    if len(seen_ids) != args.expected_images or len(set(seen_ids)) != len(seen_ids):
        raise RuntimeError(
            f"coverage failed: seen={len(seen_ids)} unique={len(set(seen_ids))} expected={args.expected_images}"
        )
    if set(map(int, tensor_records)) != tensor_ids:
        raise RuntimeError("fixed tensor subset coverage failed")
    buffer_diff = max_named_buffer_change(buffer_before, model.named_buffers())
    audit["teacher_buffer_max_abs_change"] = buffer_diff
    parity_values = [
        *audit["normal_N_max_abs"].values(),
        *audit["captured_final_max_abs"].values(),
        *audit["empty_P_identity_max_abs"].values(),
        audit["location_input_mutation_max_abs"],
        audit["attention_input_mutation_max_abs"],
        buffer_diff,
    ]
    if max(parity_values) > 2e-6 or not audit["all_teacher_parameters_frozen"]:
        raise RuntimeError(f"technical parity/freeze gate failed: {audit}")
    if not all(audit["finite_outputs"].values()):
        raise RuntimeError("non-finite N/P output")

    metrics = {name: _summarize_evaluator(evaluator) for name, evaluator in evaluators.items()}
    arrays = {"image_ids": np.asarray(seen_ids, dtype=np.int64)}
    for branch in ("N", "P"):
        for layer in range(4):
            arrays[f"{branch}_L{layer}_logits"] = np.concatenate(raw[branch][f"L{layer}"], axis=0)
        arrays[f"{branch}_final_boxes"] = np.concatenate(raw_boxes[branch], axis=0)
    np.savez_compressed(output_dir / "teacher_outputs.npz", **arrays)
    np.savez_compressed(
        output_dir / "layer_query_stats.npz",
        image_ids=np.asarray(seen_ids, dtype=np.int64),
        **{
            f"{branch}_L{layer}": np.concatenate(layer_stats[branch][layer], axis=0)
            for branch in ("N", "P") for layer in range(3)
        },
    )
    torch.save(tensor_records, output_dir / "fixed_tensor_subset.pt")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    audit.update({
        "n_images": len(seen_ids),
        "unique_images": len(set(seen_ids)),
        "tensor_subset": sorted(tensor_ids),
        "target_counts": dict(target_counts),
        "target_class_counts": {key: dict(value) for key, value in target_class_counts.items()},
        "elapsed_seconds": round(time.time() - started, 2),
    })
    (output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "scope": "train-subset frozen baseline EMA N/P target-quality probe",
        "args": json_compatible_args(args),
        "conditions": {
            "N": "Normal frozen original baseline EMA",
            "P": "GT d_BG>0.20 gate; all sampled values x0.2; decoder L0-L2; empty identity",
            "M": "GT-selected output-level mixed Oracle; N boxes; classification edits only",
        },
        "thresholds": {
            "score": SCORE_THRESHOLD,
            "positive_iou": IOU_THRESHOLD,
            "negative_max_iou": NEGATIVE_IOU_THRESHOLD,
            "bg_dependency": BGDEP_THRESHOLD,
        },
        "hashes": {
            "runner": sha256_path(__file__),
            "target_module": sha256_path(Path(__file__).with_name("cmp5L_privileged_decision_kd.py")),
            "checkpoint": sha256_path(args.checkpoint),
            "config": sha256_path(args.config),
            "subset_annotations": sha256_path(args.ann_file),
            "subset_manifest": sha256_path(args.subset_manifest),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "COMPLETED.txt").write_text("status=passed\n", encoding="utf-8")
    print(json.dumps({"metrics": metrics, "audit": audit}, ensure_ascii=False)[:12000], flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--subset-manifest", required=True)
    parser.add_argument("--img-folder", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", choices=("ema",), default="ema")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--expected-images", type=int, default=640)
    parser.add_argument("--log-every", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
