"""Fixed-subset query responsibility and hidden-cosine audit for KQ0--KQ3."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import torchvision


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ecdetseg"))
sys.path.insert(0, str(ROOT))

from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy  # noqa: E402
from scripts.ablation.cmp5L_query_behavior_kd import run_model_with_capture  # noqa: E402
from scripts.ablation.cmp5L_shared_query_kd import (  # noqa: E402
    SharedQueryReplay,
    normal_query_slice,
)
from scripts.ablation.probe_cmp5L_decoder_internal_tensors import (  # noqa: E402
    _build,
    _normalise_targets,
)
from scripts.ablation.probe_cmp5L_privileged_ema_oracle import privilege_features  # noqa: E402
from scripts.ablation.probe_cmp5L_shared_query_kd_stage1 import (  # noqa: E402
    _capture_topk_indices,
    _restore_topk,
    _run_direct_teacher,
)


GROUPS = ("KQ0", "KQ1", "KQ2", "KQ3")


def select_stratified_image_ids(ground_truth, *, per_class=24, empty=32, total=160):
    """Pre-result deterministic coverage: each class, empty images, then ID fill."""
    image_ids = sorted(int(item["id"]) for item in ground_truth["images"])
    classes_by_image = defaultdict(set)
    for annotation in ground_truth["annotations"]:
        classes_by_image[int(annotation["image_id"])].add(int(annotation["category_id"]))
    selected = []
    selected_set = set()

    def add(image_id):
        if image_id not in selected_set and len(selected) < total:
            selected.append(image_id)
            selected_set.add(image_id)

    for category_id in range(4):
        count = 0
        for image_id in image_ids:
            if category_id in classes_by_image[image_id] and image_id not in selected_set:
                add(image_id)
                count += 1
                if count >= per_class:
                    break
    count = 0
    for image_id in image_ids:
        if not classes_by_image[image_id] and image_id not in selected_set:
            add(image_id)
            count += 1
            if count >= empty:
                break
    for image_id in image_ids:
        add(image_id)
        if len(selected) >= total:
            break
    if len(selected) != min(total, len(image_ids)):
        raise ValueError("unable to construct requested fixed subset")
    return selected


def _responsibility_counts(baseline, candidate):
    baseline_by_key = {item["key"]: item for item in baseline}
    candidate_by_key = {item["key"]: item for item in candidate}
    keys = sorted(set(baseline_by_key) & set(candidate_by_key))
    return {
        "count": len(keys),
        "same_query_slot": sum(
            baseline_by_key[key]["query_slot"] == candidate_by_key[key]["query_slot"]
            for key in keys
        ),
        "same_anchor_index": sum(
            baseline_by_key[key]["anchor_index"] == candidate_by_key[key]["anchor_index"]
            for key in keys
        ),
        "stable_detected": sum(
            baseline_by_key[key]["hit"] and candidate_by_key[key]["hit"] for key in keys
        ),
        "rescued": sum(
            not baseline_by_key[key]["hit"] and candidate_by_key[key]["hit"] for key in keys
        ),
        "destroyed": sum(
            baseline_by_key[key]["hit"] and not candidate_by_key[key]["hit"] for key in keys
        ),
        "stable_missed": sum(
            not baseline_by_key[key]["hit"] and not candidate_by_key[key]["hit"] for key in keys
        ),
    }


def compare_responsibility(baseline, candidate):
    result = {"all": _responsibility_counts(baseline, candidate)}
    for category_id in range(4):
        result[f"class_{category_id}"] = _responsibility_counts(
            [item for item in baseline if int(item["class"]) == category_id],
            [item for item in candidate if int(item["class"]) == category_id],
        )
    return result


def _subset_coco(ground_truth, selected):
    selected_set = set(map(int, selected))
    return {
        **{key: value for key, value in ground_truth.items() if key not in ("images", "annotations")},
        "images": [item for item in ground_truth["images"] if int(item["id"]) in selected_set],
        "annotations": [
            item for item in ground_truth["annotations"]
            if int(item["image_id"]) in selected_set
        ],
    }


def _model_args(config, checkpoint, ann_file, args):
    return SimpleNamespace(
        config=str(config),
        checkpoint=str(checkpoint),
        ann_file=str(ann_file),
        img_folder=str(args.img_folder),
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        weights="ema",
        device=torch.device(args.device),
    )


def _absolute_targets(raw_targets, device):
    return [
        {
            "boxes": target["boxes"].as_subclass(torch.Tensor).to(device),
            "labels": target["labels"].to(device),
        }
        for target in raw_targets
    ]


def _cosines(student_layers, teacher_layers):
    return [
        F.cosine_similarity(student.float(), teacher.float(), dim=-1)
        for student, teacher in zip(student_layers, teacher_layers)
    ]


def _iou_pair(prediction, target):
    pred_xyxy = box_cxcywh_to_xyxy(prediction.reshape(1, 4))
    target_xyxy = box_cxcywh_to_xyxy(target.reshape(1, 4))
    return float(torchvision.ops.box_iou(pred_xyxy, target_xyxy)[0, 0])


def _group_probe(group, config, checkpoint, subset_file, teacher, args):
    device = torch.device(args.device)
    cfg, solver, student = _build(_model_args(config, checkpoint, subset_file, args))
    student.eval().requires_grad_(False)
    rows = []
    image_topk = {}
    layer_sums = {
        mode: [{"sum": 0.0, "sumsq": 0.0, "count": 0} for _ in range(4)]
        for mode in ("normal", "privileged")
    }
    teacher_topk_calls = 0
    student_seconds = 0.0
    images = 0
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for samples, raw_targets in cfg.val_dataloader:
            samples = samples.to(device)
            matcher_targets = _normalise_targets(
                raw_targets, samples.shape[-2], samples.shape[-1], device
            )
            absolute_targets = _absolute_targets(raw_targets, device)
            original, holder = _capture_topk_indices(student.decoder)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            try:
                student_outputs, student_trace, replay_inputs = run_model_with_capture(
                    student, samples, None
                )
            finally:
                _restore_topk(student.decoder, original)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            student_seconds += time.perf_counter() - started
            if holder["indices"] is None:
                raise RuntimeError("student Top-K was not captured")
            normal_count = int(student.decoder.num_queries)
            replay = SharedQueryReplay(
                initial_query=replay_inputs.initial_query[:, -normal_count:],
                initial_reference_unactivated=replay_inputs.initial_reference_unactivated[:, -normal_count:],
                topk_indices=holder["indices"],
                memory=replay_inputs.memory,
                spatial_shapes=replay_inputs.spatial_shapes,
                normal_query_count=normal_count,
            )
            teacher_features = teacher.backbone(samples)
            teacher_encoded = teacher.encoder([feature.clone() for feature in teacher_features])
            privileged_encoded, masks = privilege_features(
                teacher_encoded,
                absolute_targets,
                samples.shape[-2:],
                background_weight=0.2,
                context_weight=None,
                context_scale=1.5,
            )
            teacher_memory_normal, shapes = teacher.decoder._get_encoder_input(teacher_encoded)
            teacher_memory_privileged, privileged_shapes = teacher.decoder._get_encoder_input(privileged_encoded)
            if tuple(map(tuple, shapes)) != tuple(map(tuple, privileged_shapes)):
                raise RuntimeError("teacher memory shapes differ")
            _, _, normal_trace, normal_calls = _run_direct_teacher(
                teacher, replay, teacher_memory_normal, shapes
            )
            _, _, privileged_trace, privileged_calls = _run_direct_teacher(
                teacher, replay, teacher_memory_privileged, privileged_shapes
            )
            teacher_topk_calls += normal_calls + privileged_calls
            student_layers = [normal_query_slice(item, normal_count) for item in student_trace["query_outputs"]]
            normal_layers = [normal_query_slice(item, normal_count) for item in normal_trace["query_outputs"]]
            privileged_layers = [normal_query_slice(item, normal_count) for item in privileged_trace["query_outputs"]]
            cosine = {
                "normal": _cosines(student_layers, normal_layers),
                "privileged": _cosines(student_layers, privileged_layers),
            }
            for mode, layers in cosine.items():
                for layer_index, values in enumerate(layers):
                    values = values.float()
                    if not torch.isfinite(values).all():
                        raise RuntimeError("non-finite hidden cosine")
                    layer_sums[mode][layer_index]["sum"] += float(values.sum())
                    layer_sums[mode][layer_index]["sumsq"] += float((values * values).sum())
                    layer_sums[mode][layer_index]["count"] += int(values.numel())
            matches = solver.criterion.matcher(student_outputs, matcher_targets)["indices"]
            probabilities = student_outputs["pred_logits"].sigmoid()
            for batch_index, target in enumerate(raw_targets):
                image_id = int(target["image_id"].detach().cpu().reshape(-1)[0])
                image_topk[str(image_id)] = holder["indices"][batch_index].detach().cpu().tolist()
                source_indices, target_indices = matches[batch_index]
                for source_index, target_index in zip(source_indices.tolist(), target_indices.tolist()):
                    category_id = int(matcher_targets[batch_index]["labels"][target_index])
                    scores = probabilities[batch_index, :, category_id]
                    gt_score = float(scores[source_index])
                    iou = _iou_pair(
                        student_outputs["pred_boxes"][batch_index, source_index],
                        matcher_targets[batch_index]["boxes"][target_index],
                    )
                    predicted_class = int(probabilities[batch_index, source_index].argmax())
                    row = {
                        "key": f"{image_id}:{target_index}:{category_id}",
                        "image_id": image_id,
                        "target_index": target_index,
                        "class": category_id,
                        "query_slot": source_index,
                        "anchor_index": int(holder["indices"][batch_index, source_index]),
                        "iou": iou,
                        "gt_class_score": gt_score,
                        "predicted_class": predicted_class,
                        "gt_class_rank": int((scores > scores[source_index]).sum()) + 1,
                        "hit": bool(iou >= 0.5 and predicted_class == category_id and gt_score >= 0.5),
                        "cosine_normal": [float(layer[batch_index, source_index]) for layer in cosine["normal"]],
                        "cosine_privileged": [float(layer[batch_index, source_index]) for layer in cosine["privileged"]],
                    }
                    rows.append(row)
            if not all(torch.isfinite(item).all() for item in (
                student_outputs["pred_logits"], student_outputs["pred_boxes"],
                teacher_memory_normal, teacher_memory_privileged,
            )):
                raise RuntimeError("non-finite diagnostic tensor")
            images += len(raw_targets)
    if teacher_topk_calls:
        raise RuntimeError(f"teacher attempted Top-K {teacher_topk_calls} times")
    layer_summary = {}
    for mode, layers in layer_sums.items():
        layer_summary[mode] = []
        for layer_index, stats in enumerate(layers):
            mean = stats["sum"] / stats["count"]
            variance = max(stats["sumsq"] / stats["count"] - mean * mean, 0.0)
            layer_summary[mode].append({
                "layer": layer_index,
                "cosine_mean": mean,
                "cosine_std": variance ** 0.5,
                "kd_mean_one_minus_cosine": 1.0 - mean,
                "query_count": stats["count"],
            })
    result = {
        "group": group,
        "checkpoint": str(checkpoint),
        "images": images,
        "gt_matches": len(rows),
        "class_counts": {
            str(category_id): sum(item["class"] == category_id for item in rows)
            for category_id in range(4)
        },
        "student_forward_seconds": student_seconds,
        "student_ms_per_image": 1000.0 * student_seconds / images,
        "combined_diagnostic_peak_memory_mb": (
            float(torch.cuda.max_memory_allocated(device)) / (1024 ** 2)
            if device.type == "cuda" else None
        ),
        "teacher_topk_calls": teacher_topk_calls,
        "layer_hidden": layer_summary,
        "targets": rows,
        "topk_indices": image_topk,
    }
    del student, solver, cfg
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _topk_overlap(baseline, candidate):
    values = []
    for image_id in sorted(set(baseline) & set(candidate), key=int):
        left, right = set(baseline[image_id]), set(candidate[image_id])
        values.append(len(left & right) / len(left | right))
    return {
        "images": len(values),
        "jaccard_mean": sum(values) / len(values) if values else None,
        "jaccard_min": min(values) if values else None,
        "jaccard_max": max(values) if values else None,
    }


def run(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    ground_truth = json.loads(Path(args.ann_file).read_text(encoding="utf-8"))
    selected = select_stratified_image_ids(
        ground_truth, per_class=args.per_class, empty=args.empty_images, total=args.total_images
    )
    subset = _subset_coco(ground_truth, selected)
    subset_file = output_dir / "validation_query_subset.json"
    subset_file.write_text(json.dumps(subset, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "subset_manifest.json").write_text(json.dumps({
        "rule": "first unique sorted image IDs per class 0..3, then first empty IDs, then sorted fill",
        "per_class_requested": args.per_class,
        "empty_requested": args.empty_images,
        "total_requested": args.total_images,
        "image_ids": selected,
        "gt_count": len(subset["annotations"]),
        "class_counts": {
            str(category_id): sum(int(item["category_id"]) == category_id for item in subset["annotations"])
            for category_id in range(4)
        },
        "empty_images": sum(
            not any(int(item["image_id"]) == image_id for item in subset["annotations"])
            for image_id in selected
        ),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    teacher_args = _model_args(args.configs[0], args.teacher_checkpoint, subset_file, args)
    _, teacher_solver, teacher = _build(teacher_args)
    teacher.eval().requires_grad_(False)
    group_results = {}
    for group, config, checkpoint in zip(GROUPS, args.configs, args.checkpoints):
        group_results[group] = _group_probe(
            group, config, checkpoint, subset_file, teacher, args
        )
        print(json.dumps({
            "group": group,
            "images": group_results[group]["images"],
            "gt_matches": group_results[group]["gt_matches"],
            "student_ms_per_image": group_results[group]["student_ms_per_image"],
        }, sort_keys=True), flush=True)
    comparisons = {}
    for group in GROUPS[1:]:
        comparisons[f"{group}_minus_KQ0"] = {
            "responsibility": compare_responsibility(
                group_results["KQ0"]["targets"], group_results[group]["targets"]
            ),
            "topk_overlap": _topk_overlap(
                group_results["KQ0"]["topk_indices"], group_results[group]["topk_indices"]
            ),
        }
    payload = {
        "scope": "fixed stratified validation subset; best EMA; no test",
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "teacher_memory": {
            "normal": "frozen baseline backbone/HybridEncoder",
            "privileged": "same memory after GT NP mask factor 0.2 before ECTransformer projection/BN",
            "topk_source": "student exact q0/r0/top-k; teacher re-selection forbidden",
        },
        "groups": group_results,
        "comparisons": comparisons,
        "limitation": "Hungarian responsibility is diagnostic; identical decoder slot numbers are not identities when student Top-K anchor sets differ.",
    }
    (output_dir / "query_responsibility.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "COMPLETED").write_text("KQ query responsibility audit complete\n", encoding="utf-8")
    del teacher, teacher_solver


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs=4, required=True)
    parser.add_argument("--checkpoints", nargs=4, required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--img-folder", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--per-class", type=int, default=24)
    parser.add_argument("--empty-images", type=int, default=32)
    parser.add_argument("--total-images", type=int, default=160)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        output_dir = Path(args.out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "FAILED.json").write_text(json.dumps({
            "error_type": type(error).__name__, "error": str(error)
        }, indent=2) + "\n", encoding="utf-8")
        raise
