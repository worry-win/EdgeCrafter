"""Validation-only quality/behavior audit for shared-q0/r0 KQ teachers."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ecdetseg"))
sys.path.insert(0, str(ROOT))

from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy  # noqa: E402
from engine.solver.ec_engine import summarize_yolo_pr_curve_metrics  # noqa: E402
from scripts.ablation.cmp5L_kq_teacher_diagnostic import fixed_student_query_groups  # noqa: E402
from scripts.ablation.cmp5L_query_behavior_kd import (  # noqa: E402
    DecoderReplayInputs,
    _predictions_from_decoder_outputs,
    _run_decoder_with_capture,
    run_model_with_capture,
)
from scripts.ablation.cmp5L_shared_query_kd import SharedQueryReplay, normal_query_slice  # noqa: E402
from scripts.ablation.diag_cmp5L_np_whowhat import AttentionCapture  # noqa: E402
from scripts.ablation.probe_cmp5L_decoder_internal_tensors import _build, _normalise_targets  # noqa: E402
from scripts.ablation.probe_cmp5L_privileged_ema_oracle import privilege_features  # noqa: E402
from scripts.ablation.probe_cmp5L_shared_query_kd_stage1 import (  # noqa: E402
    _capture_topk_indices,
    _restore_topk,
    _run_direct_teacher,
)
from scripts.ablation.train_cmp5L_shared_query_kd import _load_frozen_teacher  # noqa: E402


ARMS = ("student", "normal_teacher", "privileged_teacher")
QUERY_GROUPS = ("all", "tp", "high_score_fp", "ordinary_unmatched", "hungarian_gt", "duct")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_args(args):
    return SimpleNamespace(
        config=args.config,
        checkpoint=args.student_checkpoint,
        ann_file=args.ann_file,
        img_folder=args.img_folder,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        weights="ema",
        device=torch.device(args.device),
    )


def _absolute_targets(raw_targets, device):
    return [
        {"boxes": item["boxes"].as_subclass(torch.Tensor).to(device), "labels": item["labels"].to(device)}
        for item in raw_targets
    ]


def _prediction_record(image_id, result):
    return {
        "image_id": int(image_id),
        "boxes_xyxy": result["boxes"].detach().cpu().tolist(),
        "scores": result["scores"].detach().cpu().tolist(),
        "labels": result["labels"].detach().cpu().tolist(),
    }


def _margin(logits):
    values = logits.topk(2, dim=-1).values
    return values[..., 0] - values[..., 1]


def _matched_iou(boxes, fixed_gt, targets):
    result = boxes.new_zeros(boxes.shape[:2])
    for batch_index, target in enumerate(targets):
        valid = fixed_gt[batch_index] >= 0
        if not bool(valid.any()) or not len(target["boxes"]):
            continue
        pred = box_cxcywh_to_xyxy(boxes[batch_index, valid])
        gt = box_cxcywh_to_xyxy(target["boxes"].as_subclass(torch.Tensor).to(boxes.device))
        indices = fixed_gt[batch_index, valid]
        left = pred
        right = gt[indices]
        lt = torch.maximum(left[:, :2], right[:, :2])
        rb = torch.minimum(left[:, 2:], right[:, 2:])
        inter = (rb - lt).clamp(min=0).prod(-1)
        union = ((left[:, 2:] - left[:, :2]).clamp(min=0).prod(-1)
                 + (right[:, 2:] - right[:, :2]).clamp(min=0).prod(-1) - inter)
        result[batch_index, valid] = inter / union.clamp(min=1e-9)
    return result


def _new_accumulator():
    return {group: {arm: defaultdict(float) for arm in ARMS} for group in QUERY_GROUPS}


def _add(accumulator, group, arm, mask, values):
    count = int(mask.sum())
    bucket = accumulator[group][arm]
    bucket["count"] += count
    if not count:
        return
    for name, tensor in values.items():
        selected = tensor[mask].detach().float()
        bucket[f"{name}_sum"] += float(selected.sum())
        bucket[f"{name}_sumsq"] += float((selected * selected).sum())


def _finish_accumulator(accumulator):
    result = {}
    for group, arms in accumulator.items():
        result[group] = {}
        for arm, raw in arms.items():
            count = int(raw.get("count", 0))
            row = {"count": count}
            for key, value in raw.items():
                if not key.endswith("_sum"):
                    continue
                name = key[:-4]
                mean = value / count if count else None
                sumsq = raw.get(f"{name}_sumsq", 0.0)
                row[f"{name}_mean"] = mean
                row[f"{name}_std"] = max(sumsq / count - mean * mean, 0.0) ** 0.5 if count else None
            result[group][arm] = row
    return result


def _summarize_evaluator(evaluator):
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    coco = evaluator.coco_eval["bbox"]
    stats = coco.stats.tolist()
    yolo = summarize_yolo_pr_curve_metrics(coco, evaluator.coco_gt) or {}
    overall = yolo.get("yolo_f1_iou50", {})
    precision = coco.eval["precision"]
    iou50 = int(np.argmin(np.abs(coco.params.iouThrs - 0.50)))
    classes = {}
    for index, category in enumerate(evaluator.coco_gt.loadCats(coco.params.catIds)):
        curve = precision[iou50, :, index, 0, -1]
        valid = curve > -1
        classes[str(category["id"])] = {
            "name": category.get("name", str(category["id"])),
            "ap50": float(curve[valid].mean()) if valid.any() else -1.0,
        }
    return {
        "ap": float(stats[0]), "ap50": float(stats[1]), "ap75": float(stats[2]),
        "ar100": float(stats[8]),
        "precision": float(overall.get("precision", 0.0)),
        "recall": float(overall.get("recall", 0.0)),
        "f1": float(overall.get("f1", 0.0)),
        "per_class_ap50": classes,
    }


def _direct_with_cross(teacher, replay, memory, shapes):
    with AttentionCapture(teacher.decoder.decoder) as capture:
        direct, outputs, trace, calls = _run_direct_teacher(teacher, replay, memory, shapes)
        cross = [item["cross_attn_output"] for item in capture.layers]
    return direct, outputs, trace, calls, cross


def _historical_np_parity(student, teacher, samples, absolute_targets, replay):
    teacher_base = teacher.backbone(samples)
    encoded = teacher.encoder([item.clone() for item in teacher_base])
    privileged, _ = privilege_features(encoded, absolute_targets, samples.shape[-2:], 0.2, None, 1.5)
    memory_n, shapes_n = teacher.decoder._get_encoder_input(encoded)
    memory_p, shapes_p = teacher.decoder._get_encoder_input(privileged)
    content_n, ref_n, _, _ = teacher.decoder._get_decoder_input(memory_n, shapes_n)
    historical = SharedQueryReplay(content_n, ref_n, replay.topk_indices, memory_p, shapes_p,
                                   normal_query_count=teacher.decoder.num_queries)
    _, hist_out, hist_trace, hist_calls, _ = _direct_with_cross(teacher, historical, memory_p, shapes_p)
    _, kq_out, kq_trace, kq_calls, _ = _direct_with_cross(teacher, replay, memory_p, shapes_p)
    fields = {
        "q0": float((content_n - replay.initial_query).abs().max()),
        "r0": float((ref_n - replay.initial_reference_unactivated).abs()[torch.isfinite(ref_n - replay.initial_reference_unactivated)].max()),
        "logits": float((hist_out[1] - kq_out[1]).abs().max()),
        "boxes": float((hist_out[0] - kq_out[0]).abs().max()),
    }
    fields["hidden_by_layer"] = [
        float((left - right).abs().max()) for left, right in zip(hist_trace["query_outputs"], kq_trace["query_outputs"])
    ]
    ordered = ["q0", "r0"] + [f"hidden_L{i}" for i in range(4)] + ["logits", "boxes"]
    values = {"q0": fields["q0"], "r0": fields["r0"], "logits": fields["logits"], "boxes": fields["boxes"],
              **{f"hidden_L{i}": value for i, value in enumerate(fields["hidden_by_layer"])}}
    fields["first_divergence_gt_2e-6"] = next((name for name in ordered if values[name] > 2e-6), None)
    fields["teacher_topk_calls"] = hist_calls + kq_calls
    return fields


def evaluate(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    cfg, solver, student = _build(_model_args(args))
    student.eval().requires_grad_(False)
    teacher = _load_frozen_teacher(student, args.teacher_checkpoint, device)
    evaluators = {arm: copy.deepcopy(cfg.evaluator) for arm in ARMS}
    for evaluator in evaluators.values():
        evaluator.cleanup()
    streams = {arm: (output_dir / f"{arm}.predictions.jsonl").open("x", encoding="utf-8") for arm in ARMS}
    accumulator = _new_accumulator()
    responsibility = defaultdict(int)
    per_image = (output_dir / "query_behavior_by_image.jsonl").open("x", encoding="utf-8")
    parity = None
    seen = 0
    started = time.time()
    try:
        with torch.no_grad():
            for batch_index, (samples, raw_targets) in enumerate(cfg.val_dataloader):
                if args.max_images and seen >= args.max_images:
                    break
                samples = samples.to(device)
                matcher_targets = _normalise_targets(raw_targets, samples.shape[-2], samples.shape[-1], device)
                absolute_targets = _absolute_targets(raw_targets, device)
                original, holder = _capture_topk_indices(student.decoder)
                with AttentionCapture(student.decoder.decoder) as student_capture:
                    try:
                        student_outputs, student_trace, replay_inputs = run_model_with_capture(student, samples, None)
                    finally:
                        _restore_topk(student.decoder, original)
                    student_cross = [item["cross_attn_output"] for item in student_capture.layers]
                replay = SharedQueryReplay(
                    replay_inputs.initial_query[:, -student.decoder.num_queries:],
                    replay_inputs.initial_reference_unactivated[:, -student.decoder.num_queries:],
                    holder["indices"], replay_inputs.memory, replay_inputs.spatial_shapes,
                    normal_query_count=student.decoder.num_queries,
                )
                teacher_base = teacher.backbone(samples)
                encoded = teacher.encoder([item.clone() for item in teacher_base])
                privileged, _ = privilege_features(encoded, absolute_targets, samples.shape[-2:], 0.2, None, 1.5)
                normal_memory, normal_shapes = teacher.decoder._get_encoder_input(encoded)
                privileged_memory, privileged_shapes = teacher.decoder._get_encoder_input(privileged)
                _, normal_tuple, normal_trace, normal_calls, normal_cross = _direct_with_cross(
                    teacher, replay, normal_memory, normal_shapes
                )
                _, privileged_tuple, privileged_trace, privileged_calls, privileged_cross = _direct_with_cross(
                    teacher, replay, privileged_memory, privileged_shapes
                )
                if normal_calls or privileged_calls:
                    raise RuntimeError("teacher attempted Top-K")
                arms = {
                    "student": student_outputs,
                    "normal_teacher": _predictions_from_decoder_outputs(normal_tuple, student.decoder.num_queries),
                    "privileged_teacher": _predictions_from_decoder_outputs(privileged_tuple, student.decoder.num_queries),
                }
                hidden = {
                    "student": [normal_query_slice(item, student.decoder.num_queries) for item in student_trace["query_outputs"]],
                    "normal_teacher": [normal_query_slice(item, student.decoder.num_queries) for item in normal_trace["query_outputs"]],
                    "privileged_teacher": [normal_query_slice(item, student.decoder.num_queries) for item in privileged_trace["query_outputs"]],
                }
                cross = {"student": student_cross, "normal_teacher": normal_cross, "privileged_teacher": privileged_cross}
                if parity is None and args.state == "baseline":
                    parity = _historical_np_parity(student, teacher, samples, absolute_targets, replay)
                    if args.require_parity and parity["first_divergence_gt_2e-6"] is not None:
                        raise RuntimeError(f"historical NP/KQ parity failed: {parity}")
                groups = fixed_student_query_groups(
                    arms["student"]["pred_logits"], arms["student"]["pred_boxes"], matcher_targets
                )
                student_matches = solver.criterion.matcher(arms["student"], matcher_targets)["indices"]
                masks = {name: groups[name] for name in ("all", "tp", "high_score_fp", "ordinary_unmatched")}
                masks["hungarian_gt"] = torch.zeros_like(groups["all"])
                masks["duct"] = torch.zeros_like(groups["all"])
                for image_index, (sources, target_indices) in enumerate(student_matches):
                    sources_on_device = sources.to(masks["hungarian_gt"].device)
                    targets_on_device = target_indices.to(
                        matcher_targets[image_index]["labels"].device
                    )
                    masks["hungarian_gt"][image_index, sources_on_device] = True
                    labels = matcher_targets[image_index]["labels"][targets_on_device]
                    masks["duct"][
                        image_index, sources_on_device[labels == 3]
                    ] = True
                arm_matches = {arm: solver.criterion.matcher(value, matcher_targets)["indices"] for arm, value in arms.items()}
                for arm in ("normal_teacher", "privileged_teacher"):
                    for image_index in range(len(raw_targets)):
                        student_map = {int(q): int(g) for q, g in zip(*[x.tolist() for x in student_matches[image_index]])}
                        teacher_map = {int(q): int(g) for q, g in zip(*[x.tolist() for x in arm_matches[arm][image_index]])}
                        for query in range(student.decoder.num_queries):
                            left, right = student_map.get(query), teacher_map.get(query)
                            if left is not None and right is not None:
                                responsibility[f"{arm}:same_gt" if left == right else f"{arm}:different_gt"] += 1
                            elif left is not None:
                                responsibility[f"{arm}:student_gt_teacher_unassigned"] += 1
                            elif right is not None:
                                responsibility[f"{arm}:student_unassigned_teacher_gt"] += 1
                            else:
                                responsibility[f"{arm}:both_unassigned"] += 1
                for arm, outputs in arms.items():
                    scores = outputs["pred_logits"].sigmoid().max(-1).values
                    margins = _margin(outputs["pred_logits"])
                    ious = _matched_iou(outputs["pred_boxes"], groups["matched_gt"], matcher_targets)
                    for group, mask in masks.items():
                        values = {"score": scores, "margin": margins, "iou_fixed_gt": ious}
                        for layer_index in range(4):
                            values[f"hidden_norm_L{layer_index}"] = hidden[arm][layer_index].float().norm(dim=-1)
                            values[f"cross_norm_L{layer_index}"] = cross[arm][layer_index].float().norm(dim=-1)
                            values[f"cosine_to_student_L{layer_index}"] = F.cosine_similarity(
                                hidden["student"][layer_index].float(), hidden[arm][layer_index].float(), dim=-1
                            )
                        _add(accumulator, group, arm, mask, values)
                original_sizes = torch.stack([item["orig_size"] for item in raw_targets]).to(device)
                results = {arm: solver.postprocessor(outputs, original_sizes) for arm, outputs in arms.items()}
                for arm in ARMS:
                    predictions = {
                        int(target["image_id"].item()): result for target, result in zip(raw_targets, results[arm])
                    }
                    evaluators[arm].update(predictions)
                    for image_id, result in predictions.items():
                        streams[arm].write(json.dumps(_prediction_record(image_id, result)) + "\n")
                for image_index, target in enumerate(raw_targets):
                    per_image.write(json.dumps({
                        "image_id": int(target["image_id"].item()),
                        "gt_count": int(len(matcher_targets[image_index]["labels"])),
                        "group_counts": {name: int(mask[image_index].sum()) for name, mask in masks.items()},
                    }) + "\n")
                seen += len(raw_targets)
                if seen % 200 < len(raw_targets):
                    print(json.dumps({"state": args.state, "seen": seen, "seconds": time.time() - started}), flush=True)
    finally:
        for stream in streams.values():
            stream.close()
        per_image.close()
    if not args.max_images and seen != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} images, got {seen}")
    metrics = {arm: _summarize_evaluator(evaluator) for arm, evaluator in evaluators.items()}
    payload = {
        "scope": "validation only; shared student q0/r0; frozen baseline teachers; no test",
        "state": args.state,
        "images": seen,
        "paths": {
            "config": args.config, "student_checkpoint": args.student_checkpoint,
            "teacher_checkpoint": args.teacher_checkpoint,
        },
        "hashes": {
            "student_checkpoint_sha256": _sha256(args.student_checkpoint),
            "teacher_checkpoint_sha256": _sha256(args.teacher_checkpoint),
            "script_sha256": _sha256(__file__),
        },
        "historical_np_kq_parity": parity,
        "metrics": metrics,
        "query_behavior": _finish_accumulator(accumulator),
        "same_forward_responsibility": dict(responsibility),
        "predictions": {arm: str(output_dir / f"{arm}.predictions.jsonl") for arm in ARMS},
        "seconds": time.time() - started,
    }
    (output_dir / "result.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    (output_dir / "COMPLETED").write_text("KQ teacher-quality diagnostic complete\n")
    print(json.dumps({"state": args.state, "images": seen, "metrics": metrics}, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", choices=("baseline", "KQ0", "KQ2"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--img-folder", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--expected-images", type=int, default=2975)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--require-parity", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
