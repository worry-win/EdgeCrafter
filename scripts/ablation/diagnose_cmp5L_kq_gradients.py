"""No-step training-mode detection-vs-KQ gradient diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ecdetseg"))
sys.path.insert(0, str(ROOT))

from scripts.ablation.cmp5L_kq_teacher_diagnostic import (  # noqa: E402
    fixed_student_query_groups,
    gradient_pair_stats,
    grouped_loss_accounting,
    restore_trainable_parameter_mask,
)
from scripts.ablation.cmp5L_shared_query_kd import normal_query_slice, resolve_kq_spec  # noqa: E402
from scripts.ablation.probe_cmp5L_shared_query_kd_stage1 import _build_train  # noqa: E402
from scripts.ablation.train_cmp5L_shared_query_kd import (  # noqa: E402
    _load_frozen_teacher,
    _module,
    _move_targets,
    _student_forward,
    _targets_absolute_xyxy,
    _teacher_forward,
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(named):
    return {name: value.detach().cpu().clone() for name, value in named}


def _delta(before, named):
    maximum = 0.0
    for name, value in named:
        reference = before[name]
        current = value.detach().cpu()
        if current.dtype == torch.bool:
            difference = float(torch.logical_xor(current, reference).any())
        elif current.numel():
            difference = float((current - reference).abs().max())
        else:
            difference = 0.0
        maximum = max(maximum, difference)
    return maximum


def _parameter_buckets(module):
    buckets = defaultdict(list)
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        buckets["all"].append((name, parameter))
        if name.startswith("backbone."):
            buckets["backbone"].append((name, parameter))
        elif name.startswith("encoder."):
            buckets["HybridEncoder"].append((name, parameter))
        elif name.startswith("decoder.decoder.layers."):
            index = name.split(".")[3]
            buckets[f"decoder_L{index}"].append((name, parameter))
        elif name.startswith("decoder."):
            buckets["decoder_other"].append((name, parameter))
    return buckets


def _gradient_map(loss, named_parameters):
    if not loss.requires_grad:
        return {name: None for name, _ in named_parameters}
    gradients = torch.autograd.grad(
        loss, [parameter for _, parameter in named_parameters],
        retain_graph=True, allow_unused=True, create_graph=False,
    )
    return {name: gradient for (name, _), gradient in zip(named_parameters, gradients)}


def _bucket_pair(loss_det, loss_kd, buckets):
    named_all = buckets["all"]
    det_map = _gradient_map(loss_det, named_all)
    kd_map = _gradient_map(loss_kd, named_all)
    result = {}
    for name, named in buckets.items():
        det = [det_map[item_name] for item_name, _ in named]
        kd = [kd_map[item_name] for item_name, _ in named]
        result[name] = gradient_pair_stats(det, kd)
        result[name]["parameter_tensors"] = len(named)
        result[name]["det_missing_tensors"] = sum(item is None for item in det)
        result[name]["kd_missing_tensors"] = sum(item is None for item in kd)
    return result


def _single_loss_norm(loss, buckets):
    named_all = buckets["all"]
    gradient_map = _gradient_map(loss, named_all)
    result = {}
    for name, named in buckets.items():
        gradients = [gradient_map[item_name] for item_name, _ in named]
        flat = [item.detach().float().reshape(-1) for item in gradients if item is not None]
        if not flat:
            result[name] = {"status": "missing", "norm": None}
            continue
        vector = torch.cat(flat)
        norm = float(vector.norm())
        result[name] = {
            "status": "nonfinite" if not math.isfinite(norm) else ("zero" if norm == 0 else "nonzero"),
            "norm": norm,
        }
    return result


def _batch_coverage(raw_targets):
    class_counts = defaultdict(int)
    ids = []
    empty = 0
    for target in raw_targets:
        image_id = target.get("image_id")
        ids.append([int(item) for item in image_id.detach().cpu().reshape(-1)] if torch.is_tensor(image_id) else image_id)
        labels = target["labels"].detach().cpu().tolist()
        empty += int(len(labels) == 0)
        for label in labels:
            class_counts[str(int(label))] += 1
    return {"image_ids": ids, "class_counts": dict(class_counts), "empty_images": empty}


def _diagnose_loader(args, state, ann_file, batch_limit, stream_name):
    build_args = SimpleNamespace(
        config=state["config"], checkpoint=state["checkpoint"], ann_file=ann_file,
        img_folder=args.img_folder, num_workers=0, batch_size=args.batch_size,
        weights="ema", device=torch.device(args.device), seed=args.seed,
    )
    cfg, solver, student = _build_train(build_args)
    trainable_mask_audit = restore_trainable_parameter_mask(student, _module(solver.model))
    if trainable_mask_audit["trainable_parameter_tensors"] == 0:
        raise RuntimeError("EMA diagnostic copy has no trainable parameters")
    student.train()
    if getattr(student, "_freeze_backbone", False):
        student.backbone.eval()
    solver.criterion.train()
    teacher = _load_frozen_teacher(student, args.teacher_checkpoint, torch.device(args.device))
    teacher.eval().requires_grad_(False)
    runtime = SimpleNamespace(teacher=teacher, spec=resolve_kq_spec("KQ2"))
    teacher_params_before = _snapshot(teacher.named_parameters())
    teacher_buffers_before = _snapshot(teacher.named_buffers())
    student_params_before = _snapshot(student.named_parameters())
    student_buffers_before = _snapshot(student.named_buffers())
    buckets = _parameter_buckets(student)
    rows = []
    if hasattr(cfg.train_dataloader, "set_epoch"):
        cfg.train_dataloader.set_epoch(0)
    for batch_index, (samples, raw_targets) in enumerate(cfg.train_dataloader):
        if batch_index >= batch_limit:
            break
        samples = samples.to(args.device)
        targets = _move_targets(raw_targets, torch.device(args.device))
        absolute = _targets_absolute_xyxy(raw_targets, samples.shape[-2:], torch.device(args.device))
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
            outputs, student_trace, replay, topk = _student_forward(student, samples, targets)
            teacher_layers, teacher_audit = _teacher_forward(
                runtime, replay, topk, samples, absolute, True
            )
        student_layers = [normal_query_slice(item, 300).float() for item in student_trace["query_outputs"]]
        teacher_layers = [item.float() for item in teacher_layers]
        with torch.autocast(device_type="cuda", enabled=False):
            loss_dict = solver.criterion(
                outputs, targets, epoch=0, step=batch_index, global_step=batch_index,
                epoch_step=len(cfg.train_dataloader),
            )
            detection = sum(loss_dict.values())
            per_layer = [1.0 - F.cosine_similarity(s, t.detach(), dim=-1) for s, t in zip(student_layers, teacher_layers)]
            raw_layer_losses = [value.mean() for value in per_layer]
            weighted_layer_terms = [0.5 * value / 4.0 for value in raw_layer_losses]
            weighted_kd = sum(weighted_layer_terms)
        fixed_targets = [
            {"boxes": item["boxes"].as_subclass(torch.Tensor).to(args.device), "labels": item["labels"].to(args.device)}
            for item in targets
        ]
        groups = fixed_student_query_groups(outputs["pred_logits"], outputs["pred_boxes"], fixed_targets)
        group_rows = {}
        for group in ("tp", "high_score_fp", "ordinary_unmatched"):
            mask = groups[group]
            combined_per_query = torch.stack(per_layer).mean(0) * 0.5
            accounting = grouped_loss_accounting(combined_per_query, mask)
            contribution = (combined_per_query * mask).sum() / combined_per_query.numel()
            group_rows[group] = {
                "accounting": accounting,
                "gradient": _single_loss_norm(contribution, buckets),
            }
        row = {
            "stream": stream_name,
            "batch": batch_index,
            "coverage": _batch_coverage(raw_targets),
            "loss": {
                "detection": float(detection.detach()),
                "kd_raw_layers": [float(value.detach()) for value in raw_layer_losses],
                "kd_weighted_layers": [float(value.detach()) for value in weighted_layer_terms],
                "kd_weighted_total": float(weighted_kd.detach()),
            },
            "gradient_pair": _bucket_pair(detection, weighted_kd, buckets),
            "per_layer_kd_gradient": [
                _single_loss_norm(term, buckets) for term in weighted_layer_terms
            ],
            "query_groups": group_rows,
            "teacher_alignment": teacher_audit["alignment"],
            "teacher_topk_calls": teacher_audit["topk_calls"],
            "amp": {
                "autocast": True,
                "gradient_scale": 1.0,
                "note": "autograd.grad is applied to unscaled FP32 losses; no GradScaler scale enters reported gradients",
            },
        }
        rows.append(row)
        del detection, weighted_kd, outputs, student_trace, replay, teacher_layers, per_layer
    result = {
        "stream": stream_name,
        "ann_file": ann_file,
        "batches": rows,
        "teacher_parameter_max_delta": _delta(teacher_params_before, teacher.named_parameters()),
        "teacher_buffer_max_delta": _delta(teacher_buffers_before, teacher.named_buffers()),
        "student_parameter_max_delta": _delta(student_params_before, student.named_parameters()),
        "student_buffer_max_delta_in_disposable_copy": _delta(student_buffers_before, student.named_buffers()),
        "optimizer_steps": 0,
        "trainable_parameter_mask": trainable_mask_audit,
        "checkpoint_hash_after": _sha256(state["checkpoint"]),
    }
    return result


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    states = {
        "baseline": {"config": args.baseline_config, "checkpoint": args.baseline_checkpoint},
        "KQ2": {"config": args.kq2_config, "checkpoint": args.kq2_checkpoint},
    }
    payload = {
        "scope": "real train transforms/DN; gradient-only; zero optimizer steps; no test",
        "seed": args.seed,
        "states": {},
        "paths": vars(args),
        "checkpoint_hash_before": {name: _sha256(item["checkpoint"]) for name, item in states.items()},
    }
    for name, state in states.items():
        payload["states"][name] = {
            "mixed": _diagnose_loader(args, state, args.mixed_ann_file, args.mixed_batches, "prelocked_mixed"),
            "empty": _diagnose_loader(args, state, args.empty_ann_file, args.empty_batches, "prelocked_empty"),
        }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")
    output.with_suffix(".COMPLETED").write_text("KQ no-step gradient diagnostic complete\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--kq2-config", required=True)
    parser.add_argument("--kq2-checkpoint", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--mixed-ann-file", required=True)
    parser.add_argument("--empty-ann-file", required=True)
    parser.add_argument("--img-folder", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--mixed-batches", type=int, default=3)
    parser.add_argument("--empty-batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260922)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
