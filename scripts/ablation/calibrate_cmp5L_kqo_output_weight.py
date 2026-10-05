"""Lock KQ-O output-KD weight from no-step decoder-L3 gradients."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ecdetseg"))
sys.path.insert(0, str(ROOT))

from scripts.ablation.cmp5L_kq_teacher_diagnostic import restore_trainable_parameter_mask  # noqa: E402
from scripts.ablation.cmp5L_shared_query_kd import resolve_kq_spec, sigmoid_output_kd  # noqa: E402
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


def _snapshot(module):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _maximum_delta(before, module):
    maximum = 0.0
    for name, value in module.state_dict().items():
        reference = before[name]
        current = value.detach().cpu()
        if current.dtype == torch.bool:
            delta = float(torch.logical_xor(reference, current).any())
        elif current.numel():
            delta = float((reference - current).abs().max())
        else:
            delta = 0.0
        maximum = max(maximum, delta)
    return maximum


def _gradient_norm(loss, parameters):
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True, create_graph=False
    )
    terms = [value.detach().float().square().sum() for value in gradients if value is not None]
    return float(torch.stack(terms).sum().sqrt()) if terms else 0.0


def _coverage(raw_targets):
    counts = {str(index): 0 for index in range(4)}
    image_ids = []
    empty = 0
    for target in raw_targets:
        image_id = target.get("image_id")
        if torch.is_tensor(image_id):
            image_ids.append([int(value) for value in image_id.detach().cpu().reshape(-1)])
        else:
            image_ids.append(image_id)
        labels = [int(value) for value in target["labels"].detach().cpu().reshape(-1)]
        empty += int(not labels)
        for label in labels:
            counts[str(label)] += 1
    return {"image_ids": image_ids, "class_counts": counts, "empty_images": empty}


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    build_args = SimpleNamespace(
        config=args.config,
        checkpoint=args.checkpoint,
        ann_file=args.ann_file,
        img_folder=args.img_folder,
        num_workers=0,
        batch_size=args.batch_size,
        weights="ema",
        device=device,
        seed=args.seed,
    )
    cfg, solver, student = _build_train(build_args)
    trainable = restore_trainable_parameter_mask(student, _module(solver.model))
    student.train()
    if getattr(student, "_freeze_backbone", False):
        student.backbone.eval()
    solver.criterion.train()
    teacher = _load_frozen_teacher(student, args.teacher_checkpoint, device)
    teacher.eval().requires_grad_(False)
    runtime = SimpleNamespace(teacher=teacher, spec=resolve_kq_spec("KQ2"))
    teacher_before = _snapshot(teacher)
    checkpoint_hash_before = _sha256(args.checkpoint)
    decoder_l3 = [
        parameter for name, parameter in student.named_parameters()
        if parameter.requires_grad and name.startswith("decoder.decoder.layers.3.")
    ]
    if not decoder_l3:
        raise RuntimeError("decoder L3 parameter bucket is empty")
    if hasattr(cfg.train_dataloader, "set_epoch"):
        cfg.train_dataloader.set_epoch(0)
    rows = []
    for batch_index, (samples, raw_targets) in enumerate(cfg.train_dataloader):
        if batch_index >= args.batches:
            break
        samples = samples.to(device)
        targets = _move_targets(raw_targets, device)
        absolute = _targets_absolute_xyxy(raw_targets, samples.shape[-2:], device)
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
            outputs, _trace, replay, topk = _student_forward(student, samples, targets)
            _teacher_layers, teacher_audit = _teacher_forward(
                runtime, replay, topk, samples, absolute, True
            )
        with torch.autocast(device_type="cuda", enabled=False):
            loss_dict = solver.criterion(
                outputs, targets, epoch=0, step=batch_index,
                global_step=batch_index, epoch_step=len(cfg.train_dataloader),
            )
            detection = sum(loss_dict.values())
            teacher_final = teacher_audit["teacher_logits"][-1].float()
            output_info = sigmoid_output_kd(
                outputs["pred_logits"].float(), teacher_final,
                normal_query_count=300, detach_teacher=True,
            )
        det_norm = _gradient_norm(detection, decoder_l3)
        output_norm = _gradient_norm(output_info["loss"], decoder_l3)
        if not (math.isfinite(det_norm) and math.isfinite(output_norm)):
            raise FloatingPointError("non-finite calibration gradient")
        if det_norm <= 0 or output_norm <= 0:
            raise RuntimeError(
                f"zero decoder-L3 gradient in batch {batch_index}: det={det_norm}, output={output_norm}"
            )
        rows.append({
            "batch": batch_index,
            "coverage": _coverage(raw_targets),
            "loss_detection": float(detection.detach()),
            "loss_output_raw": float(output_info["loss"].detach()),
            "decoder_l3_gradient_detection": det_norm,
            "decoder_l3_gradient_output_raw": output_norm,
            "raw_ratio": output_norm / det_norm,
            "teacher_topk_calls": teacher_audit["topk_calls"],
            "alignment": teacher_audit["alignment"],
        })
        del detection, output_info, outputs, replay, teacher_audit
    if len(rows) != args.batches:
        raise RuntimeError(f"expected {args.batches} batches, observed {len(rows)}")
    raw_ratios = [row["raw_ratio"] for row in rows]
    median_raw_ratio = statistics.median(raw_ratios)
    coefficient = args.target_ratio / median_raw_ratio
    for row in rows:
        row["weighted_ratio"] = row["raw_ratio"] * coefficient
    payload = {
        "scope": "baseline EMA; real train transforms/DN; no optimizer step; no validation/test",
        "selection_rule": "output_weight = target_ratio / median(per-batch raw decoder-L3 gradient ratio)",
        "target_ratio": args.target_ratio,
        "median_raw_ratio": median_raw_ratio,
        "locked_output_weight": coefficient,
        "weighted_ratio_median": statistics.median(row["weighted_ratio"] for row in rows),
        "weighted_ratio_mean": statistics.mean(row["weighted_ratio"] for row in rows),
        "batches": rows,
        "optimizer_steps": 0,
        "teacher_state_max_delta": _maximum_delta(teacher_before, teacher),
        "teacher_gradient_count": sum(parameter.grad is not None for parameter in teacher.parameters()),
        "student_trainable_parameter_mask": trainable,
        "decoder_l3_parameter_tensors": len(decoder_l3),
        "amp": "autocast forward; autograd.grad on unscaled FP32 losses",
        "checkpoint_sha256_before": checkpoint_hash_before,
        "checkpoint_sha256_after": _sha256(args.checkpoint),
        "paths": vars(args),
    }
    if payload["teacher_state_max_delta"] != 0 or payload["teacher_gradient_count"] != 0:
        raise RuntimeError("frozen teacher changed during calibration")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")
    output.with_suffix(".COMPLETED").write_text("KQ-O output weight calibration complete\n")
    print(json.dumps({
        "locked_output_weight": coefficient,
        "median_raw_ratio": median_raw_ratio,
        "weighted_ratio_median": payload["weighted_ratio_median"],
    }, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--img-folder", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--target-ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
