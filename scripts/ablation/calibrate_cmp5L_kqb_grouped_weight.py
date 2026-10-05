"""No-step, prelocked train-batch calibration for KQ-B grouped output KD."""

from __future__ import annotations

import argparse
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

from scripts.ablation.calibrate_cmp5L_kqo_output_weight import (  # noqa: E402
    _coverage, _gradient_norm, _maximum_delta, _sha256, _snapshot,
)
from scripts.ablation.cmp5L_kq_teacher_diagnostic import restore_trainable_parameter_mask  # noqa: E402
from scripts.ablation.cmp5L_shared_query_kd import grouped_sigmoid_output_kd, resolve_kq_spec  # noqa: E402
from scripts.ablation.probe_cmp5L_shared_query_kd_stage1 import _build_train  # noqa: E402
from scripts.ablation.train_cmp5L_shared_query_kd import (  # noqa: E402
    _load_frozen_teacher, _module, _move_targets, _student_forward,
    _targets_absolute_xyxy, _teacher_forward,
)


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if manifest["seed"] != args.seed or manifest["batch_size"] != 2:
        raise ValueError("prelocked manifest seed/batch size differs")
    build_args = SimpleNamespace(
        config=args.config, checkpoint=args.checkpoint,
        ann_file=args.ann_file, img_folder=args.img_folder,
        num_workers=0, batch_size=2, weights="ema", device=device, seed=args.seed,
    )
    cfg, solver, student = _build_train(build_args)
    recipe = cfg.yaml_cfg["train_dataloader"]
    transforms = recipe["dataset"]["transforms"]
    if (transforms["mosaic_epoch"], recipe["collate_fn"]["mixup_epoch"], transforms["stop_epoch"]) != (9, 9, 13):
        raise RuntimeError("calibration recipe is not the locked KQ-B 9/9/13 schedule")
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
        raise RuntimeError("decoder L3 trainable parameter bucket is empty")

    loader = cfg.train_dataloader
    image_index = {int(image_id): index for index, image_id in enumerate(loader.dataset.ids)}
    if len(image_index) != len(loader.dataset.ids):
        raise RuntimeError("train dataset image IDs are not unique")
    rows = []
    matched_class_counts = {str(index): 0 for index in range(4)}
    valid_ratios = []
    empty_check_count = 0
    for batch_index, record in enumerate(manifest["batches"]):
        image_ids = [int(value) for value in record["image_ids"]]
        if len(image_ids) != 2 or any(image_id not in image_index for image_id in image_ids):
            raise ValueError(f"invalid locked image IDs in {record['name']}")
        epoch = int(record["epoch"])
        loader.set_epoch(epoch)
        samples, raw_targets = loader.collate_fn([
            loader.dataset[image_index[image_id]] for image_id in image_ids
        ])
        samples = samples.to(device)
        targets = _move_targets(raw_targets, device)
        absolute = _targets_absolute_xyxy(raw_targets, samples.shape[-2:], device)
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
            outputs, _trace, replay, topk = _student_forward(student, samples, targets)
            _teacher_layers, teacher_audit = _teacher_forward(
                runtime, replay, topk, samples, absolute, True,
            )
        with torch.autocast(device_type="cuda", enabled=False):
            loss_dict = solver.criterion(
                outputs, targets, epoch=epoch, step=batch_index,
                global_step=epoch * len(loader) + batch_index,
                epoch_step=len(loader),
            )
            detection = sum(loss_dict.values())
            with torch.no_grad():
                matches = solver.criterion.matcher({
                    "pred_logits": outputs["pred_logits"].detach(),
                    "pred_boxes": outputs["pred_boxes"].detach(),
                }, targets)["indices"]
            grouped = grouped_sigmoid_output_kd(
                outputs["pred_logits"].float(),
                teacher_audit["teacher_logits"][-1].float(),
                outputs["pred_boxes"], targets, matches,
                normal_query_count=300, negative_topk=20, negative_max_iou=0.3,
            )
        row = {
            "batch": batch_index, "name": record["name"], "purpose": record["purpose"],
            "epoch": epoch, "locked_image_ids": image_ids,
            "actual_coverage": _coverage(raw_targets),
            "lesion_count": grouped["lesion_count"],
            "negative_count": grouped["negative_count"],
            "empty_image_count": grouped["empty_image_count"],
            "matched_class_counts": grouped["matched_class_counts"],
            "lesion_indices": grouped["lesion_indices"],
            "negative_indices": grouped["negative_indices"],
            "low_score_lesion_count_below_0_5": grouped["low_score_lesion_count_below_0_5"],
            "teacher_lower_gt_class_count": grouped["teacher_lower_gt_class_count"],
            "lesion_probability_samples": grouped["lesion_probability_samples"],
            "detection_loss": float(detection.detach()),
            "positive_loss_raw": float(grouped["positive_loss"].detach()),
            "negative_loss_raw": float(grouped["negative_loss"].detach()),
            "classification_loss_raw": float(grouped["loss"].detach()),
            "teacher_topk_calls": teacher_audit["topk_calls"],
            "alignment": teacher_audit["alignment"],
        }
        if record["purpose"] == "empty_zero_check":
            empty_check_count += 1
            if grouped["empty_image_count"] != 2 or grouped["lesion_count"] or grouped["negative_count"] or float(grouped["loss"].detach()) != 0:
                raise RuntimeError("prelocked empty-only batch has nonzero classification KD")
            row["decoder_l3_gradient_classification_raw"] = 0.0
            row["valid_for_coefficient"] = False
        elif record["purpose"] == "calibration":
            effective = grouped["lesion_count"] + grouped["negative_count"] > 0
            row["valid_for_coefficient"] = effective
            if effective:
                det_norm = _gradient_norm(detection, decoder_l3)
                cls_norm = _gradient_norm(grouped["loss"], decoder_l3)
                pos_norm = _gradient_norm(0.5 * grouped["positive_loss"], decoder_l3)
                neg_norm = _gradient_norm(0.5 * grouped["negative_loss"], decoder_l3)
                if not all(math.isfinite(value) for value in (det_norm, cls_norm, pos_norm, neg_norm)):
                    raise FloatingPointError(f"non-finite decoder-L3 calibration gradient in {record['name']}")
                if det_norm <= 0 or cls_norm <= 0:
                    raise RuntimeError(f"zero decoder-L3 calibration gradient in {record['name']}: det={det_norm}, cls={cls_norm}")
                row.update({
                    "decoder_l3_gradient_detection": det_norm,
                    "decoder_l3_gradient_positive_raw_half": pos_norm,
                    "decoder_l3_gradient_negative_raw_half": neg_norm,
                    "decoder_l3_gradient_classification_raw": cls_norm,
                    "raw_ratio": cls_norm / det_norm,
                })
                valid_ratios.append(row["raw_ratio"])
                for category, count in grouped["matched_class_counts"].items():
                    matched_class_counts[category] += int(count)
        else:
            raise ValueError("unknown calibration manifest purpose")
        rows.append(row)
        del detection, grouped, outputs, replay, teacher_audit

    if len(rows) != len(manifest["batches"]) or empty_check_count != 1:
        raise RuntimeError("fixed calibration manifest was not fully covered")
    if len(valid_ratios) < 4 or any(count <= 0 for count in matched_class_counts.values()):
        raise RuntimeError(f"insufficient valid calibration or class coverage: ratios={len(valid_ratios)}, classes={matched_class_counts}")
    median_raw = statistics.median(valid_ratios)
    if median_raw <= 0 or not math.isfinite(median_raw):
        raise RuntimeError("invalid median raw classification/detection gradient ratio")
    coefficient = args.target_ratio / median_raw
    for row in rows:
        if row["valid_for_coefficient"]:
            row["weighted_ratio"] = coefficient * row["raw_ratio"]
            row["positive_loss_weighted"] = 0.5 * coefficient * row["positive_loss_raw"]
            row["negative_loss_weighted"] = 0.5 * coefficient * row["negative_loss_raw"]
            row["decoder_l3_gradient_positive_weighted"] = coefficient * row["decoder_l3_gradient_positive_raw_half"]
            row["decoder_l3_gradient_negative_weighted"] = coefficient * row["decoder_l3_gradient_negative_raw_half"]
    payload = {
        "scope": "baseline EMA, train-only fixed IDs/real transforms/DN, no optimizer step",
        "rule": "lambda_cls = 0.30 / median(valid-batch raw grouped-KD/detection decoder-L3 gradient ratios)",
        "manifest": manifest, "manifest_sha256": _sha256(args.manifest),
        "config": args.config, "target_ratio": args.target_ratio,
        "valid_batch_count": len(valid_ratios),
        "matched_class_counts": matched_class_counts,
        "median_raw_ratio": median_raw,
        "locked_output_weight": coefficient,
        "weighted_ratio_median": statistics.median(value * coefficient for value in valid_ratios),
        "rows": rows,
        "optimizer_steps": 0,
        "teacher_state_max_delta": _maximum_delta(teacher_before, teacher),
        "teacher_gradient_count": sum(parameter.grad is not None for parameter in teacher.parameters()),
        "student_trainable_parameter_mask": trainable,
        "amp": "autocast forward; autograd.grad on unscaled FP32 losses",
        "checkpoint_sha256_before": checkpoint_hash_before,
        "checkpoint_sha256_after": _sha256(args.checkpoint),
    }
    if payload["teacher_state_max_delta"] != 0 or payload["teacher_gradient_count"] != 0 or checkpoint_hash_before != payload["checkpoint_sha256_after"]:
        raise RuntimeError("frozen Teacher or baseline checkpoint changed")
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")
    output.with_suffix(".COMPLETED").write_text("KQ-B grouped-KD calibration complete\n")
    print(json.dumps({
        "locked_output_weight": coefficient,
        "valid_batch_count": len(valid_ratios),
        "matched_class_counts": matched_class_counts,
        "weighted_ratio_median": payload["weighted_ratio_median"],
    }, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--img-folder", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
