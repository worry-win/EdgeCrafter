"""No-update, shared fixed-batch gradient calibration for KQ-R and KQ-U."""

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
from scripts.ablation.cmp5L_shared_query_kd import (  # noqa: E402
    candidate_ranking_kd, final_layer_hungarian_matches, one_way_protected_kd,
)
from scripts.ablation.probe_cmp5L_shared_query_kd_stage1 import _build_train  # noqa: E402
from scripts.ablation.train_cmp5L_shared_query_kd import (  # noqa: E402
    _load_frozen_teacher, _module, _move_targets, _student_forward,
    _targets_absolute_xyxy, _teacher_forward,
)


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    manifest = json.loads(Path(args.manifest).read_text())
    if manifest["seed"] != args.seed or manifest["batch_size"] != 2:
        raise ValueError("manifest seed/batch differs")
    build_args = SimpleNamespace(
        config=args.config, checkpoint=args.checkpoint,
        ann_file=args.ann_file, img_folder=args.img_folder,
        num_workers=0, batch_size=2, weights="ema", device=device, seed=args.seed,
    )
    cfg, solver, student = _build_train(build_args)
    recipe = cfg.yaml_cfg["train_dataloader"]
    transforms = recipe["dataset"]["transforms"]
    schedule = (transforms["mosaic_epoch"], recipe["collate_fn"]["mixup_epoch"], transforms["stop_epoch"])
    if schedule != (12, 12, 13) or cfg.yaml_cfg["epochs"] != 15:
        raise RuntimeError(f"calibration recipe drift: {schedule}, epochs={cfg.yaml_cfg['epochs']}")
    trainable = restore_trainable_parameter_mask(student, _module(solver.model))
    student.train()
    if getattr(student, "_freeze_backbone", False):
        student.backbone.eval()
    solver.criterion.train()
    teacher = _load_frozen_teacher(student, args.teacher_checkpoint, device)
    teacher.eval().requires_grad_(False)
    runtime = SimpleNamespace(teacher=teacher, spec=SimpleNamespace(teacher_memory_mode="privileged_np", num_queries=300))
    teacher_before = _snapshot(teacher)
    checkpoint_before = _sha256(args.checkpoint)
    decoder_l3 = [parameter for name, parameter in student.named_parameters()
                  if parameter.requires_grad and name.startswith("decoder.decoder.layers.3.")]
    if not decoder_l3:
        raise RuntimeError("empty trainable decoder L3 bucket")
    loader = cfg.train_dataloader
    image_index = {int(image_id): index for index, image_id in enumerate(loader.dataset.ids)}
    if len(image_index) != len(loader.dataset.ids):
        raise RuntimeError("nonunique train IDs")
    objective = {"KQR": candidate_ranking_kd, "KQU": one_way_protected_kd}[args.group]
    rows, ratios = [], []
    matched_classes = {str(i): 0 for i in range(4)}
    active_classes = {str(i): 0 for i in range(4)}
    for batch_index, record in enumerate(manifest["batches"]):
        image_ids = [int(item) for item in record["image_ids"]]
        if len(image_ids) != 2 or any(item not in image_index for item in image_ids):
            raise ValueError(f"invalid fixed image IDs: {record['name']}")
        epoch = int(record["epoch"])
        loader.set_epoch(epoch)
        samples, raw_targets = loader.collate_fn([loader.dataset[image_index[item]] for item in image_ids])
        samples = samples.to(device)
        targets = _move_targets(raw_targets, device)
        absolute = _targets_absolute_xyxy(raw_targets, samples.shape[-2:], device)
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
            outputs, _trace, replay, topk = _student_forward(student, samples, targets)
            _teacher_layers, teacher_audit = _teacher_forward(runtime, replay, topk, samples, absolute, True)
        with torch.autocast(device_type="cuda", enabled=False):
            det_dict = solver.criterion(outputs, targets, epoch=epoch, step=batch_index,
                                        global_step=epoch * len(loader) + batch_index,
                                        epoch_step=len(loader))
            detection = sum(det_dict.values())
            matches = final_layer_hungarian_matches(solver.criterion.matcher, outputs, targets)
            result = objective(outputs["pred_logits"].float(), teacher_audit["teacher_logits"][-1].float(),
                               outputs["pred_boxes"], targets, matches,
                               normal_query_count=300, negative_topk=20, negative_max_iou=.3)
        valid_targets = (result["valid_pair_count"] if args.group == "KQR" else
                         result["lesion_active_count"] + result["negative_active_dimension_count"])
        row = {
            "name": record["name"], "purpose": record["purpose"], "epoch": epoch,
            "locked_image_ids": image_ids, "actual_coverage": _coverage(raw_targets),
            "detection_loss": float(detection.detach()),
            "new_kd_loss_raw": float(result["loss"].detach()),
            "valid_target_count": valid_targets,
            "teacher_topk_calls": teacher_audit["topk_calls"],
            "alignment": teacher_audit["alignment"],
            **{key: value for key, value in result.items()
               if key not in ("loss", "positive_loss", "negative_loss")},
        }
        if args.group == "KQU":
            row["positive_loss_raw"] = float(result["positive_loss"].detach())
            row["negative_loss_raw"] = float(result["negative_loss"].detach())
        if record["purpose"] == "empty_zero_check":
            if result["empty_image_count"] != 2 or valid_targets or float(result["loss"].detach()) != 0:
                raise RuntimeError("empty-only new KD must be exactly zero")
            row["valid_for_coefficient"] = False
        elif record["purpose"] == "calibration":
            row["valid_for_coefficient"] = bool(valid_targets)
            if valid_targets:
                det_norm = _gradient_norm(detection, decoder_l3)
                kd_norm = _gradient_norm(result["loss"], decoder_l3)
                if not all(math.isfinite(value) for value in (det_norm, kd_norm)) or det_norm <= 0 or kd_norm <= 0:
                    raise RuntimeError(f"nonfinite/zero calibration gradient: {record['name']}, det={det_norm}, kd={kd_norm}")
                row.update({"decoder_l3_gradient_detection": det_norm,
                            "decoder_l3_gradient_new_kd_raw": kd_norm,
                            "raw_ratio": kd_norm / det_norm})
                ratios.append(row["raw_ratio"])
                for key, count in result["matched_class_counts"].items():
                    matched_classes[key] += int(count)
                effective = result["valid_class_counts"] if args.group == "KQR" else result["active_class_counts"]
                for key, count in effective.items():
                    active_classes[key] += int(count)
        else:
            raise ValueError("unknown manifest purpose")
        rows.append(row)
        del outputs, replay, result, detection, teacher_audit
    if len(rows) != 7 or len(ratios) < 4 or any(count == 0 for count in matched_classes.values()):
        raise RuntimeError(f"insufficient fixed-batch coverage: rows={len(rows)}, ratios={len(ratios)}, matched={matched_classes}")
    median_raw = statistics.median(ratios)
    if not math.isfinite(median_raw) or median_raw < args.minimum_raw_ratio:
        raise RuntimeError(f"raw KD gradient persistently near zero: median={median_raw}")
    coefficient = args.target_ratio / median_raw
    if not math.isfinite(coefficient) or coefficient > args.maximum_coefficient:
        raise RuntimeError(f"compensating coefficient too large: {coefficient}")
    for row in rows:
        if row["valid_for_coefficient"]:
            row["weighted_ratio"] = coefficient * row["raw_ratio"]
            row["new_kd_loss_weighted"] = coefficient * row["new_kd_loss_raw"]
    payload = {
        "group": args.group,
        "scope": "baseline EMA, train-only fixed IDs/real transforms/DN, no optimizer step",
        "rule": "lambda = 0.30 / median(valid-batch raw new-KD/detection decoder-L3 gradient ratio)",
        "manifest": manifest, "manifest_sha256": _sha256(args.manifest),
        "config": args.config, "schedule": schedule, "target_ratio": args.target_ratio,
        "minimum_raw_ratio": args.minimum_raw_ratio,
        "maximum_coefficient": args.maximum_coefficient,
        "valid_batch_count": len(ratios), "zero_target_batch_count": sum(not row["valid_for_coefficient"] for row in rows),
        "matched_class_counts": matched_classes, "active_class_counts": active_classes,
        "median_raw_ratio": median_raw, "locked_output_weight": coefficient,
        "weighted_ratio_median": statistics.median(value * coefficient for value in ratios),
        "rows": rows, "optimizer_steps": 0,
        "teacher_state_max_delta": _maximum_delta(teacher_before, teacher),
        "teacher_gradient_count": sum(parameter.grad is not None for parameter in teacher.parameters()),
        "student_trainable_parameter_mask": trainable,
        "amp": "autocast forward; autograd.grad on unscaled FP32 losses",
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": _sha256(args.checkpoint),
    }
    if payload["teacher_state_max_delta"] != 0 or payload["teacher_gradient_count"] or checkpoint_before != payload["checkpoint_sha256_after"]:
        raise RuntimeError("Teacher or baseline checkpoint changed")
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")
    out.with_suffix(".COMPLETED").write_text("calibration complete\n")
    print(json.dumps({"group": args.group, "locked_output_weight": coefficient,
                      "valid_batch_count": len(ratios), "matched_class_counts": matched_classes,
                      "active_class_counts": active_classes, "weighted_ratio_median": payload["weighted_ratio_median"]}, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("KQR", "KQU"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--img-folder", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-ratio", type=float, default=.30)
    parser.add_argument("--minimum-raw-ratio", type=float, default=.003)
    parser.add_argument("--maximum-coefficient", type=float, default=100.)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
