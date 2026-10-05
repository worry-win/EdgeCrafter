"""One-batch bounded gradient smoke for the KQ shared-query route."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ecdetseg"))
sys.path.insert(0, str(ROOT))

from engine.core import YAMLConfig  # noqa: E402
from engine.misc import dist_utils  # noqa: E402
from engine.solver import TASKS  # noqa: E402
from scripts.ablation.cmp5L_shared_query_kd import (  # noqa: E402
    cosine_query_kd,
    layer_query_audit,
    normal_query_slice,
    shared_query_alignment_audit,
)
from scripts.ablation.probe_cmp5L_shared_query_kd_stage1 import (  # noqa: E402
    _build_train,
    _run_direct_teacher,
    _train_targets_to_absolute,
)
from scripts.ablation.cmp5L_query_behavior_kd import run_model_with_capture  # noqa: E402
from scripts.ablation.probe_cmp5L_privileged_ema_oracle import privilege_features  # noqa: E402


def _state_snapshot(module):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _max_state_delta(before, module):
    maximum = 0.0
    after = module.state_dict()
    if set(before) != set(after):
        raise RuntimeError("state keys changed")
    for name, reference in before.items():
        value = after[name].detach().cpu()
        if value.dtype == torch.bool:
            delta = float(torch.logical_xor(value, reference).any())
        elif value.numel():
            delta = float((value - reference).abs().max())
        else:
            delta = 0.0
        maximum = max(maximum, delta)
    return maximum


def _norm(values):
    terms = [value.detach().float().square().sum() for value in values if value is not None]
    return float(torch.stack(terms).sum().sqrt()) if terms else 0.0


def _autograd_norm(loss, parameters):
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    if not parameters:
        return 0.0
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return _norm(gradients)


def _finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def _build(args, out_dir):
    cfg = YAMLConfig(
        args.config,
        **{
            "device": str(args.device),
            "tuning": args.checkpoint,
            "output_dir": str(out_dir / "solver"),
            "use_amp": True,
            "train_dataloader": {
                "num_workers": 0,
                "total_batch_size": args.batch_size,
                "shuffle": False,
                "drop_last": False,
            },
            "num_classes": 4,
            "remap_mscoco_category": False,
        },
    )
    for name in ("ViTAdapter", "DinoV2Adapter"):
        if name in cfg.yaml_cfg:
            cfg.yaml_cfg[name]["skip_load_backbone"] = True
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.train()
    student = dist_utils.de_parallel(solver.model).to(args.device)
    return cfg, solver, student


def _teacher_replays(teacher, replay, samples, absolute_targets, *, amp_enabled):
    with torch.no_grad(), torch.autocast(device_type=samples.device.type, enabled=amp_enabled):
        teacher_backbone = teacher.backbone(samples)
        teacher_encoded = teacher.encoder([feature.clone() for feature in teacher_backbone])
        normal_encoded = [feature.clone() for feature in teacher_encoded]
        privileged_encoded, masks = privilege_features(
            teacher_encoded,
            absolute_targets,
            samples.shape[-2:],
            background_weight=0.2,
            context_weight=None,
            context_scale=1.5,
        )
        normal_memory, shapes = teacher.decoder._get_encoder_input(normal_encoded)
        privileged_memory, privileged_shapes = teacher.decoder._get_encoder_input(privileged_encoded)
        if tuple(shapes) != tuple(privileged_shapes):
            raise RuntimeError("normal/privileged spatial shapes differ")
        _, normal_decoder_outputs, normal_trace, normal_topk = _run_direct_teacher(
            teacher, replay, normal_memory, shapes
        )
        _, privileged_decoder_outputs, privileged_trace, privileged_topk = _run_direct_teacher(
            teacher, replay, privileged_memory, shapes
        )
    if normal_topk or privileged_topk:
        raise RuntimeError("teacher attempted Top-K reselection")
    normal_outputs = {
        "pred_boxes": normal_decoder_outputs[0][-1],
        "pred_logits": normal_decoder_outputs[1][-1],
    }
    privileged_outputs = {
        "pred_boxes": privileged_decoder_outputs[0][-1],
        "pred_logits": privileged_decoder_outputs[1][-1],
    }
    return normal_outputs, normal_trace, privileged_outputs, privileged_trace, masks


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    if args.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    if args.device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    cfg, solver, student = _build(args, out_dir)
    student.train()
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    student_before = _state_snapshot(student)
    teacher_before = _state_snapshot(teacher)
    optimizer_ids = {id(parameter) for group in solver.optimizer.param_groups for parameter in group["params"]}
    trainable_student_ids = {
        id(parameter) for parameter in student.parameters() if parameter.requires_grad
    }
    if not trainable_student_ids <= optimizer_ids:
        missing = sum(int(id(parameter) not in optimizer_ids) for parameter in student.parameters() if parameter.requires_grad)
        raise RuntimeError(f"trainable student parameters missing from optimizer: {missing}")
    if solver.ema is None:
        raise RuntimeError("EMA is required for KQ training")
    teacher_parameter_ids = {id(parameter) for parameter in teacher.parameters()}
    ema_parameter_ids = {id(parameter) for parameter in solver.ema.module.parameters()}
    groups = {"KQ0": (), "KQ1": (3,), "KQ2": (0, 1, 2, 3), "KQ3": (0, 1, 2, 3)}
    records = {}
    loader = cfg.train_dataloader
    batch = next(iter(loader))
    samples, raw_targets = batch
    samples = samples.to(args.device)
    targets = [{key: value.to(args.device) if torch.is_tensor(value) else value for key, value in target.items()} for target in raw_targets]
    absolute_targets = _train_targets_to_absolute(raw_targets, samples.shape[-2:], args.device)
    initial_state = copy.deepcopy(student.state_dict())
    initial_optimizer = copy.deepcopy(solver.optimizer.state_dict())
    amp_enabled = solver.scaler is not None and not args.disable_amp
    for group, layers in groups.items():
        student.load_state_dict(initial_state, strict=True)
        solver.optimizer.load_state_dict(initial_optimizer)
        solver.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=args.device.type, enabled=amp_enabled):
            outputs, student_trace, replay_inputs = run_model_with_capture(student, samples, targets)
        replay = type("Replay", (), {})()
        replay.initial_query = replay_inputs.initial_query[:, -student.decoder.num_queries:]
        replay.initial_reference_unactivated = replay_inputs.initial_reference_unactivated[:, -student.decoder.num_queries:]
        replay.topk_indices = replay_inputs.topk_indices if hasattr(replay_inputs, "topk_indices") else torch.empty(0, device=args.device)
        replay.memory = replay_inputs.memory
        replay.spatial_shapes = replay_inputs.spatial_shapes
        replay.attention_mask = replay_inputs.attention_mask
        replay.denoising_metadata = None
        replay.normal_query_count = student.decoder.num_queries
        # The direct replay helper only needs initial query/reference and memory;
        # top-k indices are audited separately from the student capture.
        normal_outputs, normal_trace, privileged_outputs, privileged_trace, masks = _teacher_replays(
            teacher, replay, samples, absolute_targets, amp_enabled=amp_enabled
        )
        student_layers = [normal_query_slice(layer, student.decoder.num_queries) for layer in student_trace["query_outputs"]]
        normal_layers = [normal_query_slice(layer, student.decoder.num_queries) for layer in normal_trace["query_outputs"]]
        privileged_layers = [normal_query_slice(layer, student.decoder.num_queries) for layer in privileged_trace["query_outputs"]]
        if not _finite((outputs, normal_outputs, privileged_outputs, student_layers, normal_layers, privileged_layers)):
            raise RuntimeError(f"non-finite outputs in {group}")
        parity = {
            "logits": float((outputs["pred_logits"].detach() - normal_outputs["pred_logits"]).abs().max()),
            "boxes": float((outputs["pred_boxes"].detach() - normal_outputs["pred_boxes"]).abs().max()),
        }
        with torch.autocast(device_type=args.device.type, enabled=False):
            detection = sum(solver.criterion(outputs, targets, epoch=0, step=0, global_step=0, epoch_step=1).values())
            if group == "KQ0":
                kd_info = {"loss": detection * 0.0, "layer_losses": [], "layers": []}
            else:
                kd_teacher = privileged_layers if group in {"KQ1", "KQ2"} else normal_layers
                kd_info = cosine_query_kd(
                    [layer.float() for layer in student_layers],
                    [layer.float() for layer in kd_teacher],
                    layers=layers,
                    detach_teacher=True,
                )
            total = detection + kd_info["loss"]
        detector_parameters = list(student.parameters())
        detection_grad = _autograd_norm(detection, detector_parameters)
        kd_grad = _autograd_norm(kd_info["loss"], detector_parameters) if group != "KQ0" else 0.0
        if amp_enabled and solver.scaler is not None:
            solver.scaler.scale(total).backward()
            solver.scaler.unscale_(solver.optimizer)
        else:
            total.backward()
        total_grad = _norm([parameter.grad for parameter in detector_parameters])
        if not math.isfinite(total_grad) or total_grad <= 0.0:
            nonfinite_names = [
                name for name, parameter in student.named_parameters()
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            ]
            raise RuntimeError(
                f"invalid total gradient in {group}: {total_grad}; "
                f"detection={float(detection.detach())}; kd={float(kd_info['loss'].detach())}; "
                f"amp_scale={float(solver.scaler.get_scale()) if solver.scaler is not None else None}; "
                f"nonfinite_gradient_names={nonfinite_names[:20]}"
            )
        if group != "KQ0" and (not math.isfinite(kd_grad) or kd_grad <= 0.0):
            raise RuntimeError(f"KD gradient is zero in {group}: {kd_grad}")
        records[group] = {
            "layers": list(layers),
            "parity": parity,
            "detection_loss": float(detection.detach()),
            "kd_loss": float(kd_info["loss"].detach()),
            "layer_losses": [float(value.detach()) for value in kd_info["layer_losses"]],
            "gradient_norms_unscaled": {"detection": detection_grad, "kd": kd_grad, "total": total_grad},
            "student_hidden_audit": layer_query_audit(student_layers, normal_layers),
            "privileged_hidden_audit": layer_query_audit(student_layers, privileged_layers),
            "suppressed_fraction": [float((mask < 0.999999).float().mean()) for mask in masks],
            "teacher_requires_grad_count": sum(int(parameter.requires_grad) for parameter in teacher.parameters()),
        }
    payload = {
        "meta": {"config": args.config, "checkpoint": args.checkpoint, "groups": {key: list(value) for key, value in groups.items()}, "amp_enabled": amp_enabled, "batch_size": len(targets), "seed": args.seed},
        "checks": {"cuda": args.device.type == "cuda", "student_optimizer_coverage": True, "student_trainable_parameter_count": len(trainable_student_ids), "student_frozen_parameter_count": sum(int(not parameter.requires_grad) for parameter in student.parameters()), "teacher_requires_grad_count": sum(int(parameter.requires_grad) for parameter in teacher.parameters()), "teacher_in_student_ema": bool(teacher_parameter_ids & ema_parameter_ids), "teacher_parameter_max_delta": _max_state_delta(teacher_before, teacher), "student_parameter_max_delta_without_step": _max_state_delta(student_before, student), "finite": True},
        "groups": records,
    }
    if payload["checks"]["teacher_requires_grad_count"] != 0 or payload["checks"]["teacher_in_student_ema"]:
        raise RuntimeError("teacher lifecycle audit failed")
    (out_dir / "audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out_dir / "COMPLETED").write_text("KQ bounded gradient smoke complete\n")
    print(json.dumps(payload, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--disable-amp", action="store_true")
    args = parser.parse_args()
    args.device = torch.device(args.device)
    try:
        run(args)
    except Exception as error:
        output_dir = Path(args.out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "FAILED.json").write_text(json.dumps({"error_type": type(error).__name__, "error": str(error)}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
