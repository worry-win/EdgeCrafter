"""Production entrypoint for cmp5L shared-student-query distillation.

KQ0 deliberately does not use this file: the launcher routes it through the
unchanged ``ecdetseg/train.py`` entrypoint.  KQ1--KQ3 use this opt-in trainer;
the frozen full teacher lives only in a process-global runtime so it cannot be
serialized by ``ECSolver.state_dict`` or included in Student EMA/optimizer.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.multiprocessing as mp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "ecdetseg"))
sys.path.insert(0, str(PROJECT_ROOT))

from engine.core import YAMLConfig, yaml_utils  # noqa: E402
from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy  # noqa: E402
from engine.misc import MetricLogger, SmoothedValue, dist_utils  # noqa: E402
from engine.optim import ModelEMA  # noqa: E402
from engine.solver.ec_engine import _BatchSlice, _optimizer_step_due  # noqa: E402
from engine.solver.ec_solver import ECSolver  # noqa: E402
import engine.solver.ec_solver as ec_solver_module  # noqa: E402

from scripts.ablation.cmp5L_query_behavior_kd import (  # noqa: E402
    DecoderReplayInputs,
    _run_decoder_with_capture,
    run_model_with_capture,
)
from scripts.ablation.cmp5L_shared_query_kd import (  # noqa: E402
    SharedQueryReplay,
    candidate_ranking_kd,
    compose_shared_query_loss,
    cosine_query_kd,
    final_layer_hungarian_matches,
    grouped_sigmoid_output_kd,
    layer_query_audit,
    normal_query_slice,
    one_way_protected_kd,
    resolve_kq_spec,
    shared_query_alignment_audit,
    sigmoid_output_kd,
    validate_kq_config,
)
from scripts.ablation.probe_cmp5L_privileged_ema_oracle import (  # noqa: E402
    privilege_features,
)


def _rank() -> int:
    return int(torch.distributed.get_rank()) if torch.distributed.is_initialized() else 0


def _module(model):
    return model.module if hasattr(model, "module") else model


def _move_targets(targets, device):
    return [
        {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in target.items()
        }
        for target in targets
    ]


def _image_id_repr(target):
    value = target.get("image_id")
    if torch.is_tensor(value):
        return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)] if value is not None else []


def _targets_absolute_xyxy(targets, image_hw, device):
    height, width = image_hw
    scale = torch.tensor([width, height, width, height], device=device)
    converted = []
    for target in targets:
        boxes = target["boxes"].as_subclass(torch.Tensor).to(device)
        if boxes.numel() and (float(boxes.min()) < -1e-6 or float(boxes.max()) > 1.000001):
            raise RuntimeError("training boxes must be normalized cxcywh")
        converted.append({
            "boxes": box_cxcywh_to_xyxy(boxes) * scale,
            "labels": target["labels"].to(device),
        })
    return converted


def _snapshot_named_tensors(named_values):
    return {name: value.detach().cpu().clone() for name, value in named_values}


def _maximum_delta(before, named_values):
    after = dict(named_values)
    if set(before) != set(after):
        raise RuntimeError("teacher state keys changed")
    maximum = 0.0
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


def _load_frozen_teacher(student, checkpoint_path, device):
    teacher = copy.deepcopy(student).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    source = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    if source and next(iter(source)).startswith("module."):
        source = {key[7:]: value for key, value in source.items()}
    teacher.load_state_dict(source, strict=True)
    teacher.eval().requires_grad_(False)
    return teacher


def _capture_student_topk(ec_transformer):
    original = ec_transformer._select_topk
    holder = {"indices": None, "anchors_unactivated": None}

    def wrapped(memory, logits, anchors_unactivated, topk):
        method = ec_transformer.query_select_method
        if method == "default":
            indices = torch.topk(logits.max(-1).values, topk, dim=-1).indices
        elif method == "one2many":
            indices = torch.topk(logits.flatten(1), topk, dim=-1).indices
            indices = indices // ec_transformer.num_classes
        elif method == "agnostic":
            indices = torch.topk(logits.squeeze(-1), topk, dim=-1).indices
        else:
            raise RuntimeError(f"unsupported query selection method: {method}")
        holder["indices"] = indices.detach().clone()
        holder["anchors_unactivated"] = anchors_unactivated.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, anchors_unactivated.shape[-1])
        ).detach().clone()
        return original(memory, logits, anchors_unactivated, topk)

    ec_transformer._select_topk = wrapped
    return original, holder


def _student_forward(model, samples, targets):
    transformer = _module(model).decoder
    original, topk = _capture_student_topk(transformer)
    try:
        outputs, trace, replay = run_model_with_capture(model, samples, targets)
    finally:
        transformer._select_topk = original
    if topk["indices"] is None:
        raise RuntimeError("Student Top-K was not captured")
    return outputs, trace, replay, topk


def _teacher_forward(runtime, replay, topk, samples, absolute_targets, autocast_enabled):
    teacher = runtime.teacher
    num_queries = runtime.spec.num_queries
    q0 = replay.initial_query[:, -num_queries:].detach()
    r0 = replay.initial_reference_unactivated[:, -num_queries:].detach()
    with torch.no_grad(), torch.autocast(
        device_type=samples.device.type,
        enabled=autocast_enabled,
        cache_enabled=True,
    ):
        features = teacher.backbone(samples)
        encoded = teacher.encoder([feature.clone() for feature in features])
        masks = []
        normal_encoded = [feature.clone() for feature in encoded]
        if runtime.spec.teacher_memory_mode == "privileged_np":
            encoded, masks = privilege_features(
                encoded,
                absolute_targets,
                samples.shape[-2:],
                background_weight=getattr(runtime.spec, 'teacher_background_weight', 0.2),
                context_weight=None,
                context_scale=1.5,
            )
        memory, shapes = teacher.decoder._get_encoder_input(encoded)
        normal_memory, normal_shapes = teacher.decoder._get_encoder_input(normal_encoded)
        if tuple(shapes) != tuple(normal_shapes):
            raise RuntimeError("teacher memory spatial shapes changed")
        direct = SharedQueryReplay(
            initial_query=q0,
            initial_reference_unactivated=r0,
            topk_indices=topk["indices"],
            memory=memory,
            spatial_shapes=shapes,
            normal_query_count=num_queries,
        )
        student_replay = SharedQueryReplay(
            initial_query=q0,
            initial_reference_unactivated=r0,
            topk_indices=topk["indices"],
            memory=replay.memory,
            spatial_shapes=replay.spatial_shapes,
            normal_query_count=num_queries,
        )
        alignment = shared_query_alignment_audit(student_replay, direct)
        if not all(alignment[name]["exact"] for name in (
            "initial_query", "initial_reference_unactivated", "topk_indices"
        )):
            raise RuntimeError(f"shared query alignment failed: {alignment}")
        direct_inputs = DecoderReplayInputs(
            initial_query=q0,
            initial_reference_unactivated=r0,
            memory=memory,
            spatial_shapes=shapes,
            attention_mask=None,
            denoising_metadata=None,
            normal_query_count=num_queries,
        )
        original_topk = teacher.decoder._select_topk
        topk_calls = {"count": 0}

        def forbidden(*_args, **_kwargs):
            topk_calls["count"] += 1
            raise RuntimeError("frozen teacher attempted Top-K reselection")

        teacher.decoder._select_topk = forbidden
        try:
            decoder_outputs, trace = _run_decoder_with_capture(teacher.decoder, direct_inputs)
        finally:
            teacher.decoder._select_topk = original_topk
        if topk_calls["count"]:
            raise RuntimeError("frozen teacher attempted Top-K reselection")
    teacher_layers = [normal_query_slice(layer, num_queries) for layer in trace["query_outputs"]]
    return teacher_layers, {
        "alignment": alignment,
        "topk_calls": topk_calls["count"],
        "memory_max_abs_delta": float((memory - normal_memory).abs().max()),
        "memory_mean_abs_delta": float((memory - normal_memory).abs().mean()),
        "suppressed_fraction": [
            float((mask < 0.999999).float().mean()) for mask in masks
        ],
        "teacher_boxes": decoder_outputs[0].detach(),
        "teacher_logits": decoder_outputs[1].detach(),
    }


def _grad_norm(parameters):
    squares = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


def _autograd_component_norms(loss, named_parameters):
    selected = [
        (name, parameter)
        for name, parameter in named_parameters
        if parameter.requires_grad
    ]
    gradients = torch.autograd.grad(
        loss,
        [parameter for _, parameter in selected],
        retain_graph=True,
        allow_unused=True,
    )
    buckets = {"all": [], "backbone": [], "encoder": [], "decoder": []}
    for (name, _), gradient in zip(selected, gradients):
        if gradient is None:
            continue
        term = gradient.detach().float().square().sum()
        buckets["all"].append(term)
        prefix = name.split(".", 1)[0]
        if prefix in buckets:
            buckets[prefix].append(term)
    return {
        key: float(torch.stack(terms).sum().sqrt()) if terms else 0.0
        for key, terms in buckets.items()
    }


def _nonfinite_gradients(module):
    return [
        name for name, parameter in module.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]


@dataclass
class KQRuntime:
    teacher: torch.nn.Module
    spec: object
    max_train_steps: int
    strict_debug_checks: bool
    teacher_parameters_before: dict
    teacher_buffers_before: dict
    diagnostics_path: Path
    processed_steps: int = 0
    optimizer_steps: int = 0
    skipped_optimizer_steps: int = 0
    finite_unscaled_gradient_steps: int = 0
    kd_gradient_seen: bool = False
    output_gradient_seen: bool = False
    component_gradient_recorded: bool = False
    scaler_history: list = field(default_factory=list)


ACTIVE_RUNTIME: KQRuntime | None = None


def _write_diagnostic(runtime, payload):
    payload = {"rank": _rank(), **payload}
    print(json.dumps(payload, sort_keys=True, default=str), flush=True)
    runtime.diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    with runtime.diagnostics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def train_one_epoch_kq(
    self_lr_scheduler,
    lr_scheduler,
    model,
    criterion,
    data_loader,
    optimizer,
    device,
    epoch,
    max_norm=0,
    **kwargs,
):
    runtime = ACTIVE_RUNTIME
    if runtime is None or not runtime.spec.enabled:
        raise RuntimeError("KQ production runtime was not initialized")
    model.train()
    module = _module(model)
    if getattr(module, "_freeze_backbone", False):
        module.backbone.eval()
    runtime.teacher.eval()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = kwargs.get("print_freq", 10)
    writer = kwargs.get("writer")
    ema: ModelEMA | None = kwargs.get("ema")
    scaler = kwargs.get("scaler")
    warmup = kwargs.get("lr_warmup_scheduler")
    accumulation = max(1, int(kwargs.get("gradient_accumulation_steps", 1)))
    start_step = max(0, int(kwargs.get("start_step", 0)))
    checkpoint_interval = max(0, int(kwargs.get("checkpoint_interval_steps", 0)))
    checkpoint_callback = kwargs.get("checkpoint_callback")
    optimizer.zero_grad(set_to_none=True)
    cur_iters = epoch * len(data_loader)
    epoch_loader = _BatchSlice(data_loader, start_step=start_step)
    grouped_epoch = {
        "lesions": 0, "hard_negatives": 0, "empty_images": 0,
        "low_score_lesions_below_0_5": 0, "teacher_lower_gt_class": 0,
        "matched_class_0": 0, "matched_class_1": 0,
        "matched_class_2": 0, "matched_class_3": 0,
    }
    new_epoch = {"lesions": 0, "valid_lesions": 0, "candidate_pairs": 0,
                 "valid_pairs": 0, "reverse_pairs": 0, "hard_negatives": 0,
                 "active_negative_dimensions": 0, "empty_images": 0,
                 "zero_target_images": 0,
                 **{f"matched_class_{i}": 0 for i in range(4)},
                 **{f"active_class_{i}": 0 for i in range(4)}}

    for relative_i, (samples, raw_targets) in enumerate(
        metric_logger.log_every(epoch_loader, print_freq, header)
    ):
        if runtime.max_train_steps and runtime.processed_steps >= runtime.max_train_steps:
            break
        step_started = time.perf_counter()
        runtime.processed_steps += 1
        i = start_step + relative_i
        samples = samples.to(device)
        targets = _move_targets(raw_targets, device)
        absolute_targets = _targets_absolute_xyxy(raw_targets, samples.shape[-2:], device)
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))
        autocast_enabled = scaler is not None
        with torch.autocast(device_type=str(device), enabled=autocast_enabled, cache_enabled=True):
            outputs, student_trace, replay, topk = _student_forward(model, samples, targets)
            teacher_layers, teacher_audit = _teacher_forward(
                runtime, replay, topk, samples, absolute_targets, autocast_enabled
            )
        student_layers = [
            normal_query_slice(layer, runtime.spec.num_queries)
            for layer in student_trace["query_outputs"]
        ]
        with torch.autocast(device_type=str(device), enabled=False):
            loss_dict = criterion(outputs, targets, **metas)
            detection_loss = sum(loss_dict.values())
            kd_info = cosine_query_kd(
                [layer.float() for layer in student_layers],
                [layer.float() for layer in teacher_layers],
                layers=runtime.spec.layers,
                detach_teacher=True,
            )
            output_info = None
            if runtime.spec.output_weight > 0:
                if runtime.spec.group in ("KQB", "KQR", "KQU"):
                    matches = final_layer_hungarian_matches(
                        criterion.matcher, outputs, targets,
                        normal_query_count=runtime.spec.num_queries,
                    )
                    image_count = torch.tensor([len(targets)], device=device, dtype=torch.long)
                    world_size = 1
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        world_size = torch.distributed.get_world_size()
                        torch.distributed.all_reduce(image_count)
                    objective = {
                        "KQB": grouped_sigmoid_output_kd,
                        "KQR": candidate_ranking_kd,
                        "KQU": one_way_protected_kd,
                    }[runtime.spec.group]
                    output_info = objective(
                        outputs["pred_logits"].float(),
                        teacher_audit["teacher_logits"][-1].float(),
                        outputs["pred_boxes"], targets, matches,
                        normal_query_count=runtime.spec.num_queries,
                        negative_topk=20, negative_max_iou=0.3,
                        global_image_count=int(image_count.item()),
                        ddp_world_size=world_size,
                    )
                else:
                    output_info = sigmoid_output_kd(
                        outputs["pred_logits"].float(),
                        teacher_audit["teacher_logits"][-1].float(),
                        normal_query_count=runtime.spec.num_queries,
                        detach_teacher=True,
                    )
            total_loss = compose_shared_query_loss(
                detection_loss,
                kd_info["loss"],
                runtime.spec,
                output_kd_loss=output_info["loss"] if output_info else None,
            )
            weighted_hidden = kd_info["loss"] * runtime.spec.weight
            weighted_output = (
                output_info["loss"] * runtime.spec.output_weight
                if output_info else detection_loss.new_zeros(())
            )
            loss_dict["loss_shared_query_kd"] = weighted_hidden
            if output_info:
                loss_dict["loss_shared_query_output_kd"] = weighted_output

        if runtime.spec.group == "KQB":
            grouped_epoch["lesions"] += output_info["lesion_count"]
            grouped_epoch["hard_negatives"] += output_info["negative_count"]
            grouped_epoch["empty_images"] += output_info["empty_image_count"]
            grouped_epoch["low_score_lesions_below_0_5"] += output_info["low_score_lesion_count_below_0_5"]
            grouped_epoch["teacher_lower_gt_class"] += output_info["teacher_lower_gt_class_count"]
            for class_id in range(4):
                grouped_epoch[f"matched_class_{class_id}"] += output_info["matched_class_counts"][str(class_id)]

        if runtime.spec.group in ("KQR", "KQU"):
            new_epoch["empty_images"] += output_info["empty_image_count"]
            new_epoch["zero_target_images"] += output_info["zero_target_image_count"]
            for class_id in range(4):
                new_epoch[f"matched_class_{class_id}"] += output_info["matched_class_counts"][str(class_id)]
            if runtime.spec.group == "KQR":
                new_epoch["lesions"] += sum(output_info["matched_class_counts"].values())
                new_epoch["valid_lesions"] += sum(output_info["valid_class_counts"].values())
                new_epoch["candidate_pairs"] += output_info["candidate_pair_count"]
                new_epoch["valid_pairs"] += output_info["valid_pair_count"]
                new_epoch["reverse_pairs"] += output_info["reverse_pair_count"]
                for class_id in range(4):
                    new_epoch[f"active_class_{class_id}"] += output_info["valid_class_counts"][str(class_id)]
            else:
                new_epoch["lesions"] += output_info["lesion_count"]
                new_epoch["valid_lesions"] += output_info["lesion_active_count"]
                new_epoch["hard_negatives"] += output_info["negative_count"]
                new_epoch["active_negative_dimensions"] += output_info["negative_active_dimension_count"]
                for class_id in range(4):
                    new_epoch[f"active_class_{class_id}"] += output_info["active_class_counts"][str(class_id)]

        record_gradients = not runtime.component_gradient_recorded and (
            (runtime.spec.group not in ("KQB", "KQR", "KQU"))
            or (runtime.spec.group == "KQB" and output_info["lesion_count"] + output_info["negative_count"] > 0)
            or (runtime.spec.group == "KQR" and output_info["valid_pair_count"] > 0)
            or (runtime.spec.group == "KQU" and output_info["lesion_active_count"] + output_info["negative_active_dimension_count"] > 0)
        )
        if record_gradients:
            detection_grad = _autograd_component_norms(
                detection_loss, module.named_parameters()
            )
            kd_grad = _autograd_component_norms(
                weighted_hidden, module.named_parameters()
            )
            output_grad = (
                _autograd_component_norms(weighted_output, module.named_parameters())
                if output_info else None
            )
            positive_grad = negative_grad = None
            if runtime.spec.group in ("KQB", "KQU"):
                positive_grad = _autograd_component_norms(
                    0.5 * runtime.spec.output_weight * output_info["positive_loss"],
                    module.named_parameters(),
                )
                negative_grad = _autograd_component_norms(
                    0.5 * runtime.spec.output_weight * output_info["negative_loss"],
                    module.named_parameters(),
                )
            runtime.kd_gradient_seen = (
                math.isfinite(kd_grad["all"])
                and kd_grad["all"] > 0
                and kd_grad["decoder"] > 0
                and kd_grad["encoder"] > 0
                and kd_grad["backbone"] > 0
            )
            runtime.output_gradient_seen = bool(
                output_grad is not None
                and math.isfinite(output_grad["all"])
                and output_grad["all"] > 0
                and output_grad["decoder"] > 0
            )
            runtime.component_gradient_recorded = True
        else:
            detection_grad = kd_grad = output_grad = positive_grad = negative_grad = None

        scaler.scale(total_loss / accumulation).backward()
        should_step = _optimizer_step_due(i, len(data_loader), accumulation)
        nonfinite = []
        total_grad = None
        old_scale = float(scaler.get_scale())
        if should_step:
            scaler.unscale_(optimizer)
            nonfinite = _nonfinite_gradients(module)
            total_grad = _grad_norm(module.parameters())
            if not nonfinite and math.isfinite(total_grad) and total_grad > 0:
                runtime.finite_unscaled_gradient_steps += 1
            if max_norm > 0 and not nonfinite:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            new_scale = float(scaler.get_scale())
            skipped = new_scale < old_scale
            runtime.skipped_optimizer_steps += int(skipped)
            runtime.optimizer_steps += int(not skipped)
            runtime.scaler_history.append({
                "step": int(i), "before": old_scale, "after": new_scale,
                "skipped": skipped, "nonfinite_count": len(nonfinite),
            })
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            if self_lr_scheduler:
                optimizer = lr_scheduler.step(cur_iters + i, optimizer)
            elif warmup is not None:
                warmup.step()

        diagnostic_due = (
            runtime.processed_steps == 1
            or global_step % max(1, int(print_freq)) == 0
            or (runtime.strict_debug_checks and should_step)
        )
        if diagnostic_due:
            payload = {
                "event": "kq_train_diagnostic",
                "group": runtime.spec.group,
                "epoch": int(epoch),
                "step": int(i),
                "layers": list(runtime.spec.layers),
                "weight": runtime.spec.weight,
                "output_weight": runtime.spec.output_weight,
                "detection_loss": float(detection_loss.detach()),
                "kd_loss": float(kd_info["loss"].detach()),
                "weighted_kd_loss": float(weighted_hidden.detach()),
                "output_kd_loss": float(output_info["loss"].detach()) if output_info else 0.0,
                "weighted_output_kd_loss": float(weighted_output.detach()),
                "total_loss": float(total_loss.detach()),
                "layer_kd": [float(value.detach()) for value in kd_info["layer_losses"]],
                "hidden": layer_query_audit(student_layers, teacher_layers),
                "alignment": teacher_audit["alignment"],
                "teacher_topk_calls": teacher_audit["topk_calls"],
                "memory_max_abs_delta": teacher_audit["memory_max_abs_delta"],
                "memory_mean_abs_delta": teacher_audit["memory_mean_abs_delta"],
                "suppressed_fraction": teacher_audit["suppressed_fraction"],
                "initial_reference_format": "sigmoid_pre_activation_with_inf_sentinels",
                "normal_query_count": runtime.spec.num_queries,
                "topk_shape": list(topk["indices"].shape),
                "topk_indices_head": topk["indices"][0, :16].detach().cpu().tolist(),
                "topk_coordinates_head": topk["anchors_unactivated"][0, :16].detach().float().cpu().tolist(),
                "component_gradient_norms_unscaled": {
                    "detection": detection_grad,
                    "hidden_kd_weighted": kd_grad,
                    "output_kd_weighted": output_grad,
                    "kqb_lesion_output_kd_weighted": positive_grad,
                    "kqb_hard_negative_output_kd_weighted": negative_grad,
                    "total": total_grad,
                },
                "nonfinite_gradient_count": len(nonfinite),
                "amp_scale_before": old_scale,
                "cuda_max_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
                "iteration_seconds": time.perf_counter() - step_started,
            }
            if runtime.spec.group == "KQB":
                payload["kqb_grouped_classification"] = {
                    "train_composed_image_ids": [_image_id_repr(target) for target in raw_targets],
                    "lesion_count": output_info["lesion_count"],
                    "hard_negative_count": output_info["negative_count"],
                    "empty_image_count": output_info["empty_image_count"],
                    "matched_class_counts": output_info["matched_class_counts"],
                    "low_score_lesion_count_below_0_5": output_info["low_score_lesion_count_below_0_5"],
                    "teacher_lower_gt_class_count": output_info["teacher_lower_gt_class_count"],
                    "teacher_minus_student_gt_class_probability": output_info["teacher_minus_student_gt_class_probability"],
                    "lesion_probability_samples": output_info["lesion_probability_samples"],
                    "lesion_indices_head": [indices[:8] for indices in output_info["lesion_indices"]],
                    "hard_negative_indices_head": [indices[:8] for indices in output_info["negative_indices"]],
                    "lesion_loss_raw": float(output_info["positive_loss"].detach()),
                    "hard_negative_loss_raw": float(output_info["negative_loss"].detach()),
                    "lesion_loss_weighted": float((0.5 * runtime.spec.output_weight * output_info["positive_loss"]).detach()),
                    "hard_negative_loss_weighted": float((0.5 * runtime.spec.output_weight * output_info["negative_loss"]).detach()),
                    "global_image_denominator": output_info["global_image_count"],
                }
            if runtime.spec.group in ("KQR", "KQU"):
                payload["new_output_objective"] = {
                    key: (float(value.detach()) if torch.is_tensor(value) else value)
                    for key, value in output_info.items()
                    if key != "loss"
                }
                payload["new_output_objective"]["train_composed_image_ids"] = [
                    _image_id_repr(target) for target in raw_targets
                ]
                if runtime.spec.group == "KQU":
                    payload["new_output_objective"].update({
                        "positive_loss_weighted": float((0.5 * runtime.spec.output_weight * output_info["positive_loss"]).detach()),
                        "negative_loss_weighted": float((0.5 * runtime.spec.output_weight * output_info["negative_loss"]).detach()),
                    })
            _write_diagnostic(runtime, payload)

        reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(reduced.values())
        if not math.isfinite(float(loss_value.detach())):
            raise FloatingPointError(f"non-finite loss: {reduced}")
        metric_logger.update(loss=loss_value, **reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar("Loss/total", loss_value.item(), global_step)
            writer.add_scalar("Loss/shared_query_kd_raw", kd_info["loss"].item(), global_step)
            if output_info:
                writer.add_scalar("Loss/shared_query_output_kd_raw", output_info["loss"].item(), global_step)
        completed_steps = i + 1
        if should_step and checkpoint_callback and checkpoint_interval and completed_steps % checkpoint_interval == 0:
            checkpoint_callback(epoch, completed_steps)

    teacher_parameter_delta = _maximum_delta(
        runtime.teacher_parameters_before, runtime.teacher.named_parameters()
    )
    teacher_buffer_delta = _maximum_delta(
        runtime.teacher_buffers_before, runtime.teacher.named_buffers()
    )
    if runtime.spec.group == "KQB":
        count_keys = list(grouped_epoch)
        counts = torch.tensor([grouped_epoch[key] for key in count_keys], device=device, dtype=torch.long)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(counts)
        grouped_epoch = {key: int(value) for key, value in zip(count_keys, counts.tolist())}
    if runtime.spec.group in ("KQR", "KQU"):
        count_keys = list(new_epoch)
        counts = torch.tensor([new_epoch[key] for key in count_keys], device=device, dtype=torch.long)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(counts)
        new_epoch = {key: int(value) for key, value in zip(count_keys, counts.tolist())}
    summary = {
        "event": "kq_epoch_runtime_summary",
        "group": runtime.spec.group,
        "processed_steps": runtime.processed_steps,
        "optimizer_steps": runtime.optimizer_steps,
        "skipped_optimizer_steps": runtime.skipped_optimizer_steps,
        "finite_unscaled_gradient_steps": runtime.finite_unscaled_gradient_steps,
        "kd_gradient_seen": runtime.kd_gradient_seen,
        "output_gradient_seen": runtime.output_gradient_seen,
        "teacher_parameter_max_delta": teacher_parameter_delta,
        "teacher_buffer_max_delta": teacher_buffer_delta,
        "teacher_grad_exists": any(p.grad is not None for p in runtime.teacher.parameters()),
        "scaler_history_tail": runtime.scaler_history[-32:],
        "kqb_epoch_global_group_counts": grouped_epoch if runtime.spec.group == "KQB" else None,
        "new_objective_epoch_global_counts": new_epoch if runtime.spec.group in ("KQR", "KQU") else None,
    }
    _write_diagnostic(runtime, summary)
    if runtime.strict_debug_checks:
        failures = []
        if runtime.optimizer_steps < 1:
            failures.append("no optimizer update after GradScaler adaptation")
        if runtime.finite_unscaled_gradient_steps < 1:
            failures.append("no finite nonzero unscaled Student gradient")
        if not runtime.kd_gradient_seen:
            failures.append("KD did not produce a finite nonzero Student gradient")
        if runtime.spec.output_weight > 0 and not runtime.output_gradient_seen:
            failures.append("output KD did not produce a finite nonzero decoder gradient")
        if teacher_parameter_delta != 0 or teacher_buffer_delta != 0:
            failures.append("teacher parameters or buffers changed")
        if summary["teacher_grad_exists"]:
            failures.append("teacher received gradients")
        if failures:
            raise RuntimeError("; ".join(failures))
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {key: meter.global_avg for key, meter in metric_logger.meters.items()}


class SharedQueryKDSolver(ECSolver):
    def __init__(self, cfg, args):
        super().__init__(cfg)
        self.kq_args = args

    def train(self):
        global ACTIVE_RUNTIME
        super().train()
        spec = resolve_kq_spec(self.kq_args.kq_group)
        if not spec.enabled:
            raise RuntimeError("KQ0 must use the unchanged ecdetseg/train.py entrypoint")
        validate_kq_config(self.cfg.yaml_cfg.get("shared_query_kd", {}), spec)
        student = _module(self.model)
        teacher = _load_frozen_teacher(
            student, self.kq_args.teacher_checkpoint, self.device
        )
        optimizer_ids = {
            id(parameter) for group in self.optimizer.param_groups for parameter in group["params"]
        }
        teacher_ids = {id(parameter) for parameter in teacher.parameters()}
        ema_ids = {id(parameter) for parameter in self.ema.module.parameters()} if self.ema else set()
        if optimizer_ids & teacher_ids or ema_ids & teacher_ids:
            raise RuntimeError("teacher leaked into Student optimizer or EMA")
        output_dir = Path(self.cfg.output_dir)
        ACTIVE_RUNTIME = KQRuntime(
            teacher=teacher,
            spec=spec,
            max_train_steps=self.kq_args.max_train_steps,
            strict_debug_checks=self.kq_args.strict_debug_checks,
            teacher_parameters_before=_snapshot_named_tensors(teacher.named_parameters()),
            teacher_buffers_before=_snapshot_named_tensors(teacher.named_buffers()),
            diagnostics_path=output_dir / f"kq_diagnostics_rank{_rank()}.jsonl",
        )
        _write_diagnostic(ACTIVE_RUNTIME, {
            "event": "kq_runtime",
            "group": spec.group,
            "teacher_memory_mode": spec.teacher_memory_mode,
            "teacher_memory_source": "frozen teacher backbone/HybridEncoder output",
            "teacher_query_source": "Student initial query/reference",
            "teacher_topk_source": "Student",
            "memory_suppression_location": "after HybridEncoder before ECTransformer projection/BN",
            "layers": list(spec.layers),
            "weight": spec.weight,
            "output_weight": spec.output_weight,
            "teacher_trainable_parameter_count": sum(int(p.requires_grad) for p in teacher.parameters()),
            "teacher_in_optimizer": bool(optimizer_ids & teacher_ids),
            "teacher_in_ema": bool(ema_ids & teacher_ids),
        })


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume")
    parser.add_argument("-t", "--tuning")
    parser.add_argument("-d", "--device")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--summary-dir")
    parser.add_argument("-u", "--update", nargs="+")
    parser.add_argument("--print-method", default="builtin")
    parser.add_argument("--print-rank", type=int, default=0)
    parser.add_argument("--local-rank", type=int)
    parser.add_argument(
        "--kq-group", choices=("KQ1", "KQ2", "KQ3", "KQW", "KQO", "KQB", "KQR", "KQU"), required=True
    )
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--strict-debug-checks", action="store_true")
    parser.add_argument("--debug-only", action="store_true")
    return parser.parse_args()


def main(args):
    mp.set_sharing_strategy(os.environ.get("EC_MP_SHARING_STRATEGY", "file_system"))
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)
    if args.tuning and args.resume:
        raise ValueError("tuning and resume are mutually exclusive")
    update = yaml_utils.parse_cli(args.update)
    custom = {
        "update", "kq_group", "teacher_checkpoint", "max_train_steps",
        "strict_debug_checks", "debug_only",
    }
    update.update({key: value for key, value in vars(args).items() if key not in custom and value is not None})
    cfg = YAMLConfig(args.config, **update)
    if args.resume or args.tuning:
        for backbone_name in ("ViTAdapter", "DinoV2Adapter"):
            if backbone_name in cfg.yaml_cfg:
                cfg.yaml_cfg[backbone_name]["skip_load_backbone"] = True
    ec_solver_module.train_one_epoch = train_one_epoch_kq
    solver = SharedQueryKDSolver(cfg, args)
    if args.debug_only:
        if args.max_train_steps <= 0:
            raise ValueError("--debug-only requires a positive --max-train-steps")
        solver.train()
        solver.train_dataloader.set_epoch(0)
        if dist_utils.is_dist_available_and_initialized():
            solver.train_dataloader.sampler.set_epoch(0)
        stats = train_one_epoch_kq(
            False,
            solver.lr_scheduler,
            solver.model,
            solver.criterion,
            solver.train_dataloader,
            solver.optimizer,
            solver.device,
            0,
            max_norm=solver.cfg.clip_max_norm,
            print_freq=solver.cfg.print_freq,
            ema=solver.ema,
            scaler=solver.scaler,
            lr_warmup_scheduler=solver.lr_warmup_scheduler,
            gradient_accumulation_steps=solver.cfg.gradient_accumulation_steps,
            writer=solver.writer,
        )
        state = solver.state_dict()
        if any("teacher" in key.lower() for key in state):
            raise RuntimeError(f"teacher leaked into checkpoint keys: {list(state)}")
        debug_path = Path(solver.cfg.output_dir) / "debug_checkpoint.pth"
        dist_utils.save_on_master(state, debug_path)
        if dist_utils.is_dist_available_and_initialized():
            torch.distributed.barrier()
        if dist_utils.is_main_process():
            (Path(solver.cfg.output_dir) / "DEBUG_COMPLETED.json").write_text(
                json.dumps({
                    "group": args.kq_group,
                    "stats": stats,
                    "checkpoint": str(debug_path),
                    "checkpoint_keys": sorted(state),
                }, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
    else:
        solver.fit()
    dist_utils.cleanup()


if __name__ == "__main__":
    main(parse_args())
