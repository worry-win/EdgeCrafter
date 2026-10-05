"""Bounded real-augmentation update smoke for the cmp5L QG route."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ecdetseg"))
sys.path.insert(0, str(ROOT))

from engine.core import YAMLConfig  # noqa: E402
from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy  # noqa: E402
from engine.misc import dist_utils  # noqa: E402
from engine.solver import TASKS  # noqa: E402
from scripts.ablation.cmp5L_privileged_decision_kd import (  # noqa: E402
    run_with_all_score_heads,
)
from scripts.ablation.cmp5L_qg_behavior_distillation import (  # noqa: E402
    attach_query_gate_head,
    balanced_selection_bce,
    bernoulli_behavior_kd,
    compose_qg_loss,
    match_competitive_candidates,
    max_named_buffer_change,
    slice_normal_query_layers,
    temporary_query_gate_state,
)
from scripts.ablation.cmp5L_qg_stage1_smoke import (  # noqa: E402
    finite_tree,
    max_prediction_diff,
)
from scripts.ablation.evaluate_cmp5L_af_validation import (  # noqa: E402
    normal_bg_dependency,
    region_mask_with_outside,
)
from scripts.ablation.evaluate_cmp5L_gh_validation import (  # noqa: E402
    _reference_audit,
    run_explicit_gate_branch,
)
from scripts.ablation.probe_cmp5L_decoder_internal_tensors import (  # noqa: E402
    DecoderInternalCapture,
)
from scripts.ablation.probe_cmp5L_qg_stage1_full import (  # noqa: E402
    matcher_indices,
    positive_pairs,
)


def _jsonable(value):
    if isinstance(value, Counter):
        return {str(key): int(count) for key, count in value.items()}
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _state_snapshot(module):
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _max_state_change(before, module):
    maximum = 0.0
    after = module.state_dict()
    if set(before) != set(after):
        raise RuntimeError("state keys changed during audit")
    for name, reference in before.items():
        value = after[name].detach().cpu()
        if value.dtype == torch.bool:
            difference = float(torch.logical_xor(value, reference).any())
        elif value.numel():
            difference = float((value - reference).abs().max())
        else:
            difference = 0.0
        maximum = max(maximum, difference)
    return maximum


def _restore_buffers(module, snapshot):
    with torch.no_grad():
        for name, value in module.named_buffers():
            value.copy_(snapshot[name])


def _rng_snapshot(device):
    state = {"cpu": torch.random.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def _restore_rng(state, device):
    torch.random.set_rng_state(state["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def _dependency_gate(capture, targets, normal_queries):
    gt_xyxy = [box_cxcywh_to_xyxy(target["boxes"]) for target in targets]
    attention, regions = [], []
    for layer_index in range(3):
        record = capture.layers[layer_index]
        locations = record["sampling_locations"][:, -normal_queries:]
        weights = record["attention_weights"][:, -normal_queries:]
        attention.append(weights)
        regions.append(torch.stack([
            region_mask_with_outside(locations[batch], gt_xyxy[batch])[0]
            for batch in range(len(targets))
        ]))
    dependency = normal_bg_dependency(attention, regions)
    has_gt = torch.tensor(
        [bool(target["labels"].numel()) for target in targets],
        dtype=torch.bool,
        device=dependency.device,
    )
    gate = (dependency > 0.20) & has_gt[:, None]
    return dependency, gate, gt_xyxy


def _grad_norm(parameters):
    terms = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return float(torch.stack(terms).sum().sqrt()) if terms else 0.0


def _component_gradient_norm(loss, gate_parameters, class_parameters):
    parameters = gate_parameters + class_parameters
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )

    def norm(values):
        terms = [value.detach().float().square().sum() for value in values if value is not None]
        return float(torch.stack(terms).sum().sqrt()) if terms else 0.0

    gate_count = len(gate_parameters)
    return {
        "gate": norm(gradients[:gate_count]),
        "class_head": norm(gradients[gate_count:]),
    }


def _selection_confusion(probability, labels):
    prediction = probability.detach() >= 0.5
    labels = labels.to(torch.bool)
    tp = int((prediction & labels).sum())
    fp = int((prediction & ~labels).sum())
    fn = int((~prediction & labels).sum())
    tn = int((~prediction & ~labels).sum())
    return {
        "suppress_tp": tp,
        "false_suppression": fp,
        "missed_suppression": fn,
        "protect_tn": tn,
        "suppress_precision": tp / max(1, tp + fp),
        "suppress_recall": tp / max(1, tp + fn),
    }


def _probability_stats(probability):
    coefficient = 1.0 - 0.8 * probability
    return {
        "p_min": float(probability.min()),
        "p_mean": float(probability.mean()),
        "p_max": float(probability.max()),
        "c_min": float(coefficient.min()),
        "c_mean": float(coefficient.mean()),
        "c_max": float(coefficient.max()),
    }


def _build(args, output_dir):
    cfg = YAMLConfig(
        args.config,
        **{
            "device": str(args.device),
            "tuning": args.checkpoint,
            "output_dir": str(output_dir / "solver"),
            "use_amp": True,
            "train_dataloader": {
                "num_workers": 0,
                "total_batch_size": args.batch_size,
            },
            "num_classes": 4,
            "remap_mscoco_category": False,
        },
    )
    for name in ("ViTAdapter", "DinoV2Adapter"):
        if name in cfg.yaml_cfg:
            cfg.yaml_cfg[name]["skip_load_backbone"] = True
    attach_query_gate_head(
        cfg.model,
        hidden_dim=args.head_hidden,
        prior_probability=0.01,
    )
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.train()
    student = dist_utils.de_parallel(solver.model)
    student.decoder.decoder.query_gate_enabled = True
    return cfg, solver, student


def _class_parameters(student):
    modules = [student.decoder.dec_score_head]
    lqe_layers = getattr(student.decoder.decoder, "lqe_layers", None)
    if lqe_layers is not None:
        modules.append(lqe_layers)
    seen, parameters = set(), []
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
    return parameters


def run(args, output_dir):
    started = time.time()
    cfg, solver, student = _build(args, output_dir)
    gate_module = student.decoder.decoder
    gate_head = gate_module.query_gate_head
    gate_parameters = list(gate_head.parameters())
    class_parameters = _class_parameters(student)
    detector_parameters = [
        parameter
        for name, parameter in student.named_parameters()
        if "query_gate_head" not in name
    ]
    optimizer_ids = {
        id(parameter)
        for group in solver.optimizer.param_groups
        for parameter in group["params"]
    }
    optimizer_covers_gate = all(id(parameter) in optimizer_ids for parameter in gate_parameters)
    ema_keys = list(solver.ema.module.state_dict()) if solver.ema is not None else []
    ema_gate_key_count = sum("query_gate_head" in key for key in ema_keys)
    if not optimizer_covers_gate or not ema_gate_key_count:
        raise RuntimeError("gate head missing from optimizer or EMA")

    teacher_full = copy.deepcopy(student).eval().requires_grad_(False)
    teacher_full.decoder.decoder.query_gate_enabled = False
    teacher_decoder = copy.deepcopy(teacher_full.decoder).eval().requires_grad_(False)
    teacher_full_before = _state_snapshot(teacher_full)
    teacher_decoder_before = _state_snapshot(teacher_decoder)
    gate_before = _state_snapshot(gate_head)
    oracle_audit = _reference_audit()
    records = []
    label_class_counts = {str(index): Counter() for index in range(4)}
    first_component_gradients = None
    first_gate_off_parity = None
    last_samples = None
    last_targets = None
    normal_queries = int(args.normal_queries)
    solver.optimizer.zero_grad(set_to_none=True)
    solver.model.train()
    solver.criterion.train()
    if hasattr(solver.train_dataloader, "set_epoch"):
        solver.train_dataloader.set_epoch(0)

    for step, (samples, raw_targets) in enumerate(solver.train_dataloader):
        if step >= args.steps:
            break
        samples = samples.to(args.device)
        targets = [
            {key: value.to(args.device) for key, value in target.items()}
            for target in raw_targets
        ]
        if not all(
            bool(((target["boxes"] >= 0) & (target["boxes"] <= 1)).all())
            for target in targets
        ):
            raise RuntimeError("augmented train boxes are not normalized cxcywh")
        if first_gate_off_parity is None:
            was_training = student.training
            student.eval()
            with torch.no_grad(), temporary_query_gate_state(gate_module, False):
                disabled = student(samples)
                baseline = teacher_full(samples)
            first_gate_off_parity = max_prediction_diff(disabled, baseline)
            student.train(was_training)

        with torch.no_grad():
            with DecoderInternalCapture(teacher_full.decoder) as teacher_capture:
                _, _ = run_with_all_score_heads(
                    teacher_full.decoder,
                    lambda: teacher_full(samples),
                )
            _, teacher_gate, gt_xyxy = _dependency_gate(
                teacher_capture, targets, normal_queries
            )
            privileged, privileged_layers = run_with_all_score_heads(
                teacher_decoder,
                lambda: run_explicit_gate_branch(
                    teacher_decoder,
                    teacher_capture.decoder_input,
                    "E",
                    teacher_gate,
                    gt_xyxy,
                    list(range(len(targets))),
                    set(),
                    oracle_audit,
                )[0],
            )

        rng = _rng_snapshot(args.device)
        student_buffers = {
            name: value.detach().clone()
            for name, value in student.named_buffers()
        }
        with torch.no_grad(), temporary_query_gate_state(gate_module, False):
            with DecoderInternalCapture(student.decoder) as label_capture:
                student(samples, targets=targets)
        _, selection_labels, _ = _dependency_gate(
            label_capture, targets, normal_queries
        )
        _restore_buffers(student, student_buffers)
        _restore_rng(rng, args.device)

        autocast_enabled = solver.scaler is not None
        with torch.autocast(
            device_type=args.device.type,
            enabled=autocast_enabled,
            cache_enabled=True,
        ):
            with temporary_query_gate_state(gate_module, True):
                with DecoderInternalCapture(student.decoder) as gated_capture:
                    outputs, captured_layers = run_with_all_score_heads(
                        student.decoder,
                        lambda: solver.model(samples, targets=targets),
                    )
        student_layers = slice_normal_query_layers(
            captured_layers,
            normal_queries=normal_queries,
        )
        if not all(finite_tree(value) for value in (outputs, privileged, student_layers)):
            raise RuntimeError("non-finite output during bounded update")
        query_diff = float((
            label_capture.decoder_input["query_content"][:, -normal_queries:]
            - gated_capture.decoder_input["query_content"][:, -normal_queries:]
        ).abs().max())
        reference_diff = float((
            label_capture.decoder_input["reference_unact"][:, -normal_queries:]
            - gated_capture.decoder_input["reference_unact"][:, -normal_queries:]
        ).abs().max())
        if query_diff != 0.0 or reference_diff != 0.0:
            raise RuntimeError(
                f"selection replay identity mismatch: query={query_diff}, reference={reference_diff}"
            )
        gate_logits = gate_module.last_query_gate_logits
        gate_probability = gate_module.last_query_gate_probability
        if gate_logits is None or gate_probability is None:
            raise RuntimeError("student gate output was not captured")
        if gate_logits.shape != selection_labels.shape:
            raise RuntimeError("selection labels do not align with normal-query gate logits")

        with torch.autocast(device_type=args.device.type, enabled=False):
            loss_dict = solver.criterion(
                outputs,
                targets,
                epoch=0,
                step=step,
                global_step=step,
                epoch_step=len(solver.train_dataloader),
            )
            detection_loss = sum(loss_dict.values())
            selection = balanced_selection_bce(
                gate_logits.float(),
                selection_labels.float(),
                torch.ones_like(selection_labels, dtype=torch.bool),
            )
            student_matches = matcher_indices(solver.criterion, outputs, targets)
            teacher_matches = matcher_indices(solver.criterion, privileged, targets)
            positives, positive_class_counts = positive_pairs(
                student_matches, teacher_matches, targets
            )
            competition = match_competitive_candidates(
                outputs["pred_logits"].float(),
                outputs["pred_boxes"].float(),
                privileged["pred_logits"].float(),
                privileged["pred_boxes"].float(),
                positive_pairs=positives,
                top_k=20,
                score_threshold=0.05,
                min_iou=0.30,
            )
            behavior = bernoulli_behavior_kd(
                [layer.float() for layer in student_layers],
                [layer.float() for layer in privileged_layers],
                positive_pairs=positives,
                competitive_pairs=competition["pairs"],
            )
            composed = compose_qg_loss(
                detection_loss,
                selection["loss"],
                behavior["loss"],
                group="QG3",
            )

        positive_kd = torch.stack(behavior["positive_loss_per_layer"]).mean()
        competitive_kd = torch.stack(behavior["competitive_loss_per_layer"]).mean()
        if first_component_gradients is None:
            first_component_gradients = {
                "detection": _component_gradient_norm(
                    detection_loss, gate_parameters, class_parameters
                ),
                "selection": _component_gradient_norm(
                    selection["loss"], gate_parameters, class_parameters
                ),
                "positive_behavior": _component_gradient_norm(
                    positive_kd, gate_parameters, class_parameters
                ),
                "competitive_behavior": _component_gradient_norm(
                    competitive_kd, gate_parameters, class_parameters
                ),
            }

        if solver.scaler is not None:
            solver.scaler.scale(composed["total"]).backward()
            solver.scaler.unscale_(solver.optimizer)
        else:
            composed["total"].backward()
        gate_grad_norm = _grad_norm(gate_parameters)
        detector_grad_norm = _grad_norm(detector_parameters)
        nonfinite = [
            name
            for name, parameter in student.named_parameters()
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ]
        if nonfinite or not (
            math.isfinite(gate_grad_norm)
            and gate_grad_norm > 0
            and math.isfinite(detector_grad_norm)
            and detector_grad_norm > 0
        ):
            raise RuntimeError(
                f"invalid unscaled gradients: gate={gate_grad_norm}, "
                f"detector={detector_grad_norm}, nonfinite={nonfinite[:8]}"
            )
        if solver.scaler is not None:
            solver.scaler.step(solver.optimizer)
            solver.scaler.update()
        else:
            solver.optimizer.step()
        solver.optimizer.zero_grad(set_to_none=True)
        if solver.ema is not None:
            solver.ema.update(solver.model)

        confusion = _selection_confusion(gate_probability, selection_labels)
        for batch, target in enumerate(targets):
            student_query, student_gt = student_matches[batch]
            for query, gt_index in zip(student_query.tolist(), student_gt.tolist()):
                class_index = str(int(target["labels"][gt_index]))
                truth = "suppress" if bool(selection_labels[batch, query]) else "protect"
                prediction = "suppress" if float(gate_probability[batch, query]) >= 0.5 else "protect"
                label_class_counts[class_index][f"gt_{truth}"] += 1
                label_class_counts[class_index][f"pred_{prediction}"] += 1

        records.append({
            "step": step,
            "images": len(targets),
            "gt": sum(int(target["labels"].numel()) for target in targets),
            "mixup_images": sum("mixup" in target for target in targets),
            "query_identity_max_abs_diff": query_diff,
            "reference_identity_max_abs_diff": reference_diff,
            "selection": {
                **_probability_stats(gate_probability.detach()),
                **confusion,
                "protect_count": selection["protect_count"],
                "suppress_count": selection["suppress_count"],
                "protect_loss": float(selection["protect_loss"].detach()),
                "suppress_loss": float(selection["suppress_loss"].detach()),
            },
            "pairs": {
                "positive": int(positives.shape[0]),
                "positive_by_class": dict(positive_class_counts),
                "competitive": int(competition["pairs"].shape[0]),
            },
            "loss": {
                "detection": float(detection_loss.detach()),
                "selection": float(selection["loss"].detach()),
                "behavior": float(behavior["loss"].detach()),
                "positive_behavior": float(positive_kd.detach()),
                "competitive_behavior": float(competitive_kd.detach()),
                "total": float(composed["total"].detach()),
            },
            "unscaled_gradient_norm": {
                "gate": gate_grad_norm,
                "detector_without_gate": detector_grad_norm,
            },
        })
        last_samples = samples.detach().clone()
        last_targets = targets

    if len(records) != args.steps:
        raise RuntimeError(f"processed {len(records)} steps, expected {args.steps}")
    if max(first_gate_off_parity.values()) > 2e-6:
        raise RuntimeError(f"initial gate-off parity failed: {first_gate_off_parity}")
    if _max_state_change(gate_before, gate_head) <= 0.0:
        raise RuntimeError("gate head parameters did not update")

    student.eval()
    with torch.no_grad(), temporary_query_gate_state(gate_module, True):
        current_on = student(last_samples)
        current_probability = gate_module.last_query_gate_probability.detach().clone()
    with torch.no_grad(), temporary_query_gate_state(gate_module, False):
        current_off = student(last_samples)
    gate_on_off = max_prediction_diff(current_on, current_off)
    if max(gate_on_off.values()) <= 0.0:
        raise RuntimeError("updated gate has no effect on detector output")

    checkpoint = {
        "model": student.state_dict(),
        "ema": solver.ema.state_dict() if solver.ema is not None else None,
        "optimizer": solver.optimizer.state_dict(),
        "scaler": solver.scaler.state_dict() if solver.scaler is not None else None,
        "qg_protocol": {
            "group": "QG3-smoke",
            "lambda_select": 1.0,
            "lambda_behavior": 0.5,
            "normal_queries": normal_queries,
        },
    }
    checkpoint_path = output_dir / "checkpoint.pth"
    torch.save(checkpoint, checkpoint_path)
    loaded_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    restored = copy.deepcopy(student).to(args.device).eval()
    restored.load_state_dict(loaded_state["ema"]["module"], strict=True)
    restored.decoder.decoder.query_gate_enabled = True
    ema_module = solver.ema.module.eval()
    ema_module.decoder.decoder.query_gate_enabled = True
    with torch.no_grad():
        ema_output = ema_module(last_samples)
        restored_output = restored(last_samples)
    ema_reload_parity = max_prediction_diff(ema_output, restored_output)
    if max(ema_reload_parity.values()) > 2e-6:
        raise RuntimeError(f"EMA strict reload parity failed: {ema_reload_parity}")

    teacher_gradient_count = sum(
        parameter.grad is not None
        for module in (teacher_full, teacher_decoder)
        for parameter in module.parameters()
    )
    teacher_state_change = max(
        _max_state_change(teacher_full_before, teacher_full),
        _max_state_change(teacher_decoder_before, teacher_decoder),
    )
    if teacher_gradient_count or teacher_state_change != 0.0:
        raise RuntimeError(
            f"teacher mutation: gradients={teacher_gradient_count}, "
            f"state_change={teacher_state_change}"
        )
    if (
        oracle_audit["location_input_mutation_max_abs"] != 0.0
        or oracle_audit["attention_input_mutation_max_abs"] != 0.0
    ):
        raise RuntimeError(f"Oracle replay mutated routing inputs: {oracle_audit}")

    result = {
        "meta": {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "group": "QG3-smoke",
            "steps": args.steps,
            "batch_size": args.batch_size,
            "uses_real_train_augmentation": True,
            "normal_queries": normal_queries,
            "selection_label_source": "same-current-student ungated deterministic RNG replay",
            "teacher_p": "baseline EMA Normal d_BG>0.20; E all-point c=0.2 on L0-L2",
        },
        "lifecycle": {
            "optimizer_covers_gate": optimizer_covers_gate,
            "ema_gate_key_count": ema_gate_key_count,
            "strict_ema_reload": True,
            "ema_reload_parity": ema_reload_parity,
            "inference_call_uses_samples_only": True,
        },
        "checks": {
            "initial_gate_off_parity": first_gate_off_parity,
            "gate_parameter_max_abs_change": _max_state_change(gate_before, gate_head),
            "current_gate_on_off": gate_on_off,
            "current_gate_distribution": _probability_stats(current_probability),
            "teacher_gradient_count": teacher_gradient_count,
            "teacher_state_max_abs_change": teacher_state_change,
            "oracle_location_input_mutation": oracle_audit["location_input_mutation_max_abs"],
            "oracle_attention_input_mutation": oracle_audit["attention_input_mutation_max_abs"],
            "finite": finite_tree((current_on, current_off, ema_output, restored_output)),
        },
        "component_gradient_norms_first_step": first_component_gradients,
        "gt_matched_selection_by_class": label_class_counts,
        "records": records,
        "resources": {
            "elapsed_seconds": time.time() - started,
            "cuda_max_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
        },
    }
    return _jsonable(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--head-hidden", type=int, default=64)
    parser.add_argument("--normal-queries", type=int, default=300)
    args = parser.parse_args()
    args.device = torch.device(args.device)
    if args.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        result = run(args, output_dir)
    except Exception as error:
        (output_dir / "FAILED.json").write_text(json.dumps({
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }, indent=2) + "\n")
        raise
    (output_dir / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (output_dir / "COMPLETED").write_text("QG bounded update smoke complete\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
