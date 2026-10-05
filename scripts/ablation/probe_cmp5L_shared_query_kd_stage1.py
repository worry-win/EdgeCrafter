"""Stage-1 technical gates for shared-query KQ distillation."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from types import MethodType

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ecdetseg"))
sys.path.insert(0, str(ROOT))

from engine.core import YAMLConfig  # noqa: E402
from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy  # noqa: E402
from engine.misc import dist_utils  # noqa: E402
from engine.solver import TASKS  # noqa: E402
from scripts.ablation.cmp5L_query_behavior_kd import (  # noqa: E402
    DecoderReplayInputs,
    _predictions_from_decoder_outputs,
    _run_decoder_with_capture,
    run_model_with_capture,
)
from scripts.ablation.probe_cmp5L_decoder_internal_tensors import (  # noqa: E402
    _build,
)
from scripts.ablation.probe_cmp5L_privileged_ema_oracle import (  # noqa: E402
    privilege_features,
)
from scripts.ablation.cmp5L_shared_query_kd import (  # noqa: E402
    SharedQueryReplay,
    layer_query_audit,
    normal_query_slice,
    shared_query_alignment_audit,
)


def _finite_tree(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return True


def _state_snapshot(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _max_state_delta(before, module):
    maximum = 0.0
    after = module.state_dict()
    if set(before) != set(after):
        raise RuntimeError("teacher state keys changed")
    for name, reference in before.items():
        value = after[name].detach()
        if reference.dtype == torch.bool:
            delta = float(torch.logical_xor(reference, value).any())
        elif reference.numel():
            delta = float((reference - value).abs().max())
        else:
            delta = 0.0
        maximum = max(maximum, delta)
    return maximum


def _capture_topk_indices(ec_transformer):
    """Capture student _select_topk without changing its return contract."""
    original = ec_transformer._select_topk
    holder = {"indices": None, "anchors": None}

    def wrapped(memory, outputs_logits, outputs_anchors_unact, topk):
        if ec_transformer.query_select_method == "default":
            _, indices = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)
        elif ec_transformer.query_select_method == "one2many":
            _, indices = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            indices = indices // ec_transformer.num_classes
        elif ec_transformer.query_select_method == "agnostic":
            _, indices = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)
        else:
            raise RuntimeError("unknown query selection method")
        holder["indices"] = indices.detach().clone()
        holder["anchors"] = outputs_anchors_unact.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, outputs_anchors_unact.shape[-1])
        ).detach().clone()
        return original(memory, outputs_logits, outputs_anchors_unact, topk)

    ec_transformer._select_topk = wrapped
    return original, holder


def _restore_topk(ec_transformer, original):
    ec_transformer._select_topk = original


def _train_targets_to_absolute(raw_targets, image_hw, device):
    height, width = image_hw
    scale = torch.tensor([width, height, width, height], device=device)
    result = []
    for target in raw_targets:
        boxes = target["boxes"].as_subclass(torch.Tensor).to(device)
        if boxes.numel() and (float(boxes.min()) < -1e-6 or float(boxes.max()) > 1.000001):
            raise RuntimeError("train loader boxes are not normalized cxcywh")
        xyxy = box_cxcywh_to_xyxy(boxes) * scale
        result.append({"boxes": xyxy, "labels": target["labels"].to(device)})
    return result


def _memory_stats(normal, privileged, masks, empty):
    difference = (privileged - normal).abs()
    stats = {
        "normal_shape": list(normal.shape),
        "privileged_shape": list(privileged.shape),
        "max_abs_delta": float(difference.max()) if difference.numel() else 0.0,
        "mean_abs_delta": float(difference.mean()) if difference.numel() else 0.0,
        "mask_min": float(masks.min()),
        "mask_max": float(masks.max()),
        "suppressed_fraction": float((masks < 0.999999).float().mean()),
    }
    if bool(empty.any()):
        empty_delta = difference[empty]
        stats["empty_identity_max_abs"] = float(empty_delta.max()) if empty_delta.numel() else 0.0
        stats["empty_identity_count"] = int(empty.sum())
    else:
        stats["empty_identity_max_abs"] = None
        stats["empty_identity_count"] = 0
    return stats


def _run_direct_teacher(teacher, replay, memory, shapes):
    direct = SharedQueryReplay(
        initial_query=replay.initial_query.detach(),
        initial_reference_unactivated=replay.initial_reference_unactivated.detach(),
        topk_indices=replay.topk_indices.detach(),
        memory=memory,
        spatial_shapes=shapes,
        attention_mask=replay.attention_mask,
        denoising_metadata=None,
        normal_query_count=replay.normal_query_count,
    )
    decoder_replay = DecoderReplayInputs(
        initial_query=direct.initial_query,
        initial_reference_unactivated=direct.initial_reference_unactivated,
        memory=direct.memory,
        spatial_shapes=direct.spatial_shapes,
        attention_mask=None,
        denoising_metadata=None,
        normal_query_count=direct.normal_query_count,
    )
    original_select = teacher.decoder._select_topk
    calls = {"count": 0}

    def forbidden(*_args, **_kwargs):
        calls["count"] += 1
        raise RuntimeError("teacher attempted to reselect Top-K queries")

    teacher.decoder._select_topk = forbidden
    try:
        outputs, trace = _run_decoder_with_capture(teacher.decoder, decoder_replay)
    finally:
        teacher.decoder._select_topk = original_select
    return direct, outputs, trace, calls["count"]


def _build_train(args):
    cfg = YAMLConfig(args.config, **{
        "train_dataloader": {
            "dataset": {"ann_file": args.ann_file, "img_folder": args.img_folder},
            "num_workers": args.num_workers,
            "total_batch_size": args.batch_size,
            "shuffle": False,
            "drop_last": False,
        },
        "val_dataloader": {
            "num_workers": 0,
            "total_batch_size": 1,
        },
        "num_classes": 4,
        "remap_mscoco_category": False,
    })
    for name in ("ViTAdapter", "DinoV2Adapter"):
        if name in cfg.yaml_cfg:
            cfg.yaml_cfg[name]["skip_load_backbone"] = True
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver._setup()
    solver.load_resume_state(args.checkpoint)
    model = solver.ema.module if args.weights == "ema" and solver.ema is not None else solver.model
    model = dist_utils.de_parallel(model).to(args.device).eval()
    return cfg, solver, model


def _stats_tensor(value):
    value = value.detach().float()
    finite = value[torch.isfinite(value)]
    return {
        "shape": list(value.shape),
        "finite_count": int(finite.numel()),
        "nonfinite_count": int((~torch.isfinite(value)).sum()),
        "mean": float(finite.mean()) if finite.numel() else None,
        "std": float(finite.std(unbiased=False)) if finite.numel() else None,
        "min": float(finite.min()) if finite.numel() else None,
        "max": float(finite.max()) if finite.numel() else None,
    }


def _image_id_repr(target):
    value = target.get("image_id")
    if torch.is_tensor(value):
        return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def run(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    cfg, solver, student = _build_train(args)
    student.eval().requires_grad_(False)
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    identity_teacher = copy.deepcopy(student).eval().requires_grad_(False)
    student_decoder = student.decoder
    teacher_decoder = teacher.decoder
    identity_decoder = identity_teacher.decoder
    teacher_before = _state_snapshot(teacher)
    identity_before = _state_snapshot(identity_teacher)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    audit = {
        "meta": {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "ann_file": args.ann_file,
            "images_requested": args.limit,
            "teacher_memory_mode": "privileged_np",
            "teacher_memory_source": "frozen teacher backbone/encoder output",
            "teacher_query_source": "student initial query/reference",
            "teacher_topk_source": "student",
            "memory_suppression_location": "after teacher HybridEncoder, before ECTransformer input projection/BN",
            "memory_suppression_ratio": 0.2,
            "normal_query_count": student.decoder.num_queries,
        },
        "checks": {
            "cuda": device.type == "cuda",
            "teacher_requires_grad_count": sum(int(p.requires_grad) for p in teacher.parameters()),
            "teacher_in_student_optimizer": False,
            "teacher_in_student_ema": False,
            "teacher_topk_calls": 0,
            "finite": True,
        },
        "identity_layers": None,
        "batches": [],
        "empty_probe": [],
    }
    seen = 0
    first_fixed = None
    loader = cfg.val_dataloader
    # The supplied ann-file is the fixed train diagnostic subset; use its real train transforms.
    loader = cfg.train_dataloader
    if hasattr(loader, "set_epoch"):
        loader.set_epoch(0)
    with torch.no_grad():
        for batch_index, (samples, raw_targets) in enumerate(loader):
            if seen >= args.limit:
                break
            samples = samples.to(device)
            take = min(len(raw_targets), args.limit - seen)
            if take != len(raw_targets):
                samples = samples[:take]
                raw_targets = raw_targets[:take]
            absolute_targets = _train_targets_to_absolute(
                raw_targets, samples.shape[-2:], device
            )
            topk_original, topk_holder = _capture_topk_indices(student_decoder)
            try:
                student_outputs, student_trace, replay_inputs = run_model_with_capture(
                    student, samples, None
                )
            finally:
                _restore_topk(student_decoder, topk_original)
            if topk_holder["indices"] is None:
                raise RuntimeError("student top-k indices were not captured")
            replay = SharedQueryReplay(
                initial_query=replay_inputs.initial_query[:, -student.decoder.num_queries:],
                initial_reference_unactivated=replay_inputs.initial_reference_unactivated[:, -student.decoder.num_queries:],
                topk_indices=topk_holder["indices"],
                memory=replay_inputs.memory,
                spatial_shapes=replay_inputs.spatial_shapes,
                normal_query_count=student.decoder.num_queries,
            )

            # Same-weight identity: identical memory and exact student q0/r0 must match.
            _, identity_outputs, identity_trace, identity_topk_calls = _run_direct_teacher(
                identity_teacher, replay, replay.memory, replay.spatial_shapes
            )
            if identity_topk_calls:
                raise RuntimeError("identity teacher attempted Top-K selection")
            identity_layers = [normal_query_slice(layer, replay.normal_query_count) for layer in student_trace["query_outputs"]]
            identity_teacher_layers = [normal_query_slice(layer, replay.normal_query_count) for layer in identity_trace["query_outputs"]]
            identity_rows = layer_query_audit(identity_layers, identity_teacher_layers)
            if max(row["max_abs"] for row in identity_rows) > 2e-6:
                raise RuntimeError(f"same-weight identity failed: {identity_rows}")

            # Teacher owns its own normal/privileged encoder memory.
            teacher_backbone = teacher.backbone(samples)
            teacher_encoded = teacher.encoder([feature.clone() for feature in teacher_backbone])
            teacher_memory_normal, teacher_shapes = teacher.decoder._get_encoder_input(teacher_encoded)
            teacher_encoded_privileged, masks = privilege_features(
                teacher_encoded,
                absolute_targets,
                samples.shape[-2:],
                background_weight=0.2,
                context_weight=None,
                context_scale=1.5,
            )
            teacher_memory_privileged, privileged_shapes = teacher.decoder._get_encoder_input(teacher_encoded_privileged)
            if tuple(tuple(int(v) for v in shape) for shape in teacher_shapes) != tuple(tuple(int(v) for v in shape) for shape in privileged_shapes):
                raise RuntimeError("teacher normal/privileged spatial shapes differ")
            _, teacher_normal_outputs, teacher_normal_trace, normal_topk_calls = _run_direct_teacher(
                teacher, replay, teacher_memory_normal, teacher_shapes
            )
            shared_replay, teacher_privileged_outputs, teacher_privileged_trace, privileged_topk_calls = _run_direct_teacher(
                teacher, replay, teacher_memory_privileged, teacher_shapes
            )
            if normal_topk_calls or privileged_topk_calls:
                raise RuntimeError("teacher attempted Top-K selection")
            audit["checks"]["teacher_topk_calls"] += identity_topk_calls + normal_topk_calls + privileged_topk_calls
            alignment = shared_query_alignment_audit(replay, shared_replay)
            if not all(alignment[key]["exact"] for key in ("initial_query", "initial_reference_unactivated", "topk_indices")):
                raise RuntimeError(f"shared query alignment failed: {alignment}")
            if not _finite_tree((student_outputs, identity_outputs, teacher_normal_outputs, teacher_privileged_outputs)):
                raise RuntimeError("non-finite KQ output")

            student_layers = [normal_query_slice(layer, replay.normal_query_count) for layer in student_trace["query_outputs"]]
            teacher_normal_layers = [normal_query_slice(layer, replay.normal_query_count) for layer in teacher_normal_trace["query_outputs"]]
            teacher_privileged_layers = [normal_query_slice(layer, replay.normal_query_count) for layer in teacher_privileged_trace["query_outputs"]]
            memory_rows = []
            for level, (normal_memory, privileged_memory, mask) in enumerate(zip(
                teacher_memory_normal.split([h * w for h, w in teacher_shapes], dim=1),
                teacher_memory_privileged.split([h * w for h, w in teacher_shapes], dim=1),
                masks,
            )):
                memory_rows.append({
                    "level": level,
                    **_memory_stats(normal_memory, privileged_memory, mask, torch.tensor([
                        len(target["boxes"]) == 0 for target in absolute_targets
                    ], device=device)),
                })
            empty = torch.tensor([len(target["boxes"]) == 0 for target in absolute_targets], device=device)
            empty_output_delta = {
                "logits": float((teacher_privileged_outputs[1][-1][empty] - teacher_normal_outputs[1][-1][empty]).abs().max()) if bool(empty.any()) else 0.0,
                "boxes": float((teacher_privileged_outputs[0][-1][empty] - teacher_normal_outputs[0][-1][empty]).abs().max()) if bool(empty.any()) else 0.0,
            }
            row = {
                "batch": batch_index,
                "image_ids": [_image_id_repr(target) for target in raw_targets],
                "positive_images": sum(bool(target["boxes"].numel()) for target in absolute_targets),
                "empty_images": int(empty.sum()),
                "student_topk_indices": replay.topk_indices.cpu().tolist(),
                "topk_coordinates_sigmoid": replay.initial_reference_unactivated[..., :2].sigmoid().cpu().tolist(),
                "initial_query": _stats_tensor(replay.initial_query),
                "initial_reference_unactivated": _stats_tensor(replay.initial_reference_unactivated),
                "initial_reference_sigmoid": _stats_tensor(replay.initial_reference_unactivated.sigmoid()),
                "alignment": alignment,
                "identity_layers": identity_rows,
                "student_vs_teacher_normal_layers": layer_query_audit(student_layers, teacher_normal_layers),
                "student_vs_teacher_privileged_layers": layer_query_audit(student_layers, teacher_privileged_layers),
                "teacher_normal_vs_privileged_layers": layer_query_audit(teacher_normal_layers, teacher_privileged_layers),
                "memory": memory_rows,
                "empty_teacher_output_identity": empty_output_delta,
                "student_final": {"logits": _stats_tensor(student_outputs["pred_logits"]), "boxes": _stats_tensor(student_outputs["pred_boxes"])},
                "teacher_privileged_final": {"logits": _stats_tensor(teacher_privileged_outputs[1][-1]), "boxes": _stats_tensor(teacher_privileged_outputs[0][-1])},
            }
            audit["batches"].append(row)
            if first_fixed is None:
                first_fixed = {
                    "student_layers": [layer.detach().cpu() for layer in student_layers],
                    "teacher_normal_layers": [layer.detach().cpu() for layer in teacher_normal_layers],
                    "teacher_privileged_layers": [layer.detach().cpu() for layer in teacher_privileged_layers],
                }
            seen += take
    # The real train Mosaic stream may contain no all-empty composed samples.
    # Use the same fixed subset's validation view for the mandatory empty-image
    # NP identity gate, without treating it as a train prevalence estimate.
    empty_seen = 0
    with torch.no_grad():
        for samples, raw_targets in cfg.val_dataloader:
            empty_indices = [
                index for index, target in enumerate(raw_targets)
                if int(target["boxes"].shape[0]) == 0
            ]
            if not empty_indices:
                continue
            for index in empty_indices:
                if empty_seen >= args.empty_limit:
                    break
                one_sample = samples[index:index + 1].to(device)
                one_target = raw_targets[index]
                absolute_targets = [{
                    "boxes": one_target["boxes"].as_subclass(torch.Tensor).to(device),
                    "labels": one_target["labels"].to(device),
                }]
                topk_original, topk_holder = _capture_topk_indices(student_decoder)
                try:
                    student_outputs, student_trace, replay_inputs = run_model_with_capture(
                        student, one_sample, None
                    )
                finally:
                    _restore_topk(student_decoder, topk_original)
                replay = SharedQueryReplay(
                    initial_query=replay_inputs.initial_query[:, -student.decoder.num_queries:],
                    initial_reference_unactivated=replay_inputs.initial_reference_unactivated[:, -student.decoder.num_queries:],
                    topk_indices=topk_holder["indices"],
                    memory=replay_inputs.memory,
                    spatial_shapes=replay_inputs.spatial_shapes,
                    normal_query_count=student.decoder.num_queries,
                )
                teacher_backbone = teacher.backbone(one_sample)
                teacher_encoded = teacher.encoder([feature.clone() for feature in teacher_backbone])
                teacher_encoded_privileged, masks = privilege_features(
                    teacher_encoded,
                    absolute_targets,
                    one_sample.shape[-2:],
                    background_weight=0.2,
                    context_weight=None,
                    context_scale=1.5,
                )
                teacher_memory_normal, teacher_shapes = teacher.decoder._get_encoder_input(teacher_encoded)
                teacher_memory_privileged, privileged_shapes = teacher.decoder._get_encoder_input(teacher_encoded_privileged)
                _, teacher_normal_outputs, _, normal_topk_calls = _run_direct_teacher(
                    teacher, replay, teacher_memory_normal, teacher_shapes
                )
                _, teacher_privileged_outputs, _, privileged_topk_calls = _run_direct_teacher(
                    teacher, replay, teacher_memory_privileged, privileged_shapes
                )
                memory_delta = max(
                    float((normal - privileged).abs().max())
                    for normal, privileged in zip(
                        teacher_memory_normal.split([h * w for h, w in teacher_shapes], dim=1),
                        teacher_memory_privileged.split([h * w for h, w in privileged_shapes], dim=1),
                    )
                )
                output_delta = {
                    "logits": float((teacher_normal_outputs[1][-1] - teacher_privileged_outputs[1][-1]).abs().max()),
                    "boxes": float((teacher_normal_outputs[0][-1] - teacher_privileged_outputs[0][-1]).abs().max()),
                }
                mask_delta = max(float((mask - 1.0).abs().max()) for mask in masks)
                audit["empty_probe"].append({
                    "image_ids": _image_id_repr(one_target),
                    "student_topk_indices": replay.topk_indices.cpu().tolist(),
                    "mask_max_abs_from_identity": mask_delta,
                    "memory_max_abs_delta": memory_delta,
                    "teacher_output_identity": output_delta,
                    "teacher_topk_calls": normal_topk_calls + privileged_topk_calls,
                    "finite": _finite_tree((student_outputs, teacher_normal_outputs, teacher_privileged_outputs)),
                })
                if mask_delta > 0.0 or memory_delta > 0.0 or max(output_delta.values()) > 2e-6:
                    raise RuntimeError("empty-image privileged memory is not identity")
                empty_seen += 1
            if empty_seen >= args.empty_limit:
                break
    if seen != args.limit:
        raise RuntimeError(f"processed {seen} images; expected {args.limit}")
    if empty_seen < args.empty_limit:
        raise RuntimeError(f"empty-image validation coverage {empty_seen} < {args.empty_limit}")
    audit["checks"].update({
        "images_processed": seen,
        "positive_images": sum(row["positive_images"] for row in audit["batches"]),
        "empty_images": empty_seen,
        "identity_max_abs": max(row["identity_layers"][layer]["max_abs"] for row in audit["batches"] for layer in range(4)),
        "teacher_parameter_max_delta": _max_state_delta(teacher_before, teacher),
        "identity_teacher_parameter_max_delta": _max_state_delta(identity_before, identity_teacher),
    })
    if audit["checks"]["identity_max_abs"] > 2e-6:
        raise RuntimeError("identity gate failed")
    if audit["checks"]["teacher_parameter_max_delta"] != 0.0:
        raise RuntimeError("teacher parameters changed")
    if audit["checks"]["teacher_requires_grad_count"] != 0:
        raise RuntimeError("teacher is trainable")
    if audit["checks"]["empty_images"] < args.empty_limit:
        raise RuntimeError("privileged teacher empty-image coverage is incomplete")
    if first_fixed is not None:
        torch.save(first_fixed, output_dir / "fixed_batch_tensors.pt")
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (output_dir / "COMPLETED").write_text("KQ shared-query Stage-1 gates complete\n")
    print(json.dumps(audit, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--img-folder", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--empty-limit", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    args.device = torch.device(args.device)
    try:
        run(args)
    except Exception as error:
        output_dir = Path(args.out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "FAILED.json").write_text(json.dumps({
            "error_type": type(error).__name__,
            "error": str(error),
        }, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
