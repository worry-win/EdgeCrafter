"""Frozen, validation-only Stage-1 smoke for the QG route.

This script never updates weights. It checks the single-forward student gate,
replays the already verified P intervention for target construction, and writes
an auditable JSON manifest before any QG training is considered.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ecdetseg"))
sys.path.insert(0, str(ROOT))

from engine.core import YAMLConfig  # noqa: E402
from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy  # noqa: E402
from engine.misc import dist_utils  # noqa: E402
from engine.solver import TASKS  # noqa: E402
from scripts.ablation.evaluate_cmp5L_af_validation import (  # noqa: E402
    normal_bg_dependency,
    region_mask_with_outside,
)
from scripts.ablation.evaluate_cmp5L_gh_validation import (  # noqa: E402
    _reference_audit,
    run_explicit_gate_branch,
)
from scripts.ablation.cmp5L_qg_behavior_distillation import (  # noqa: E402
    attach_query_gate_head,
    max_named_buffer_change,
    temporary_query_gate_state,
)
from scripts.ablation.probe_cmp5L_decoder_internal_tensors import (  # noqa: E402
    DecoderInternalCapture,
    _normalise_targets,
)


def build(args):
    cfg = YAMLConfig(
        args.config,
        **{
            "val_dataloader": {
                "dataset": {
                    "ann_file": args.ann_file,
                    "img_folder": args.img_folder,
                },
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
    solver._setup()
    solver.load_resume_state(args.checkpoint)
    model = solver.ema.module if args.weights == "ema" and solver.ema is not None else solver.model
    return cfg, solver, dist_utils.de_parallel(model).to(args.device).eval()


def finite_tree(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return True


def max_prediction_diff(first, second):
    return {
        "logits": float((first["pred_logits"] - second["pred_logits"]).abs().max().detach()),
        "boxes": float((first["pred_boxes"] - second["pred_boxes"]).abs().max().detach()),
    }


def run(args):
    if args.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback")
    cfg, solver, model = build(args)
    baseline_model = copy.deepcopy(model).eval().requires_grad_(False)
    teacher_transformer = copy.deepcopy(baseline_model.decoder).eval().requires_grad_(False)
    teacher_buffers_before = {
        name: value.detach().clone()
        for name, value in teacher_transformer.named_buffers()
    }
    transformer = model.decoder
    module = transformer.decoder
    attach_query_gate_head(model, hidden_dim=args.head_hidden, prior_probability=0.01)
    module.query_gate_enabled = False
    records = []
    seen = 0
    positive_images = empty_images = 0
    max_off_diff = {"logits": 0.0, "boxes": 0.0}
    max_input_mutation = 0.0
    max_capture_diff = {"logits": 0.0, "boxes": 0.0}
    max_query_init_diff = 0.0
    max_reference_init_diff = 0.0
    max_empty_identity_diff = {"logits": 0.0, "boxes": 0.0}
    p_dependency = []
    oracle_audit = _reference_audit()
    last_samples = None
    last_gated = None
    with torch.no_grad():
        for samples, targets in cfg.val_dataloader:
            if seen >= args.limit:
                break
            samples = samples.to(args.device)
            before = samples.detach().clone()
            normal_live = baseline_model(samples)
            with DecoderInternalCapture(baseline_model.decoder) as normal_capture:
                normal = baseline_model(samples)
            capture_diff = max_prediction_diff(normal_live, normal)
            for key in max_capture_diff:
                max_capture_diff[key] = max(max_capture_diff[key], capture_diff[key])
            matcher_targets = _normalise_targets(
                targets, samples.shape[-2], samples.shape[-1], args.device
            )
            with temporary_query_gate_state(module, False):
                with DecoderInternalCapture(model.decoder) as student_capture:
                    disabled = model(samples)
            off_diff = max_prediction_diff(normal, disabled)
            for key in max_off_diff:
                max_off_diff[key] = max(max_off_diff[key], off_diff[key])
            max_query_init_diff = max(
                max_query_init_diff,
                float((
                    normal_capture.decoder_input["query_content"]
                    - student_capture.decoder_input["query_content"]
                ).abs().max()),
            )
            max_reference_init_diff = max(
                max_reference_init_diff,
                float((
                    normal_capture.decoder_input["reference_unact"]
                    - student_capture.decoder_input["reference_unact"]
                ).abs().max()),
            )
            with temporary_query_gate_state(module, True):
                gated = model(samples)
            if not finite_tree(gated):
                raise RuntimeError("non-finite gated output")
            gate_logits = module.last_query_gate_logits
            gate_probability = module.last_query_gate_probability
            if gate_logits is None or gate_probability is None:
                raise RuntimeError("QG gate did not expose L0 logits/probability")
            if not (0.0 <= float(gate_probability.min()) <= float(gate_probability.max()) <= 1.0):
                raise RuntimeError("gate probability outside [0,1]")
            coefficients = 1.0 - 0.8 * gate_probability
            if float(coefficients.min()) < 0.2 - 1e-6 or float(coefficients.max()) > 1.0 + 1e-6:
                raise RuntimeError("gate coefficient outside [0.2,1]")
            gt_xyxy = [
                box_cxcywh_to_xyxy(target["boxes"])
                for target in matcher_targets
            ]
            regions = []
            attention = []
            for layer_index in range(3):
                layer = normal_capture.layers[layer_index]
                attention.append(layer["attention_weights"])
                regions.append(torch.stack([
                    region_mask_with_outside(
                        layer["sampling_locations"][batch_index],
                        gt_xyxy[batch_index],
                    )[0]
                    for batch_index in range(len(targets))
                ]))
            dependency = normal_bg_dependency(attention, regions)
            has_gt = torch.tensor(
                [bool(boxes.shape[0]) for boxes in gt_xyxy],
                device=args.device,
            )
            gt_gate = (dependency > 0.20) & has_gt[:, None]
            image_ids = [int(target["image_id"].item()) for target in targets]
            p_output, _, _ = run_explicit_gate_branch(
                teacher_transformer,
                normal_capture.decoder_input,
                "E",
                gt_gate,
                gt_xyxy,
                image_ids,
                set(),
                oracle_audit,
            )
            empty_indices = [
                index for index, target in enumerate(targets)
                if int(target["labels"].numel()) == 0
            ]
            if empty_indices:
                for key in ("pred_logits", "pred_boxes"):
                    short = "logits" if key == "pred_logits" else "boxes"
                    difference = float(
                        (p_output[key][empty_indices] - normal[key][empty_indices]).abs().max()
                    )
                    max_empty_identity_diff[short] = max(
                        max_empty_identity_diff[short], difference
                    )
            after = samples.detach()
            max_input_mutation = max(max_input_mutation, float((before - after).abs().max()))
            positive = sum(int(target["labels"].numel() > 0) for target in targets)
            positive_images += positive
            empty_images += len(targets) - positive
            p_dependency.append(float(gt_gate.float().mean()))
            records.append({
                "image_ids": [int(target["image_id"].item()) for target in targets],
                "positive_image_count": positive,
                "empty_image_count": len(targets) - positive,
                "gate_probability_mean": float(gate_probability.mean()),
                "gate_probability_min": float(gate_probability.min()),
                "gate_probability_max": float(gate_probability.max()),
                "coefficient_mean": float(coefficients.mean()),
                "coefficient_min": float(coefficients.min()),
                "coefficient_max": float(coefficients.max()),
                "p_dependency_high_fraction": p_dependency[-1],
                "normal_off_parity": off_diff,
                "normal_capture_parity": capture_diff,
                "p_finite": finite_tree(p_output),
            })
            last_samples = samples.detach().clone()
            last_gated = {key: value.detach().clone() for key, value in gated.items() if torch.is_tensor(value)}
            seen += len(targets)
    if seen != args.limit:
        raise RuntimeError(f"processed {seen} images, expected {args.limit}")
    if positive_images == 0 or empty_images == 0:
        raise RuntimeError("smoke must cover both positive and empty images")
    fresh = copy.deepcopy(model)
    fresh.load_state_dict(model.state_dict(), strict=True)
    with temporary_query_gate_state(fresh.decoder.decoder, True):
        loaded = fresh(last_samples)
    loaded_diff = max_prediction_diff(last_gated, loaded)
    teacher_buffer_mutation = max_named_buffer_change(
        teacher_buffers_before,
        teacher_transformer.named_buffers(),
    )
    parity_checks = {
        "gate_off_parity": max_off_diff,
        "normal_capture_parity": max_capture_diff,
        "empty_p_identity_parity": max_empty_identity_diff,
        "strict_loaded_output_parity": loaded_diff,
    }
    for name, values in parity_checks.items():
        if max(values.values()) > 2e-6:
            raise RuntimeError(f"{name} failed: {values}")
    if max_query_init_diff != 0.0 or max_reference_init_diff != 0.0:
        raise RuntimeError(
            f"student/baseline initialization mismatch: query={max_query_init_diff}, "
            f"reference={max_reference_init_diff}"
        )
    if (
        oracle_audit["location_input_mutation_max_abs"] != 0.0
        or oracle_audit["attention_input_mutation_max_abs"] != 0.0
    ):
        raise RuntimeError(f"Oracle mutated routing inputs: {oracle_audit}")
    for layer_index in range(3):
        if abs(oracle_audit["coeff_min"]["E"][layer_index] - 0.2) > 1e-6:
            raise RuntimeError(f"Oracle L{layer_index} never applied coefficient 0.2")
        if abs(oracle_audit["coeff_max"]["E"][layer_index] - 1.0) > 1e-6:
            raise RuntimeError(f"Oracle L{layer_index} did not preserve coefficient 1")
        if oracle_audit["coeff_shapes"]["E"][layer_index] is None:
            raise RuntimeError(f"Oracle L{layer_index} coefficient shape missing")
    if max_input_mutation != 0.0 or teacher_buffer_mutation != 0.0:
        raise RuntimeError(
            f"mutation audit failed: input={max_input_mutation}, "
            f"teacher_buffers={teacher_buffer_mutation}"
        )
    if any(parameter.grad is not None for parameter in teacher_transformer.parameters()):
        raise RuntimeError("frozen teacher received gradients")
    payload = {
        "meta": {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "weights": args.weights,
            "ann_file": args.ann_file,
            "n_images": seen,
            "head_hidden": args.head_hidden,
            "head_prior_probability": 0.01,
            "qg_layers": [0, 1, 2],
            "selection_label_source": "same-current-student-frozen-ungated-replay",
            "behavior_pool": {"top_k": 20, "score_threshold": 0.05, "iou": 0.30},
        },
        "checks": {
            "cuda": args.device.type == "cuda",
            "strict_gate_state_load": True,
            "gate_off_parity": max_off_diff,
            "normal_capture_parity": max_capture_diff,
            "initial_query_max_abs_diff": max_query_init_diff,
            "initial_reference_max_abs_diff": max_reference_init_diff,
            "empty_p_identity_parity": max_empty_identity_diff,
            "strict_loaded_output_parity": loaded_diff,
            "input_max_abs_mutation": max_input_mutation,
            "teacher_buffer_max_abs_mutation": teacher_buffer_mutation,
            "teacher_gradient_count": sum(
                parameter.grad is not None for parameter in teacher_transformer.parameters()
            ),
            "p_dependency_mean": sum(p_dependency) / len(p_dependency),
            "oracle_coeff_min": oracle_audit["coeff_min"]["E"],
            "oracle_coeff_max": oracle_audit["coeff_max"]["E"],
            "oracle_coeff_shapes": oracle_audit["coeff_shapes"]["E"],
            "oracle_location_input_mutation": oracle_audit["location_input_mutation_max_abs"],
            "oracle_attention_input_mutation": oracle_audit["attention_input_mutation_max_abs"],
            "positive_images": positive_images,
            "empty_images": empty_images,
            "finite": True,
            "gate_state_keys": sum("query_gate_head" in key for key in model.state_dict()),
        },
        "records": records,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output.parent / "COMPLETED").write_text("stage1 smoke completed\n", encoding="utf-8")
    print(json.dumps(payload["checks"], sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--img-folder", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--head-hidden", type=int, default=64)
    args = parser.parse_args()
    args.device = torch.device(args.device)
    run(args)


if __name__ == "__main__":
    main()
