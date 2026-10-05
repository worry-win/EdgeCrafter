"""Uniform mechanism evaluation for cmp5L query-behavior KD checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "ecdetseg"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ablation.cmp5L_query_behavior_kd import (  # noqa: E402
    replay_frozen_teacher,
    run_model_with_capture,
    summarize_behavior_batch,
)
from scripts.ablation.probe_cmp5L_decoder_internal_tensors import (  # noqa: E402
    _build,
    _normalise_targets,
)
from scripts.ablation.train_cmp5L_query_behavior_kd import (  # noqa: E402
    _load_frozen_teacher_decoder,
    _matches,
)


def _add_nested(target, source):
    for key, value in source.items():
        if isinstance(value, dict):
            _add_nested(target.setdefault(key, {}), value)
        else:
            target[key] = target.get(key, 0) + value


def _finalize(accumulator):
    negative = accumulator["negative"]
    positive = accumulator["positive"]
    n_negative = max(int(negative["count"]), 1)
    n_positive = max(int(positive["count"]), 1)
    result = {
        "negative_queries": {
            "count": int(negative["count"]),
            "mean_student_max_class_score": negative["student_score_sum"] / n_negative,
            "mean_privileged_teacher_max_class_score": negative["teacher_score_sum"] / n_negative,
            "student_high_confidence_count": int(negative["student_high_confidence_count"]),
            "privileged_teacher_high_confidence_count": int(negative["teacher_high_confidence_count"]),
        },
        "matched_positive_queries": {
            "count": int(positive["count"]),
            "mean_student_gt_class_probability": positive["student_gt_probability_sum"] / n_positive,
            "mean_privileged_teacher_gt_class_probability": positive["teacher_gt_probability_sum"] / n_positive,
            "mean_student_class_margin": positive["student_margin_sum"] / n_positive,
            "mean_privileged_teacher_class_margin": positive["teacher_margin_sum"] / n_positive,
            "mean_student_iou": positive["student_iou_sum"] / n_positive,
            "mean_privileged_teacher_iou": positive["teacher_iou_sum"] / n_positive,
        },
        "layers": {},
    }
    for layer, values in accumulator["layers"].items():
        count = max(int(values["count"]), 1)
        result["layers"][layer] = {
            "matched_query_count": int(values["count"]),
            "mean_student_teacher_delta_q_cosine": values["delta_cosine_sum"] / count,
            "mean_relative_sampling_l1_to_teacher": values["relative_sampling_l1_sum"] / count,
        }
    return result


def evaluate(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    args.device = device
    cfg, solver, model = _build(args)
    teacher = _load_frozen_teacher_decoder(
        model.decoder, args.teacher_checkpoint, device
    )
    teacher.eval()
    teacher.requires_grad_(False)

    accumulator = {}
    oracle_dependency_sum = 0.0
    oracle_high_fraction_sum = 0.0
    oracle_batches = 0
    seen = 0
    started = time.time()
    with torch.no_grad():
        for samples, raw_targets in cfg.val_dataloader:
            if args.max_images and seen >= args.max_images:
                break
            samples = samples.to(device)
            matcher_targets = _normalise_targets(
                raw_targets, samples.shape[-2], samples.shape[-1], device
            )
            predictions, student_trace, replay_inputs = run_model_with_capture(
                model, samples, None
            )
            teacher_predictions, teacher_trace, diagnostics = replay_frozen_teacher(
                teacher,
                replay_inputs,
                matcher_targets,
                privileged=True,
                distill_layers=(0, 1, 2),
            )
            if not (
                diagnostics["initial_query_exact"]
                and diagnostics["initial_reference_exact"]
            ):
                raise RuntimeError(f"shared initialization failed: {diagnostics}")
            matches = _matches(solver.criterion, predictions, matcher_targets)
            batch_summary = summarize_behavior_batch(
                student_trace,
                teacher_trace,
                predictions,
                teacher_predictions,
                matcher_targets,
                matches,
                distill_layers=(0, 1, 2),
                high_confidence=args.high_confidence,
            )
            _add_nested(accumulator, batch_summary)
            oracle_dependency_sum += diagnostics["bg_dependency_mean"]
            oracle_high_fraction_sum += diagnostics["high_dependency_fraction"]
            oracle_batches += 1
            seen += len(raw_targets)
            if args.log_every and seen % args.log_every < len(raw_targets):
                print(f"[{seen}/{len(cfg.val_dataloader.dataset)}]", flush=True)

    if args.expected_images and seen != args.expected_images:
        raise RuntimeError(f"evaluated {seen}, expected {args.expected_images}")
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("teacher unexpectedly received gradients")
    payload = {
        "meta": {
            "checkpoint": args.checkpoint,
            "teacher_checkpoint": args.teacher_checkpoint,
            "weights": args.weights,
            "ann_file": args.ann_file,
            "n_images": seen,
            "high_confidence_threshold": args.high_confidence,
            "seconds": round(time.time() - started, 2),
            "teacher_eval": not teacher.training,
            "teacher_trainable_parameters": sum(
                parameter.numel() for parameter in teacher.parameters()
                if parameter.requires_grad
            ),
            "teacher_gradient_tensors": sum(
                parameter.grad is not None for parameter in teacher.parameters()
            ),
        },
        "oracle": {
            "mean_bg_dependency": oracle_dependency_sum / max(oracle_batches, 1),
            "mean_high_dependency_fraction": oracle_high_fraction_sum / max(oracle_batches, 1),
        },
        "mechanism": _finalize(accumulator),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["mechanism"], ensure_ascii=False, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--img-folder", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--expected-images", type=int, default=2011)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--high-confidence", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
