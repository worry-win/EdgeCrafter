"""Frozen 640-image target and pairing audit for QG0-QG3 admission."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ecdetseg"))
sys.path.insert(0, str(ROOT))

from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy  # noqa: E402
from scripts.ablation.cmp5L_privileged_decision_kd import run_with_all_score_heads  # noqa: E402
from scripts.ablation.cmp5L_qg_behavior_distillation import (  # noqa: E402
    attach_query_gate_head,
    bernoulli_behavior_kd,
    match_competitive_candidates,
    temporary_query_gate_state,
)
from scripts.ablation.cmp5L_qg_stage1_smoke import build, finite_tree  # noqa: E402
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
    _normalise_targets,
)


def matcher_indices(criterion, output, targets):
    result = criterion.matcher({
        "pred_logits": output["pred_logits"].detach().float(),
        "pred_boxes": output["pred_boxes"].detach().float(),
    }, targets)
    return result["indices"] if isinstance(result, dict) else result


def positive_pairs(student_matches, teacher_matches, targets):
    pairs = []
    class_counts = Counter()
    for batch, ((student_query, student_gt), (teacher_query, teacher_gt)) in enumerate(
        zip(student_matches, teacher_matches)
    ):
        student_by_gt = {int(gt): int(query) for query, gt in zip(student_query, student_gt)}
        teacher_by_gt = {int(gt): int(query) for query, gt in zip(teacher_query, teacher_gt)}
        for gt_index in sorted(set(student_by_gt) & set(teacher_by_gt)):
            pairs.append([batch, student_by_gt[gt_index], teacher_by_gt[gt_index]])
            class_counts[int(targets[batch]["labels"][gt_index])] += 1
    device = targets[0]["labels"].device
    return torch.tensor(pairs, dtype=torch.long, device=device).reshape(-1, 3), class_counts


def run(args):
    if args.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cfg, solver, student = build(args)
    baseline = copy.deepcopy(student).eval().requires_grad_(False)
    teacher = copy.deepcopy(baseline.decoder).eval().requires_grad_(False)
    gate_module = student.decoder.decoder
    attach_query_gate_head(student, hidden_dim=64, prior_probability=0.01)
    gate_module.query_gate_enabled = False
    oracle_audit = _reference_audit()
    summary = {
        "images": 0,
        "empty_images": 0,
        "gt": 0,
        "gt_by_class": Counter(),
        "selection_queries": 0,
        "selection_protect": 0,
        "selection_suppress": 0,
        "gt_selection_by_class": {str(index): Counter() for index in range(4)},
        "positive_pairs": 0,
        "positive_pairs_by_class": Counter(),
        "competitive_pairs": 0,
        "competitive_pairs_empty": 0,
        "competitive_pairs_by_top_class": Counter(),
        "student_candidates": 0,
        "teacher_candidates": 0,
        "final_kd_sum": 0.0,
        "all_layer_kd_sum": 0.0,
        "batches": 0,
    }
    with torch.no_grad():
        for samples, raw_targets in cfg.val_dataloader:
            samples = samples.to(args.device)
            targets = _normalise_targets(
                raw_targets, samples.shape[-2], samples.shape[-1], args.device
            )
            image_ids = [int(target["image_id"].item()) for target in raw_targets]
            with DecoderInternalCapture(baseline.decoder) as capture:
                normal, normal_layers = run_with_all_score_heads(
                    baseline.decoder, lambda: baseline(samples)
                )
            gt_xyxy = [box_cxcywh_to_xyxy(target["boxes"]) for target in targets]
            attention, regions = [], []
            for layer_index in range(3):
                record = capture.layers[layer_index]
                attention.append(record["attention_weights"])
                regions.append(torch.stack([
                    region_mask_with_outside(
                        record["sampling_locations"][batch], gt_xyxy[batch]
                    )[0]
                    for batch in range(len(targets))
                ]))
            dependency = normal_bg_dependency(attention, regions)
            has_gt = torch.tensor(
                [bool(target["labels"].numel()) for target in targets],
                device=args.device,
            )
            gt_gate = (dependency > 0.20) & has_gt[:, None]
            privileged, privileged_layers = run_with_all_score_heads(
                teacher,
                lambda: run_explicit_gate_branch(
                    teacher, capture.decoder_input, "E", gt_gate, gt_xyxy,
                    image_ids, set(), oracle_audit,
                )[0],
            )
            with temporary_query_gate_state(gate_module, True):
                student_output, student_layers = run_with_all_score_heads(
                    student.decoder, lambda: student(samples)
                )
            if not all(finite_tree(value) for value in (normal, privileged, student_output)):
                raise RuntimeError("non-finite frozen probe output")
            student_match = matcher_indices(solver.criterion, student_output, targets)
            teacher_match = matcher_indices(solver.criterion, privileged, targets)
            positives, class_pairs = positive_pairs(student_match, teacher_match, targets)
            competition = match_competitive_candidates(
                student_output["pred_logits"], student_output["pred_boxes"],
                privileged["pred_logits"], privileged["pred_boxes"],
                positive_pairs=positives, top_k=20, score_threshold=0.05, min_iou=0.30,
            )
            final_kd = bernoulli_behavior_kd(
                [student_layers[-1]], [privileged_layers[-1]],
                positive_pairs=positives, competitive_pairs=competition["pairs"],
            )
            all_kd = bernoulli_behavior_kd(
                student_layers, privileged_layers,
                positive_pairs=positives, competitive_pairs=competition["pairs"],
            )
            summary["images"] += len(targets)
            summary["empty_images"] += int((~has_gt).sum())
            summary["batches"] += 1
            summary["selection_queries"] += int(gt_gate.numel())
            summary["selection_suppress"] += int(gt_gate.sum())
            summary["selection_protect"] += int((~gt_gate).sum())
            summary["positive_pairs"] += int(positives.shape[0])
            summary["positive_pairs_by_class"].update(class_pairs)
            summary["competitive_pairs"] += int(competition["pairs"].shape[0])
            summary["student_candidates"] += competition["student_candidate_count"]
            summary["teacher_candidates"] += competition["teacher_candidate_count"]
            summary["final_kd_sum"] += float(final_kd["loss"])
            summary["all_layer_kd_sum"] += float(all_kd["loss"])
            empty_batches = {index for index, value in enumerate(has_gt.tolist()) if not value}
            for record in competition["records"]:
                summary["competitive_pairs_by_top_class"][record["top_class"]] += 1
                if record["batch"] in empty_batches:
                    summary["competitive_pairs_empty"] += 1
            for batch, target in enumerate(targets):
                labels = target["labels"]
                summary["gt"] += int(labels.numel())
                summary["gt_by_class"].update(map(int, labels.tolist()))
                student_query, student_gt = student_match[batch]
                for query, gt_index in zip(student_query.tolist(), student_gt.tolist()):
                    label = int(labels[gt_index])
                    kind = "suppress" if bool(gt_gate[batch, query]) else "protect"
                    summary["gt_selection_by_class"][str(label)][kind] += 1
    result = {
        key: (dict(value) if isinstance(value, Counter) else value)
        for key, value in summary.items()
    }
    result["gt_selection_by_class"] = {
        key: dict(value) for key, value in summary["gt_selection_by_class"].items()
    }
    result["final_kd_mean"] = summary["final_kd_sum"] / summary["batches"]
    result["all_layer_kd_mean"] = summary["all_layer_kd_sum"] / summary["batches"]
    result["oracle_audit"] = {
        "coeff_min": oracle_audit["coeff_min"]["E"],
        "coeff_max": oracle_audit["coeff_max"]["E"],
        "coeff_shapes": oracle_audit["coeff_shapes"]["E"],
        "location_input_mutation": oracle_audit["location_input_mutation_max_abs"],
        "attention_input_mutation": oracle_audit["attention_input_mutation_max_abs"],
    }
    failures = []
    if result["images"] != 640:
        failures.append(f"images={result['images']}")
    for class_index in range(4):
        gt_count = int(result["gt_by_class"].get(class_index, result["gt_by_class"].get(str(class_index), 0)))
        pair_count = int(result["positive_pairs_by_class"].get(class_index, result["positive_pairs_by_class"].get(str(class_index), 0)))
        if gt_count < 32:
            failures.append(f"class{class_index}_gt={gt_count}<32")
        if gt_count and pair_count / gt_count < 0.95:
            failures.append(f"class{class_index}_positive_pair_coverage={pair_count/gt_count:.4f}<0.95")
    if result["selection_protect"] == 0 or result["selection_suppress"] == 0:
        failures.append("selection label lacks protect or suppress")
    if result["competitive_pairs"] < 100 or result["competitive_pairs_empty"] < 10:
        failures.append("competitive pair coverage below locked floor")
    if not (result["final_kd_mean"] > 0 and result["all_layer_kd_mean"] > 0):
        failures.append("behavior KD is zero or non-finite")
    if (
        result["oracle_audit"]["location_input_mutation"] != 0.0
        or result["oracle_audit"]["attention_input_mutation"] != 0.0
    ):
        failures.append("Oracle mutated routing inputs")
    result["go_failures"] = failures
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(result, indent=2) + "\n")
    if failures:
        raise RuntimeError(f"QG frozen Go gates failed: {failures}")
    (output.parent / "COMPLETED").write_text("QG frozen 640 probe complete\n")
    print(json.dumps(result, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--img-folder", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", default="ema")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--head-hidden", type=int, default=64)
    args = parser.parse_args()
    args.device = torch.device(args.device)
    run(args)


if __name__ == "__main__":
    main()
