"""Pre-locked single/small-group query counterfactual attribution probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy
from scripts.ablation.cmp5L_privileged_decision_kd import (
    build_counterfactual_gates,
    max_named_buffer_change,
)
from scripts.ablation.evaluate_cmp5L_af_validation import (
    checkpoint_key_audit,
    normal_bg_dependency,
    region_mask_with_outside,
)
from scripts.ablation.evaluate_cmp5L_gh_validation import run_explicit_gate_branch
from scripts.ablation.probe_cmp5L_decoder_internal_tensors import (
    DecoderInternalCapture,
    _build,
    _normalise_targets,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_iou(box, target):
    left = box_cxcywh_to_xyxy(box[None])[0]
    right = box_cxcywh_to_xyxy(target[None])[0]
    tl = torch.maximum(left[:2], right[:2])
    br = torch.minimum(left[2:], right[2:])
    inter = (br - tl).clamp(min=0).prod()
    area_l = (left[2:] - left[:2]).clamp(min=0).prod()
    area_r = (right[2:] - right[:2]).clamp(min=0).prod()
    return float(inter / (area_l + area_r - inter).clamp(min=1e-12))


def targeted_state(output, query, class_index, target=None):
    scores = output["pred_logits"][0, query].sigmoid()
    state = {
        "score": float(scores[class_index]),
        "top_class": int(scores.argmax()),
        "margin": float(scores[class_index] - torch.cat((scores[:class_index], scores[class_index + 1:])).max()),
        "box": output["pred_boxes"][0, query].detach().cpu().tolist(),
    }
    if target is None:
        state["is_suppressed"] = state["score"] < 0.5
    else:
        state["iou"] = paired_iou(output["pred_boxes"][0, query], target)
        state["is_correct"] = state["score"] >= 0.5 and state["top_class"] == class_index and state["iou"] >= 0.5
    return state


def evaluate(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    args.device = device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    locked = json.loads(Path(args.case_manifest).read_text(encoding="utf-8"))
    cases_by_image = {}
    for category, cases in locked["cases"].items():
        for ordinal, case in enumerate(cases):
            item = {**case, "case_id": f"{category}_{ordinal}"}
            cases_by_image.setdefault(int(case["image_id"]), []).append(item)

    cfg, solver, model = _build(args)
    model.eval().requires_grad_(False)
    load = checkpoint_key_audit(model, args.checkpoint, args.weights)
    if load["missing_keys"] or load["unexpected_keys"]:
        raise RuntimeError(f"strict checkpoint mismatch: {load}")
    buffer_before = {name: value.detach().clone() for name, value in model.named_buffers()}
    audit = {
        "model_load": load,
        "normal_capture_max_abs": {"logits": 0.0, "boxes": 0.0},
        "location_input_mutation_max_abs": 0.0,
        "attention_input_mutation_max_abs": 0.0,
        "coeff_min": {name: [float("inf")] * 3 for name in ("N", "P", "selected", "excluded")},
        "coeff_max": {name: [float("-inf")] * 3 for name in ("N", "P", "selected", "excluded")},
        "coeff_shapes": {name: [None] * 3 for name in ("N", "P", "selected", "excluded")},
    }
    rows, seen = [], []
    with torch.no_grad():
        for samples, targets in cfg.val_dataloader:
            samples = samples.to(device)
            image_id = int(targets[0]["image_id"].item())
            if image_id not in cases_by_image:
                raise RuntimeError(f"unexpected case image {image_id}")
            normalized = _normalise_targets(targets, samples.shape[-2], samples.shape[-1], device)
            gt_xyxy = [box_cxcywh_to_xyxy(normalized[0]["boxes"])]
            live = model(samples)
            with DecoderInternalCapture(model.decoder) as capture:
                hooked = model(samples)
            audit["normal_capture_max_abs"]["logits"] = max(audit["normal_capture_max_abs"]["logits"], float((live["pred_logits"] - hooked["pred_logits"]).abs().max()))
            audit["normal_capture_max_abs"]["boxes"] = max(audit["normal_capture_max_abs"]["boxes"], float((live["pred_boxes"] - hooked["pred_boxes"]).abs().max()))
            attention, regions = [], []
            for layer in range(3):
                rec = capture.layers[layer]
                attention.append(rec["attention_weights"])
                regions.append(torch.stack([region_mask_with_outside(rec["sampling_locations"][0], gt_xyxy[0])[0]]))
            full_gate = normal_bg_dependency(attention, regions) > 0.20
            zero_gate = torch.zeros_like(full_gate)
            n_out, _, _ = run_explicit_gate_branch(model.decoder, capture.decoder_input, "N", zero_gate, gt_xyxy, [image_id], set(), audit)
            p_out, _, _ = run_explicit_gate_branch(model.decoder, capture.decoder_input, "P", full_gate, gt_xyxy, [image_id], set(), audit)
            for case in cases_by_image[image_id]:
                selected_gate, excluded_gate = build_counterfactual_gates(full_gate, [case["query_ids"]])
                selected_out, _, _ = run_explicit_gate_branch(model.decoder, capture.decoder_input, "selected", selected_gate, gt_xyxy, [image_id], set(), audit)
                excluded_out, _, _ = run_explicit_gate_branch(model.decoder, capture.decoder_input, "excluded", excluded_gate, gt_xyxy, [image_id], set(), audit)
                query_states = {}
                components = case.get("components") or [case]
                for component in components:
                    query = int(component["query_ids"][0])
                    label = int(component["class_index"])
                    target = None if component.get("gt_index") is None else normalized[0]["boxes"][int(component["gt_index"])]
                    query_states[str(query)] = {
                        name: targeted_state(output, query, label, target)
                        for name, output in (("N", n_out), ("P", p_out), ("selected", selected_out), ("excluded", excluded_out))
                    }
                rows.append({
                    **case,
                    "full_gate_count": int(full_gate.sum()),
                    "requested_active_count": int(selected_gate.sum()),
                    "states": query_states,
                })
            seen.append(image_id)

    if sorted(seen) != sorted(map(int, locked["image_ids"])):
        raise RuntimeError(f"case image coverage mismatch: {seen}")
    audit["teacher_buffer_max_abs_change"] = max_named_buffer_change(buffer_before, model.named_buffers())
    if max([*audit["normal_capture_max_abs"].values(), audit["location_input_mutation_max_abs"], audit["attention_input_mutation_max_abs"], audit["teacher_buffer_max_abs_change"]]) > 2e-6:
        raise RuntimeError(f"counterfactual technical audit failed: {audit}")

    summary = {"repair": {"cases": 0, "selected_recovers": 0}, "protect": {"cases": 0, "selected_breaks": 0, "excluded_restores": 0}, "suppress": {"cases": 0, "selected_suppresses": 0}, "mixed": {"cases": 0, "all_selected_suppress": 0}}
    for row in rows:
        category = row["category"]
        summary[category]["cases"] += 1
        values = list(row["states"].values())
        if category == "repair":
            summary[category]["selected_recovers"] += int(values[0]["selected"]["is_correct"] and not values[0]["N"]["is_correct"])
        elif category == "protect":
            summary[category]["selected_breaks"] += int(values[0]["N"]["is_correct"] and not values[0]["selected"]["is_correct"])
            summary[category]["excluded_restores"] += int(values[0]["excluded"]["is_correct"] and not values[0]["P"]["is_correct"])
        elif category == "suppress":
            summary[category]["selected_suppresses"] += int(values[0]["selected"]["is_suppressed"] and not values[0]["N"]["is_suppressed"])
        else:
            summary[category]["all_selected_suppress"] += int(all(v["selected"]["is_suppressed"] for v in values))
    result = {
        "scope": "prelocked single/small-group all-point L0-L2 query interventions",
        "summary": summary,
        "cases": rows,
        "audit": audit,
        "hashes": {"case_manifest": sha256(args.case_manifest), "annotations": sha256(args.ann_file), "checkpoint": sha256(args.checkpoint), "runner": sha256(__file__)},
    }
    (out_dir / "counterfactual_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out_dir / "COMPLETED.txt").write_text("status=passed\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "audit": audit}, ensure_ascii=False), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--case-manifest", required=True)
    parser.add_argument("--img-folder", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", default="ema", choices=("ema",))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
