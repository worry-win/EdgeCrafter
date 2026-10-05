"""Frozen-baseline cmp5L G/H validation-only gate experiments."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


H_THRESHOLDS = (0.05, 0.20, 0.50, 0.80, 0.95)
LAMBDA_MIN = 0.2


def all_point_coeff(gate, heads, points, lambda_min=LAMBDA_MIN):
    """Broadcast a binary query gate to every head and sampled point."""
    active = gate.to(dtype=torch.float32)[:, :, None, None]
    coeff = 1.0 - (1.0 - float(lambda_min)) * active
    return coeff.expand(-1, -1, int(heads), int(points))


def stratified_image_split(image_ids, positive_ids, duct_ids, seed=20260919):
    """Return a fixed 50/50 image split balanced over empty/duct/other strata."""
    image_ids = set(map(int, image_ids))
    positive_ids = set(map(int, positive_ids))
    duct_ids = set(map(int, duct_ids))
    if not duct_ids <= positive_ids <= image_ids:
        raise ValueError("split strata must satisfy duct <= positive <= all images")
    strata = (
        image_ids - positive_ids,
        duct_ids,
        positive_ids - duct_ids,
    )
    rng = np.random.default_rng(int(seed))
    calibration, evaluation = [], []
    for stratum in strata:
        values = np.asarray(sorted(stratum), dtype=np.int64)
        rng.shuffle(values)
        split_at = len(values) // 2
        calibration.extend(map(int, values[:split_at]))
        evaluation.extend(map(int, values[split_at:]))
    return {
        "calibration": sorted(calibration),
        "evaluation": sorted(evaluation),
    }


def _threshold_name(threshold):
    return f"H{int(round(float(threshold) * 100)):03d}"


def build_h_gates(p_suppress, thresholds=H_THRESHOLDS):
    """Build the pre-locked conservative predictor gates."""
    if tuple(map(float, thresholds)) != H_THRESHOLDS:
        raise ValueError(f"H thresholds must remain locked to {H_THRESHOLDS}")
    return {
        _threshold_name(threshold): p_suppress > float(threshold)
        for threshold in thresholds
    }


def build_g_gates(g_gt, g_pred, has_gt):
    """Build G0–G3 gates; empty images keep the deployable G0 behavior."""
    positive = has_gt.to(dtype=torch.bool)[:, None]
    corrected_fs = g_pred & g_gt
    corrected_ms = g_pred | g_gt
    return {
        "G0": g_pred,
        "G1": torch.where(positive, corrected_fs, g_pred),
        "G2": torch.where(positive, corrected_ms, g_pred),
        "G3": torch.where(positive, g_gt, g_pred),
    }


def gate_change_audit(gates, g_gt, g_pred, has_gt):
    """Verify and count the exact FS/MS edits defining G1–G3."""
    positive = has_gt.to(dtype=torch.bool)[:, None].expand_as(g_gt)
    false_suppress = positive & ~g_gt & g_pred
    missed_suppress = positive & g_gt & ~g_pred
    expected = {
        "G0": torch.zeros_like(g_pred),
        "G1": false_suppress,
        "G2": missed_suppress,
        "G3": false_suppress | missed_suppress,
    }
    for name, expected_change in expected.items():
        actual_change = gates[name] != g_pred
        if not torch.equal(actual_change, expected_change):
            raise RuntimeError(f"{name} changed queries outside its locked error set")
    return {
        "fs_count": int(false_suppress.sum()),
        "ms_count": int(missed_suppress.sum()),
        **{f"{name}_changed": int(mask.sum()) for name, mask in expected.items()},
    }


def _stored_tensor(tensor):
    tensor = tensor.detach().cpu()
    if tensor.dtype == torch.bool:
        return tensor.to(torch.uint8)
    if tensor.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
        return tensor.to(torch.int16)
    return tensor.to(torch.float16)


def _update_gate_hash(hasher, image_ids, gate):
    packed = np.packbits(gate.detach().cpu().numpy().astype(np.uint8), axis=-1)
    for image_id, row in zip(image_ids, packed):
        hasher.update(struct.pack("<q", int(image_id)))
        hasher.update(row.tobytes())


def _reference_audit():
    return {
        "coeff_min": {name: [float("inf")] * 3 for name in ("E", "F")},
        "coeff_max": {name: [float("-inf")] * 3 for name in ("E", "F")},
        "coeff_shapes": {name: [None] * 3 for name in ("E", "F")},
        "historical_coeff_max_abs": {"B": [None] * 3, "C": [None] * 3},
        "location_input_mutation_max_abs": 0.0,
        "attention_input_mutation_max_abs": 0.0,
    }


def run_explicit_gate_branch(
    ec,
    decoder_input,
    condition,
    gate,
    gt_xyxy,
    image_ids,
    tensor_subset,
    audit,
):
    """Replay the verified decoder path with an explicit all-point query gate."""
    from scripts.ablation.diag_cmp5L_adaptive_suppression import (
        run_suppressed,
        suppression_core,
    )
    from scripts.ablation.evaluate_cmp5L_af_validation import (
        FG,
        RING,
        FAR_BG,
        layer_query_statistics,
        region_mask_with_outside,
        sampled_value_tensor,
        _tensor_output,
    )

    modules = [ec.decoder.layers[index].cross_attn for index in range(3)]
    originals = [module.ms_deformable_attn_core for module in modules]
    handles = []
    batch_stats = [None, None, None]
    tensor_records = {}
    selected = {
        index: int(image_id)
        for index, image_id in enumerate(image_ids)
        if int(image_id) in tensor_subset
    }

    def record(image_id, layer, name, value):
        tensor_records.setdefault(str(image_id), {}).setdefault(condition, {}).setdefault(
            "layers", {}
        ).setdefault(str(layer), {})[name] = _stored_tensor(value)

    for layer_index, module in enumerate(modules):
        def pre_hook_factory(index):
            def hook(_module, inputs):
                for batch_index, image_id in selected.items():
                    record(image_id, index, "query_in", inputs[0][batch_index])
            return hook

        def post_hook_factory(index):
            def hook(_module, _inputs, output):
                value = _tensor_output(output)
                for batch_index, image_id in selected.items():
                    record(image_id, index, "query_out", value[batch_index])
            return hook

        handles.append(
            ec.decoder.layers[layer_index].register_forward_pre_hook(
                pre_hook_factory(layer_index)
            )
        )
        handles.append(
            ec.decoder.layers[layer_index].register_forward_hook(
                post_hook_factory(layer_index)
            )
        )

        def core_factory(index):
            def core(value, spatial_shapes, locations, attention, num_points_list):
                location_before = locations.detach().clone()
                attention_before = attention.detach().clone()
                regions, outside_flags = [], []
                for batch_index in range(locations.shape[0]):
                    region, outside = region_mask_with_outside(
                        locations[batch_index], gt_xyxy[batch_index]
                    )
                    regions.append(region)
                    outside_flags.append(outside)
                region = torch.stack(regions)
                outside = torch.stack(outside_flags)
                coeff = all_point_coeff(gate, locations.shape[2], locations.shape[3])
                stats = layer_query_statistics(attention, region, outside, coeff)
                batch_stats[index] = stats.detach().cpu().to(torch.float16).numpy()
                audit["coeff_min"][condition][index] = min(
                    audit["coeff_min"][condition][index], float(coeff.min())
                )
                audit["coeff_max"][condition][index] = max(
                    audit["coeff_max"][condition][index], float(coeff.max())
                )
                audit["coeff_shapes"][condition][index] = list(coeff.shape)

                if selected:
                    sampled = sampled_value_tensor(
                        value, spatial_shapes, locations, num_points_list
                    )
                    z_pre = (sampled * attention[..., None]).sum(-2).flatten(-2)
                    z_post = (
                        sampled * attention[..., None] * coeff[..., None]
                    ).sum(-2).flatten(-2)
                    z_delta = z_post - z_pre
                    for batch_index, image_id in selected.items():
                        record(image_id, index, "sampling_locations", locations[batch_index])
                        record(image_id, index, "attention_weights", attention[batch_index])
                        record(image_id, index, "region", region[batch_index])
                        record(image_id, index, "outside", outside[batch_index])
                        record(image_id, index, "coeff", coeff[batch_index])
                        record(image_id, index, "spatial_stats", stats[batch_index])
                        record(image_id, index, "z_pre_norm", z_pre[batch_index].norm(dim=-1))
                        record(image_id, index, "z_post_norm", z_post[batch_index].norm(dim=-1))
                        record(image_id, index, "z_delta_norm", z_delta[batch_index].norm(dim=-1))
                        for label, name in ((FG, "fg"), (RING, "ring"), (FAR_BG, "far_bg")):
                            contribution = (
                                sampled
                                * attention[..., None]
                                * (region == label)[..., None].to(sampled.dtype)
                            ).sum(-2).flatten(-2)
                            post_contribution = (
                                sampled
                                * attention[..., None]
                                * coeff[..., None]
                                * (region == label)[..., None].to(sampled.dtype)
                            ).sum(-2).flatten(-2)
                            record(
                                image_id,
                                index,
                                f"z_{name}_norm",
                                contribution[batch_index].norm(dim=-1),
                            )
                            record(
                                image_id,
                                index,
                                f"z_post_{name}_norm",
                                post_contribution[batch_index].norm(dim=-1),
                            )

                output = suppression_core(
                    value, spatial_shapes, locations, attention, num_points_list, coeff
                )
                audit["location_input_mutation_max_abs"] = max(
                    audit["location_input_mutation_max_abs"],
                    float((locations - location_before).abs().max()),
                )
                audit["attention_input_mutation_max_abs"] = max(
                    audit["attention_input_mutation_max_abs"],
                    float((attention - attention_before).abs().max()),
                )
                return output
            return core

        module.ms_deformable_attn_core = core_factory(layer_index)

    try:
        output = run_suppressed(
            ec,
            decoder_input["query_content"],
            decoder_input["reference_unact"],
            decoder_input["memory"],
            decoder_input["spatial_shapes"],
            set(),
            "NN",
            None,
            [],
            None,
            None,
            None,
        )
    finally:
        for handle in handles:
            handle.remove()
        for module, original in zip(modules, originals):
            module.ms_deformable_attn_core = original
    if any(
        module.ms_deformable_attn_core is not original
        for module, original in zip(modules, originals)
    ):
        raise RuntimeError("cross-attention core restoration failed")
    return output, batch_stats, tensor_records


def _merge_tensor_records(destination, source):
    for image_id, conditions in source.items():
        destination.setdefault(image_id, {}).update(conditions)


def _load_split(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    calibration = {int(item["image_id"]) for item in payload["calibration"]}
    evaluation = {int(item["image_id"]) for item in payload["evaluation"]}
    if calibration & evaluation:
        raise ValueError("H calibration/evaluation split overlaps")
    return payload, calibration, evaluation


def evaluate(args):
    from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy
    from scripts.ablation.evaluate_cmp5L_af_validation import (
        LAYER_STAT_NAMES,
        _normal_query_diagnostics,
        _prediction_record,
        _summarize_evaluator,
        checkpoint_key_audit,
        json_compatible_args,
        load_query_predictor,
        normal_bg_dependency,
        region_mask_with_outside,
        require_tensor_subset_coverage,
        run_af_branch,
        sha256_path,
        topk_query_predictions,
    )
    from scripts.ablation.probe_cmp5L_decoder_internal_tensors import (
        DecoderInternalCapture,
        _build,
        _normalise_targets,
    )

    mode = args.mode.upper()
    if mode not in ("G", "H"):
        raise ValueError("mode must be G or H")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    args.device = device
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    tensor_subset = set(
        map(int, json.loads(Path(args.tensor_subset).read_text())["image_ids"])
    )
    split_payload, calibration_ids, evaluation_ids = _load_split(args.split_file)
    dataset = json.loads(Path(args.ann_file).read_text(encoding="utf-8"))
    image_by_id = {int(image["id"]): image for image in dataset["images"]}
    annotations_by_image = defaultdict(list)
    for annotation in dataset["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    all_annotation_ids = {
        int(image_id): [int(annotation["id"]) for annotation in annotations]
        for image_id, annotations in annotations_by_image.items()
    }

    predictor, predictor_spec, predictor_load = load_query_predictor(
        args.query_predictor, device
    )
    cfg, solver, model = _build(args)
    model.eval()
    model_load = checkpoint_key_audit(model, args.checkpoint, args.weights)
    if model_load["missing_keys"] or model_load["unexpected_keys"]:
        raise RuntimeError(f"model checkpoint key mismatch: {model_load}")

    if mode == "G":
        conditions = ["A", "G0", "G1", "G2", "G3"]
    else:
        conditions = ["A", "H005", "H020", "H050", "H080", "H095"]
    evaluators = {condition: copy.deepcopy(cfg.evaluator) for condition in conditions}
    for evaluator in evaluators.values():
        evaluator.cleanup()
    streams = {
        condition: (output_dir / f"predictions_{condition}.jsonl").open(
            "x", encoding="utf-8"
        )
        for condition in conditions
    }
    audit = {
        "mode": mode,
        "baseline_A_max_abs": {"logits": 0.0, "boxes": 0.0},
        "normal_hook_max_abs": {"logits": 0.0, "boxes": 0.0},
        "A_repeat_after_branches_max_abs": None,
        "G0_reference_F_max_abs": {"logits": 0.0, "boxes": 0.0},
        "G3_reference_E_positive_max_abs": {"logits": 0.0, "boxes": 0.0},
        "G3_empty_G0_max_abs": {"logits": 0.0, "boxes": 0.0},
        "H005_reference_F_max_abs": {"logits": 0.0, "boxes": 0.0},
        "location_input_mutation_max_abs": 0.0,
        "attention_input_mutation_max_abs": 0.0,
        "coeff_min": {condition: [float("inf")] * 3 for condition in conditions},
        "coeff_max": {condition: [float("-inf")] * 3 for condition in conditions},
        "coeff_shapes": {condition: [None] * 3 for condition in conditions},
        "finite_outputs": {condition: True for condition in conditions},
        "predictor_load": predictor_load,
        "model_load": model_load,
        "model_eval": not model.training,
        "conditions": conditions,
        "direct_layers": [0, 1, 2],
        "gate_counts": defaultdict(int),
    }
    reference_audit = _reference_audit()
    gate_hashers = {condition: hashlib.sha256() for condition in conditions if condition != "A"}
    query_keys = (
        "image_id",
        "d_bg",
        "p_suppress",
        "g_gt",
        "g_pred",
        "false_suppress",
        "missed_suppress",
        "score",
        "top1_class",
        "margin",
        "best_gt_index",
        "best_iou",
        "gt_class_logit",
    )
    query_batches = {key: [] for key in query_keys}
    gate_batches = {condition: [] for condition in conditions if condition != "A"}
    layer_batches = {condition: [[] for _ in range(3)] for condition in conditions}
    raw_batches = {
        condition: {"logits": [], "boxes": []} for condition in conditions
    }
    tensor_records = {}
    seen_ids = []
    first_batch = True
    started = time.time()
    try:
        with torch.no_grad():
            for samples, targets in cfg.val_dataloader:
                if args.limit and len(seen_ids) >= args.limit:
                    break
                samples = samples.to(device)
                image_ids = [int(target["image_id"].item()) for target in targets]
                matcher_targets = _normalise_targets(
                    targets, samples.shape[-2], samples.shape[-1], device
                )
                gt_xyxy = [
                    box_cxcywh_to_xyxy(target["boxes"]) for target in matcher_targets
                ]
                gt_labels = [target["labels"] for target in matcher_targets]
                has_gt = torch.tensor(
                    [boxes.shape[0] > 0 for boxes in gt_xyxy], device=device
                )
                baseline = model(samples)
                with DecoderInternalCapture(model.decoder) as capture:
                    normal = model(samples)
                for key in ("logits", "boxes"):
                    output_key = f"pred_{key}"
                    audit["normal_hook_max_abs"][key] = max(
                        audit["normal_hook_max_abs"][key],
                        float((baseline[output_key] - normal[output_key]).abs().max()),
                    )

                normal_regions, normal_attention = [], []
                for layer_index in range(3):
                    record = capture.layers[layer_index]
                    normal_attention.append(record["attention_weights"])
                    normal_regions.append(
                        torch.stack(
                            [
                                region_mask_with_outside(
                                    record["sampling_locations"][batch_index],
                                    gt_xyxy[batch_index],
                                )[0]
                                for batch_index in range(len(image_ids))
                            ]
                        )
                    )
                d_bg = normal_bg_dependency(normal_attention, normal_regions)
                g_gt = d_bg > float(predictor_spec["bgdep_tau"])
                q_l1 = capture.layers[1]["query_in"].float()
                p_suppress = torch.sigmoid(
                    predictor((q_l1 - predictor_spec["mean"]) / predictor_spec["std"])
                )
                g_pred = p_suppress > float(predictor_spec["threshold"])
                if mode == "G":
                    gates = build_g_gates(g_gt, g_pred, has_gt)
                    changes = gate_change_audit(gates, g_gt, g_pred, has_gt)
                    for key, value in changes.items():
                        audit["gate_counts"][key] += int(value)
                else:
                    gates = build_h_gates(p_suppress)
                    if not torch.equal(gates["H005"], g_pred):
                        raise RuntimeError("H005 does not reproduce stored 0.05 predictor gate")
                gates = {"A": torch.zeros_like(g_pred), **gates}
                for condition, gate in gates.items():
                    if condition == "A":
                        continue
                    gate_batches[condition].append(gate.detach().cpu().numpy())
                    _update_gate_hash(gate_hashers[condition], image_ids, gate)
                    audit["gate_counts"][f"{condition}_active"] += int(gate.sum())

                diagnostics = _normal_query_diagnostics(normal, gt_xyxy, gt_labels)
                repeated_ids = torch.tensor(image_ids)[:, None].expand(-1, d_bg.shape[1])
                payloads = {
                    "image_id": repeated_ids,
                    "d_bg": d_bg,
                    "p_suppress": p_suppress,
                    "g_gt": g_gt,
                    "g_pred": g_pred,
                    "false_suppress": has_gt[:, None] & g_pred & ~g_gt,
                    "missed_suppress": has_gt[:, None] & ~g_pred & g_gt,
                    **diagnostics,
                }
                for key, value in payloads.items():
                    query_batches[key].append(value.detach().cpu().numpy())

                decoder_input = capture.decoder_input
                for batch_index, image_id in enumerate(image_ids):
                    if image_id not in tensor_subset:
                        continue
                    shared = tensor_records.setdefault(str(image_id), {}).setdefault(
                        "shared", {}
                    )
                    shared.update(
                        {
                            "query_idx": torch.arange(d_bg.shape[1], dtype=torch.int16),
                            "q_l1": _stored_tensor(q_l1[batch_index]),
                            "query_content": _stored_tensor(
                                decoder_input["query_content"][batch_index]
                            ),
                            "initial_reference": _stored_tensor(
                                decoder_input["reference_unact"][batch_index].sigmoid()
                            ),
                            "d_bg": _stored_tensor(d_bg[batch_index]),
                            "p_suppress": _stored_tensor(p_suppress[batch_index]),
                            "g_gt": _stored_tensor(g_gt[batch_index]),
                            "g_pred": _stored_tensor(g_pred[batch_index]),
                        }
                    )
                    for condition, gate in gates.items():
                        shared[f"gate_{condition}"] = _stored_tensor(gate[batch_index])
                    for layer_index in range(3):
                        shared.setdefault("normal_layers", {})[str(layer_index)] = {
                            "query_in": _stored_tensor(
                                capture.layers[layer_index]["query_in"][batch_index]
                            ),
                            "sampling_locations": _stored_tensor(
                                capture.layers[layer_index]["sampling_locations"][batch_index]
                            ),
                            "attention_weights": _stored_tensor(
                                capture.layers[layer_index]["attention_weights"][batch_index]
                            ),
                            "region": _stored_tensor(normal_regions[layer_index][batch_index]),
                        }

                outputs = {}
                for condition in conditions:
                    branch, statistics, tensors = run_explicit_gate_branch(
                        model.decoder,
                        decoder_input,
                        condition,
                        gates[condition],
                        gt_xyxy,
                        image_ids,
                        tensor_subset,
                        audit,
                    )
                    outputs[condition] = branch
                    _merge_tensor_records(tensor_records, tensors)
                    for layer_index, statistics_for_layer in enumerate(statistics):
                        layer_batches[condition][layer_index].append(statistics_for_layer)
                    for name in ("logits", "boxes"):
                        raw_batches[condition][name].append(
                            branch[f"pred_{name}"].detach().cpu().to(torch.float16).numpy()
                        )
                    audit["finite_outputs"][condition] &= bool(
                        torch.isfinite(branch["pred_logits"]).all()
                        and torch.isfinite(branch["pred_boxes"]).all()
                    )
                for key in ("logits", "boxes"):
                    output_key = f"pred_{key}"
                    audit["baseline_A_max_abs"][key] = max(
                        audit["baseline_A_max_abs"][key],
                        float((outputs["A"][output_key] - normal[output_key]).abs().max()),
                    )

                reference_f, _, _ = run_af_branch(
                    model.decoder,
                    decoder_input,
                    "F",
                    gt_xyxy,
                    has_gt,
                    g_gt,
                    g_pred,
                    image_ids,
                    set(),
                    reference_audit,
                )
                primary = "G0" if mode == "G" else "H005"
                parity_key = (
                    "G0_reference_F_max_abs"
                    if mode == "G"
                    else "H005_reference_F_max_abs"
                )
                for key in ("logits", "boxes"):
                    output_key = f"pred_{key}"
                    audit[parity_key][key] = max(
                        audit[parity_key][key],
                        float((outputs[primary][output_key] - reference_f[output_key]).abs().max()),
                    )

                if mode == "G":
                    reference_e, _, _ = run_af_branch(
                        model.decoder,
                        decoder_input,
                        "E",
                        gt_xyxy,
                        has_gt,
                        g_gt,
                        g_pred,
                        image_ids,
                        set(),
                        reference_audit,
                    )
                    positive = has_gt.nonzero(as_tuple=False).flatten()
                    empty = (~has_gt).nonzero(as_tuple=False).flatten()
                    for key in ("logits", "boxes"):
                        output_key = f"pred_{key}"
                        if positive.numel():
                            audit["G3_reference_E_positive_max_abs"][key] = max(
                                audit["G3_reference_E_positive_max_abs"][key],
                                float(
                                    (
                                        outputs["G3"][output_key][positive]
                                        - reference_e[output_key][positive]
                                    ).abs().max()
                                ),
                            )
                        if empty.numel():
                            audit["G3_empty_G0_max_abs"][key] = max(
                                audit["G3_empty_G0_max_abs"][key],
                                float(
                                    (
                                        outputs["G3"][output_key][empty]
                                        - outputs["G0"][output_key][empty]
                                    ).abs().max()
                                ),
                            )

                if first_batch:
                    repeated_a, _, _ = run_explicit_gate_branch(
                        model.decoder,
                        decoder_input,
                        "A",
                        gates["A"],
                        gt_xyxy,
                        image_ids,
                        set(),
                        audit,
                    )
                    audit["A_repeat_after_branches_max_abs"] = {
                        key: float(
                            (
                                repeated_a[f"pred_{key}"]
                                - outputs["A"][f"pred_{key}"]
                            ).abs().max()
                        )
                        for key in ("logits", "boxes")
                    }
                    first_batch = False

                original_sizes = torch.stack(
                    [target["orig_size"] for target in targets]
                ).to(device)
                for condition, branch in outputs.items():
                    official = solver.postprocessor(branch, original_sizes)
                    identified = topk_query_predictions(branch, original_sizes, topk=300)
                    if not seen_ids:
                        for left, right in zip(official, identified):
                            if not (
                                torch.equal(left["labels"], right["labels"])
                                and torch.allclose(left["scores"], right["scores"])
                                and torch.allclose(left["boxes"], right["boxes"])
                            ):
                                raise RuntimeError(
                                    "query-identity postprocess differs from official postprocessor"
                                )
                    evaluators[condition].update(
                        {
                            image_id: result
                            for image_id, result in zip(image_ids, official)
                        }
                    )
                    for image_id, result in zip(image_ids, identified):
                        streams[condition].write(
                            json.dumps(_prediction_record(image_id, result)) + "\n"
                        )
                seen_ids.extend(image_ids)
                if args.log_every and len(seen_ids) % args.log_every < len(image_ids):
                    print(
                        f"[{len(seen_ids)}/{len(cfg.val_dataloader.dataset)}] "
                        f"{time.time()-started:.1f}s",
                        flush=True,
                    )
    finally:
        for stream in streams.values():
            stream.close()

    expected = args.limit or args.expected_images
    if len(seen_ids) != expected or len(set(seen_ids)) != len(seen_ids):
        raise RuntimeError(
            f"image coverage failed: seen={len(seen_ids)}, "
            f"unique={len(set(seen_ids))}, expected={expected}"
        )
    if not args.limit:
        split_union = calibration_ids | evaluation_ids
        if split_union != set(seen_ids):
            raise RuntimeError("locked H split does not cover the full validation images")
    parity = [
        *audit["baseline_A_max_abs"].values(),
        *audit["normal_hook_max_abs"].values(),
        *audit["A_repeat_after_branches_max_abs"].values(),
        audit["location_input_mutation_max_abs"],
        audit["attention_input_mutation_max_abs"],
        reference_audit["location_input_mutation_max_abs"],
        reference_audit["attention_input_mutation_max_abs"],
    ]
    if mode == "G":
        parity.extend(audit["G0_reference_F_max_abs"].values())
        parity.extend(audit["G3_reference_E_positive_max_abs"].values())
        parity.extend(audit["G3_empty_G0_max_abs"].values())
    else:
        parity.extend(audit["H005_reference_F_max_abs"].values())
    if max(parity) > 2e-6 or not all(audit["finite_outputs"].values()):
        raise RuntimeError(f"technical gate failed: {audit}")

    metrics = {
        condition: _summarize_evaluator(evaluator)
        for condition, evaluator in evaluators.items()
    }
    query_arrays = {
        key: np.concatenate(values, axis=0) for key, values in query_batches.items()
    }
    np.savez_compressed(
        output_dir / "query_diagnostics.npz",
        query_idx=np.arange(query_arrays["d_bg"].shape[1], dtype=np.int16),
        **query_arrays,
        **{
            f"gate_{condition}": np.concatenate(values, axis=0)
            for condition, values in gate_batches.items()
        },
    )
    np.savez_compressed(
        output_dir / "layer_query_stats.npz",
        stat_names=np.asarray(LAYER_STAT_NAMES),
        image_ids=np.asarray(seen_ids, dtype=np.int64),
        **{
            f"{condition}_L{layer_index}": np.concatenate(
                layer_batches[condition][layer_index], axis=0
            )
            for condition in conditions
            for layer_index in range(3)
        },
    )
    np.savez_compressed(
        output_dir / "raw_query_outputs.npz",
        image_ids=np.asarray(seen_ids, dtype=np.int64),
        **{
            f"{condition}_{name}": np.concatenate(raw_batches[condition][name], axis=0)
            for condition in conditions
            for name in ("logits", "boxes")
        },
    )
    torch.save(tensor_records, output_dir / "fixed_tensor_subset.pt")
    (output_dir / "metrics_full.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "image_manifest.jsonl").open("x", encoding="utf-8") as handle:
        for image_id in seen_ids:
            annotations = annotations_by_image[image_id]
            handle.write(
                json.dumps(
                    {
                        "image_id": image_id,
                        "file_name": image_by_id[image_id]["file_name"],
                        "gt_count": len(annotations),
                        "has_duct": any(
                            int(annotation["category_id"]) == 3
                            for annotation in annotations
                        ),
                        "annotation_ids": all_annotation_ids.get(image_id, []),
                        "split": (
                            "calibration"
                            if image_id in calibration_ids
                            else "evaluation"
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    audit["gate_counts"] = dict(audit["gate_counts"])
    audit.update(
        {
            "n_images": len(seen_ids),
            "unique_image_ids": len(set(seen_ids)),
            "tensor_subset_requested": sorted(tensor_subset),
            "tensor_subset_observed": sorted(map(int, tensor_records)),
            "gate_identity_sha256": {
                condition: hasher.hexdigest()
                for condition, hasher in gate_hashers.items()
            },
            "elapsed_seconds": round(time.time() - started, 2),
        }
    )
    (output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if not args.limit:
        require_tensor_subset_coverage(tensor_subset, tensor_records)
    manifest = {
        "scope": f"cmp5L {mode} frozen-original-EMA validation only",
        "args": json_compatible_args(args),
        "conditions": conditions,
        "condition_definitions": (
            {
                "A": "identity",
                "G0": "original predictor gate; all points",
                "G1": "G0 with positive-image false-suppress corrected",
                "G2": "G0 with positive-image missed-suppress corrected",
                "G3": "G0 with both errors corrected; equals GT gate only on positive images",
            }
            if mode == "G"
            else {
                "A": "identity",
                **{
                    _threshold_name(threshold): f"p_BG > {threshold:.2f}; all points"
                    for threshold in H_THRESHOLDS
                },
            }
        ),
        "hashes": {
            "runner": sha256_path(__file__),
            "checkpoint": sha256_path(args.checkpoint),
            "predictor": sha256_path(args.query_predictor),
            "config": sha256_path(args.config),
            "annotations": sha256_path(args.ann_file),
            "split": sha256_path(args.split_file),
            "tensor_subset": sha256_path(args.tensor_subset),
            "af_manifest": sha256_path(Path(args.af_result_dir) / "manifest.json"),
            "af_query_diagnostics": sha256_path(
                Path(args.af_result_dir) / "query_diagnostics.npz"
            ),
        },
        "predictor": {
            "variant": predictor_spec["variant"],
            "input_tensor": "Normal capture.layers[1]['query_in']",
            "input_timing": "actual decoder index 1 input; after L0",
            "normal_forward_required": True,
            "output_semantics": "p_BG=P(d_BG>0.20)",
            "stored_threshold": float(predictor_spec["threshold"]),
        },
        "h_thresholds": list(H_THRESHOLDS),
        "lambda_min": LAMBDA_MIN,
        "direct_layers": [0, 1, 2],
        "spatial_scope": "all heads and all sampled value points",
        "attention_renormalized": False,
        "split_summary": {
            "seed": split_payload["seed"],
            "unit": split_payload["unit"],
            "calibration_images": len(calibration_ids),
            "evaluation_images": len(evaluation_ids),
            "independent_generalization_estimate": False,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "COMPLETED.txt").write_text("status=passed\n", encoding="utf-8")
    print(
        json.dumps({"audit": audit, "metrics": metrics}, ensure_ascii=False)[:20000],
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("G", "H"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--query-predictor", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--img-folder", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--tensor-subset", required=True)
    parser.add_argument("--af-result-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", choices=("ema",), default="ema")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--expected-images", type=int, default=2975)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
