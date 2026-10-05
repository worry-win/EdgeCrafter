"""Privileged query-behavior distillation primitives for cmp5L.

The module is intentionally independent from the default detector forward path.
Training entrypoints opt in explicitly; importing it cannot change ECDet behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy


@dataclass
class DecoderReplayInputs:
    initial_query: torch.Tensor
    initial_reference_unactivated: torch.Tensor
    memory: torch.Tensor
    spatial_shapes: object
    attention_mask: torch.Tensor | None
    denoising_metadata: object
    normal_query_count: int


def _region_mask(
    locations: torch.Tensor,
    gt_xyxy: torch.Tensor,
    ring_scale: float,
) -> torch.Tensor:
    """Return 2=GT, 1=ring, 0=far background for ``[Q,H,P,2]`` points."""

    if gt_xyxy.numel() == 0:
        return torch.zeros(
            locations.shape[:-1], dtype=torch.long, device=locations.device
        )
    flat = locations.reshape(-1, 2)
    x_coord, y_coord = flat.unbind(-1)
    inside = torch.zeros(flat.shape[0], dtype=torch.bool, device=flat.device)
    in_ring = torch.zeros_like(inside)
    for box in gt_xyxy:
        x0, y0, x1, y1 = box
        inside |= (
            (x_coord >= x0)
            & (x_coord <= x1)
            & (y_coord >= y0)
            & (y_coord <= y1)
        )
        center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
        half_width = (x1 - x0) * ring_scale / 2
        half_height = (y1 - y0) * ring_scale / 2
        in_ring |= (
            (x_coord >= center_x - half_width)
            & (x_coord <= center_x + half_width)
            & (y_coord >= center_y - half_height)
            & (y_coord <= center_y + half_height)
        )
    region = torch.where(
        inside,
        torch.full_like(inside, 2, dtype=torch.long),
        torch.where(
            in_ring & ~inside,
            torch.ones_like(inside, dtype=torch.long),
            torch.zeros_like(inside, dtype=torch.long),
        ),
    )
    return region.reshape(locations.shape[:-1])


def adaptive_farbg_coefficients(
    sampling_locations_by_layer: Sequence[torch.Tensor],
    attention_weights_by_layer: Sequence[torch.Tensor],
    gt_xyxy_by_image: Sequence[torch.Tensor],
    *,
    threshold: float = 0.20,
    far_bg_scale: float = 0.20,
    ring_scale: float = 1.50,
):
    """Reproduce the validated L0-L2 adaptive far-BG Oracle.

    Dependency is the mean far-background attention mass over the supplied
    normal layers.  Images without GT keep all-one coefficients.
    """

    if not sampling_locations_by_layer:
        raise ValueError("at least one decoder layer is required")
    if len(sampling_locations_by_layer) != len(attention_weights_by_layer):
        raise ValueError("location/attention layer counts differ")

    batch, queries = sampling_locations_by_layer[0].shape[:2]
    regions_by_layer = []
    dependencies = []
    for locations, weights in zip(
        sampling_locations_by_layer, attention_weights_by_layer
    ):
        layer_regions = []
        layer_dependency = []
        for batch_index in range(batch):
            region = _region_mask(
                locations[batch_index],
                gt_xyxy_by_image[batch_index],
                ring_scale,
            )
            layer_regions.append(region)
            if gt_xyxy_by_image[batch_index].numel() == 0:
                layer_dependency.append(
                    torch.zeros(queries, device=weights.device, dtype=weights.dtype)
                )
            else:
                layer_dependency.append(
                    (weights[batch_index] * (region == 0).to(weights.dtype))
                    .sum(-1)
                    .mean(-1)
                )
        regions_by_layer.append(torch.stack(layer_regions))
        dependencies.append(torch.stack(layer_dependency))

    bg_dependency = torch.stack(dependencies).mean(0)
    high_dependency = bg_dependency > threshold
    coefficients = []
    for layer_index, weights in enumerate(attention_weights_by_layer):
        coeff = torch.ones_like(weights)
        far_bg = regions_by_layer[layer_index] == 0
        suppress = high_dependency[:, :, None, None] & far_bg
        coeff = torch.where(
            suppress,
            torch.as_tensor(far_bg_scale, dtype=coeff.dtype, device=coeff.device),
            coeff,
        )
        coefficients.append(coeff)
    return coefficients, bg_dependency


def _paired_iou(cxcywh_a: torch.Tensor, cxcywh_b: torch.Tensor) -> torch.Tensor:
    a = box_cxcywh_to_xyxy(cxcywh_a)
    b = box_cxcywh_to_xyxy(cxcywh_b)
    left_top = torch.maximum(a[:, :2], b[:, :2])
    right_bottom = torch.minimum(a[:, 2:], b[:, 2:])
    intersection = (right_bottom - left_top).clamp(min=0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(-1)
    return intersection / (area_a + area_b - intersection).clamp(min=1e-9)


def _class_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    gt_logit = logits.gather(-1, labels[:, None]).squeeze(-1)
    if logits.shape[-1] == 1:
        return gt_logit
    other = logits.clone()
    other.scatter_(-1, labels[:, None], float("-inf"))
    return gt_logit - other.max(-1).values


def compute_behavior_kd_losses(
    student_trace,
    teacher_trace,
    student_predictions,
    teacher_predictions,
    targets,
    matches,
    *,
    distill_layers=(0, 1, 2),
):
    """Compute unweighted query, negative, and relative-sampling KD losses.

    Positive query updates use the union teacher-better gate from GT-class
    margin and paired IoU.  Unmatched query scores use the asymmetric
    ``relu(student - teacher)^2`` objective.  Teacher operands are detached at
    this boundary even if a caller accidentally supplies grad-enabled tensors.
    """

    student_logits = student_predictions["pred_logits"].float()
    student_boxes = student_predictions["pred_boxes"].float()
    teacher_logits = teacher_predictions["pred_logits"].float().detach()
    teacher_boxes = teacher_predictions["pred_boxes"].float().detach()
    graph_zero = student_logits.sum() * 0.0

    query_terms = []
    sampling_terms = []
    negative_terms = []
    teacher_better_count = 0
    negative_count = 0

    for batch_index, (source_indices, target_indices) in enumerate(matches):
        source_indices = source_indices.to(student_logits.device)
        target_indices = target_indices.to(student_logits.device)
        matched_labels = targets[batch_index]["labels"][target_indices]
        matched_boxes = targets[batch_index]["boxes"][target_indices]

        if source_indices.numel():
            student_matched_logits = student_logits[batch_index, source_indices]
            teacher_matched_logits = teacher_logits[batch_index, source_indices]
            student_margin = _class_margin(student_matched_logits, matched_labels)
            teacher_margin = _class_margin(teacher_matched_logits, matched_labels)
            student_iou = _paired_iou(
                student_boxes[batch_index, source_indices], matched_boxes
            )
            teacher_iou = _paired_iou(
                teacher_boxes[batch_index, source_indices], matched_boxes
            )
            teacher_better = (teacher_margin > student_margin) | (
                teacher_iou > student_iou
            )
            kept_sources = source_indices[teacher_better]
            teacher_better_count += int(teacher_better.sum().item())

            if kept_sources.numel():
                for layer_index in distill_layers:
                    student_delta = (
                        student_trace["query_outputs"][layer_index][
                            batch_index, kept_sources
                        ]
                        - student_trace["query_inputs"][layer_index][
                            batch_index, kept_sources
                        ]
                    ).float()
                    teacher_delta = (
                        teacher_trace["query_outputs"][layer_index][
                            batch_index, kept_sources
                        ]
                        - teacher_trace["query_inputs"][layer_index][
                            batch_index, kept_sources
                        ]
                    ).float().detach()
                    query_terms.append(
                        1.0
                        - F.cosine_similarity(
                            student_delta, teacher_delta, dim=-1, eps=1e-8
                        )
                    )

                    if (
                        "relative_sampling" in student_trace
                        and "relative_sampling" in teacher_trace
                    ):
                        student_sampling = student_trace["relative_sampling"][
                            layer_index
                        ][batch_index, kept_sources].float()
                        teacher_sampling = teacher_trace["relative_sampling"][
                            layer_index
                        ][batch_index, kept_sources].float().detach()
                        sampling_terms.append(
                            F.smooth_l1_loss(
                                student_sampling,
                                teacher_sampling,
                                reduction="none",
                            ).flatten(1).mean(-1)
                        )

        query_count = student_logits.shape[1]
        negative_mask = torch.ones(
            query_count, dtype=torch.bool, device=student_logits.device
        )
        negative_mask[source_indices] = False
        student_negative_score = student_logits[batch_index, negative_mask].sigmoid().max(-1).values
        teacher_negative_score = teacher_logits[batch_index, negative_mask].sigmoid().max(-1).values
        negative_terms.append(
            F.relu(student_negative_score - teacher_negative_score).square()
        )
        negative_count += int(negative_mask.sum().item())

    def mean_or_zero(terms):
        nonempty = [term.reshape(-1) for term in terms if term.numel()]
        return torch.cat(nonempty).mean() if nonempty else graph_zero

    losses = {
        "loss_query_update": mean_or_zero(query_terms),
        "loss_negative_behavior": mean_or_zero(negative_terms),
        "loss_sampling": mean_or_zero(sampling_terms),
    }
    stats = {
        "teacher_better_count": teacher_better_count,
        "negative_count": negative_count,
    }
    return losses, stats


@torch.no_grad()
def summarize_behavior_batch(
    student_trace,
    teacher_trace,
    student_predictions,
    teacher_predictions,
    targets,
    matches,
    *,
    distill_layers=(0, 1, 2),
    high_confidence=0.5,
):
    """Return additive mechanism statistics for one evaluation batch.

    Negatives are Hungarian-unmatched normal queries. Positives are matched
    queries paired to their GT target. All fields are sums plus explicit counts
    so callers can aggregate batches without averaging averages.
    """

    student_logits = student_predictions["pred_logits"].float()
    student_boxes = student_predictions["pred_boxes"].float()
    teacher_logits = teacher_predictions["pred_logits"].float()
    teacher_boxes = teacher_predictions["pred_boxes"].float()
    summary = {
        "negative": {
            "count": 0,
            "student_score_sum": 0.0,
            "teacher_score_sum": 0.0,
            "student_high_confidence_count": 0,
            "teacher_high_confidence_count": 0,
        },
        "positive": {
            "count": 0,
            "student_gt_probability_sum": 0.0,
            "teacher_gt_probability_sum": 0.0,
            "student_margin_sum": 0.0,
            "teacher_margin_sum": 0.0,
            "student_iou_sum": 0.0,
            "teacher_iou_sum": 0.0,
        },
        "layers": {
            str(layer): {
                "count": 0,
                "delta_cosine_sum": 0.0,
                "relative_sampling_l1_sum": 0.0,
            }
            for layer in distill_layers
        },
    }

    for batch_index, (source_indices, target_indices) in enumerate(matches):
        device = student_logits.device
        source_indices = source_indices.to(device)
        target_indices = target_indices.to(device)
        negative_mask = torch.ones(
            student_logits.shape[1], dtype=torch.bool, device=device
        )
        negative_mask[source_indices] = False
        student_negative = (
            student_logits[batch_index, negative_mask].sigmoid().max(-1).values
        )
        teacher_negative = (
            teacher_logits[batch_index, negative_mask].sigmoid().max(-1).values
        )
        negative = summary["negative"]
        negative["count"] += int(student_negative.numel())
        negative["student_score_sum"] += float(student_negative.sum())
        negative["teacher_score_sum"] += float(teacher_negative.sum())
        negative["student_high_confidence_count"] += int(
            (student_negative >= high_confidence).sum()
        )
        negative["teacher_high_confidence_count"] += int(
            (teacher_negative >= high_confidence).sum()
        )

        if not source_indices.numel():
            continue
        labels = targets[batch_index]["labels"][target_indices]
        gt_boxes = targets[batch_index]["boxes"][target_indices]
        student_matched_logits = student_logits[batch_index, source_indices]
        teacher_matched_logits = teacher_logits[batch_index, source_indices]
        student_gt_probability = student_matched_logits.sigmoid().gather(
            -1, labels[:, None]
        ).squeeze(-1)
        teacher_gt_probability = teacher_matched_logits.sigmoid().gather(
            -1, labels[:, None]
        ).squeeze(-1)
        positive = summary["positive"]
        positive["count"] += int(source_indices.numel())
        positive["student_gt_probability_sum"] += float(
            student_gt_probability.sum()
        )
        positive["teacher_gt_probability_sum"] += float(
            teacher_gt_probability.sum()
        )
        positive["student_margin_sum"] += float(
            _class_margin(student_matched_logits, labels).sum()
        )
        positive["teacher_margin_sum"] += float(
            _class_margin(teacher_matched_logits, labels).sum()
        )
        positive["student_iou_sum"] += float(
            _paired_iou(student_boxes[batch_index, source_indices], gt_boxes).sum()
        )
        positive["teacher_iou_sum"] += float(
            _paired_iou(teacher_boxes[batch_index, source_indices], gt_boxes).sum()
        )

        for layer in distill_layers:
            student_delta = (
                student_trace["query_outputs"][layer][batch_index, source_indices]
                - student_trace["query_inputs"][layer][batch_index, source_indices]
            ).float()
            teacher_delta = (
                teacher_trace["query_outputs"][layer][batch_index, source_indices]
                - teacher_trace["query_inputs"][layer][batch_index, source_indices]
            ).float()
            layer_summary = summary["layers"][str(layer)]
            layer_summary["count"] += int(source_indices.numel())
            layer_summary["delta_cosine_sum"] += float(
                F.cosine_similarity(
                    student_delta, teacher_delta, dim=-1, eps=1e-8
                ).sum()
            )
            student_sampling = student_trace["relative_sampling"][layer][
                batch_index, source_indices
            ].float()
            teacher_sampling = teacher_trace["relative_sampling"][layer][
                batch_index, source_indices
            ].float()
            layer_summary["relative_sampling_l1_sum"] += float(
                (student_sampling - teacher_sampling)
                .abs()
                .flatten(1)
                .mean(-1)
                .sum()
            )
    return summary


def _relative_sampling(
    locations: torch.Tensor,
    references: torch.Tensor,
    points_per_level,
) -> torch.Tensor:
    """Express sampled points relative to the current reference box."""

    if references.shape[-1] == 2:
        if references.shape[2] == 1:
            center = references[:, :, None, 0, None, :]
            return locations - center
        pieces = []
        for level, locations_level in enumerate(
            locations.split(points_per_level, dim=-2)
        ):
            center = references[:, :, None, level, None, :]
            pieces.append(locations_level - center)
        return torch.cat(pieces, dim=-2)

    if references.shape[2] == 1:
        center = references[:, :, None, 0, None, :2]
        size = references[:, :, None, 0, None, 2:].clamp(min=1e-6)
        return (locations - center) / size

    pieces = []
    for level, locations_level in enumerate(
        locations.split(points_per_level, dim=-2)
    ):
        center = references[:, :, None, level, None, :2]
        size = references[:, :, None, level, None, 2:].clamp(min=1e-6)
        pieces.append((locations_level - center) / size)
    return torch.cat(pieces, dim=-2)


def _run_decoder_with_capture(
    ec_transformer,
    replay_inputs: DecoderReplayInputs,
    suppress_coefficients=None,
    decoder_forward=None,
):
    decoder = ec_transformer.decoder
    if decoder_forward is None:
        decoder_forward = decoder.forward
    records = [dict() for _ in decoder.layers]
    originals = []

    for layer_index, layer in enumerate(decoder.layers):
        original_core = layer.cross_attn.ms_deformable_attn_core
        original_forward = layer.forward
        originals.append((layer, original_forward, original_core))

        def core_wrapper(
            value,
            spatial_shapes,
            sampling_locations,
            attention_weights,
            num_points_list,
            *,
            _index=layer_index,
            _normal_core=original_core,
        ):
            records[_index]["sampling_locations"] = sampling_locations
            records[_index]["attention_weights"] = attention_weights
            if suppress_coefficients is None or _index >= len(suppress_coefficients):
                return _normal_core(
                    value,
                    spatial_shapes,
                    sampling_locations,
                    attention_weights,
                    num_points_list,
                )
            coefficient = suppress_coefficients[_index]
            return sampled_value_suppression_core(
                value,
                spatial_shapes,
                sampling_locations,
                attention_weights,
                num_points_list,
                coefficient,
            )

        def layer_wrapper(
            *args,
            _index=layer_index,
            _forward=original_forward,
            _points=tuple(layer.cross_attn.num_points_list),
            **kwargs,
        ):
            query_input = args[0] if args else kwargs["target"]
            references = args[1] if len(args) > 1 else kwargs["reference_points"]
            records[_index]["query_input"] = query_input
            records[_index]["reference"] = references
            query_output = _forward(*args, **kwargs)
            records[_index]["query_output"] = query_output
            locations = records[_index].get("sampling_locations")
            if locations is not None:
                records[_index]["relative_sampling"] = _relative_sampling(
                    locations,
                    references,
                    _points,
                )
            return query_output

        layer.cross_attn.ms_deformable_attn_core = core_wrapper
        layer.forward = layer_wrapper

    try:
        decoder_outputs = decoder_forward(
            None,
            replay_inputs.initial_query,
            replay_inputs.initial_reference_unactivated,
            replay_inputs.memory,
            replay_inputs.spatial_shapes,
            ec_transformer.dec_bbox_head,
            ec_transformer.dec_score_head,
            ec_transformer.query_pos_head,
            ec_transformer.pre_bbox_head,
            ec_transformer.integral,
            ec_transformer.up,
            ec_transformer.reg_scale,
            attn_mask=replay_inputs.attention_mask,
            dn_meta=replay_inputs.denoising_metadata,
            continuous_bbox_head=ec_transformer.continuous_bbox_head,
        )
    finally:
        for layer, original_forward, original_core in originals:
            layer.forward = original_forward
            layer.cross_attn.ms_deformable_attn_core = original_core

    trace = {
        "query_inputs": [record["query_input"] for record in records],
        "query_outputs": [record["query_output"] for record in records],
        "sampling_locations": [record["sampling_locations"] for record in records],
        "attention_weights": [record["attention_weights"] for record in records],
        "relative_sampling": [record["relative_sampling"] for record in records],
    }
    return decoder_outputs, trace


def _slice_normal_trace(trace, normal_query_count):
    return {
        key: [tensor[:, -normal_query_count:] for tensor in values]
        for key, values in trace.items()
    }


def _predictions_from_decoder_outputs(decoder_outputs, normal_query_count):
    boxes, logits = decoder_outputs[0], decoder_outputs[1]
    return {
        "pred_logits": logits[-1, :, -normal_query_count:],
        "pred_boxes": boxes[-1, :, -normal_query_count:],
    }


def _run_forward_with_capture(ec_transformer, forward_callable):
    captured = {}
    original_decoder_forward = ec_transformer.decoder.forward

    def decoder_forward_wrapper(*args, **kwargs):
        captured["replay_inputs"] = DecoderReplayInputs(
            initial_query=args[1],
            initial_reference_unactivated=args[2],
            memory=args[3],
            spatial_shapes=args[4],
            attention_mask=kwargs.get("attn_mask"),
            denoising_metadata=kwargs.get("dn_meta"),
            normal_query_count=ec_transformer.num_queries,
        )
        replay = captured["replay_inputs"]
        decoder_outputs, trace = _run_decoder_with_capture(
            ec_transformer,
            replay,
            decoder_forward=original_decoder_forward,
        )
        captured["trace"] = trace
        return decoder_outputs

    ec_transformer.decoder.forward = decoder_forward_wrapper
    try:
        predictions = forward_callable()
    finally:
        ec_transformer.decoder.forward = original_decoder_forward

    normal_query_count = predictions["pred_logits"].shape[1]
    captured["replay_inputs"].normal_query_count = normal_query_count
    trace = _slice_normal_trace(captured["trace"], normal_query_count)
    return predictions, trace, captured["replay_inputs"]


def run_student_with_capture(ec_transformer, features, targets):
    """Run an ECTransformer forward while capturing decoder replay inputs."""

    return _run_forward_with_capture(
        ec_transformer,
        lambda: ec_transformer(features, targets),
    )


def run_model_with_capture(model, samples, targets):
    """Run the unchanged full Student/DDP forward with decoder capture."""

    student_module = model.module if hasattr(model, "module") else model
    ec_transformer = student_module.decoder
    return _run_forward_with_capture(
        ec_transformer,
        lambda: model(samples, targets=targets),
    )


def replay_frozen_teacher(
    teacher_ec_transformer,
    replay_inputs: DecoderReplayInputs,
    targets,
    *,
    privileged=True,
    distill_layers=(0, 1, 2),
    dependency_threshold=0.20,
    far_bg_scale=0.20,
    ring_scale=1.50,
):
    """Replay a frozen decoder from the exact Student-normal q0/r0/memory."""

    teacher_ec_transformer.eval()
    teacher_ec_transformer.requires_grad_(False)
    detached_inputs = DecoderReplayInputs(
        initial_query=replay_inputs.initial_query.detach(),
        initial_reference_unactivated=(
            replay_inputs.initial_reference_unactivated.detach()
        ),
        memory=replay_inputs.memory.detach(),
        spatial_shapes=replay_inputs.spatial_shapes,
        attention_mask=(
            replay_inputs.attention_mask.detach()
            if torch.is_tensor(replay_inputs.attention_mask)
            else replay_inputs.attention_mask
        ),
        denoising_metadata=replay_inputs.denoising_metadata,
        normal_query_count=replay_inputs.normal_query_count,
    )
    def exact_comparison(first, second):
        second = second.detach()
        close = torch.isclose(first, second, rtol=0, atol=0, equal_nan=True)
        finite = torch.isfinite(first) & torch.isfinite(second)
        max_diff = (
            float((first[finite] - second[finite]).abs().max())
            if finite.any()
            else 0.0
        )
        return max_diff, bool(close.all())

    query_diff, query_equal = exact_comparison(
        detached_inputs.initial_query, replay_inputs.initial_query
    )
    reference_diff, reference_equal = exact_comparison(
        detached_inputs.initial_reference_unactivated,
        replay_inputs.initial_reference_unactivated,
    )
    diagnostics = {
        "initial_query_max_abs_diff": query_diff,
        "initial_query_exact": query_equal,
        "initial_reference_max_abs_diff": reference_diff,
        "initial_reference_exact": reference_equal,
    }

    with torch.no_grad():
        normal_outputs, normal_trace = _run_decoder_with_capture(
            teacher_ec_transformer, detached_inputs
        )
        if not privileged:
            return (
                _predictions_from_decoder_outputs(
                    normal_outputs, detached_inputs.normal_query_count
                ),
                _slice_normal_trace(
                    normal_trace, detached_inputs.normal_query_count
                ),
                diagnostics,
            )

        gt_xyxy = [box_cxcywh_to_xyxy(target["boxes"]) for target in targets]
        selected_locations = [
            normal_trace["sampling_locations"][index]
            for index in distill_layers
        ]
        selected_weights = [
            normal_trace["attention_weights"][index]
            for index in distill_layers
        ]
        selected_coefficients, bg_dependency = adaptive_farbg_coefficients(
            selected_locations,
            selected_weights,
            gt_xyxy,
            threshold=dependency_threshold,
            far_bg_scale=far_bg_scale,
            ring_scale=ring_scale,
        )
        coefficients = [None] * len(teacher_ec_transformer.decoder.layers)
        for layer_index, coefficient in zip(
            distill_layers, selected_coefficients
        ):
            coefficients[layer_index] = coefficient
        # Non-privileged layers need explicit all-one coefficients because the
        # capture helper treats a list entry as active suppression.
        for index, coefficient in enumerate(coefficients):
            if coefficient is None:
                coefficients[index] = torch.ones_like(
                    normal_trace["attention_weights"][index]
                )

        privileged_outputs, privileged_trace = _run_decoder_with_capture(
            teacher_ec_transformer,
            detached_inputs,
            suppress_coefficients=coefficients,
        )
        diagnostics["bg_dependency_mean"] = float(bg_dependency.mean())
        diagnostics["high_dependency_fraction"] = float(
            (bg_dependency > dependency_threshold).float().mean()
        )
        return (
            _predictions_from_decoder_outputs(
                privileged_outputs, detached_inputs.normal_query_count
            ),
            _slice_normal_trace(
                privileged_trace, detached_inputs.normal_query_count
            ),
            diagnostics,
        )


def sampled_value_suppression_core(
    value: Sequence[torch.Tensor],
    value_spatial_shapes,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_list,
    suppress_coeff: torch.Tensor,
) -> torch.Tensor:
    """Apply coefficients after sampling and before attention aggregation.

    ``suppress_coeff`` has shape ``[B, Q, H, sum(points)]``.  Sampling
    locations and attention weights are consumed unchanged.  With an all-one
    coefficient this is bit-for-bit the project's normal PyTorch core.
    """

    batch, heads, channels, _ = value[0].shape
    query_count = sampling_locations.shape[1]

    sampling_grids = 2 * sampling_locations - 1
    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)

    sampled_values = []
    for level, (height, width) in enumerate(value_spatial_shapes):
        value_level = value[level].reshape(
            batch * heads, channels, height, width
        )
        sampled_values.append(
            F.grid_sample(
                value_level,
                sampling_locations_list[level],
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )

    sampled = torch.concat(sampled_values, dim=-1)
    coeff = suppress_coeff.permute(0, 2, 1, 3).reshape(
        batch * heads, 1, query_count, -1
    )
    sampled = sampled * coeff

    weights = attention_weights.permute(0, 2, 1, 3).reshape(
        batch * heads, 1, query_count, -1
    )
    output = (sampled * weights).sum(-1).reshape(
        batch, heads * channels, query_count
    )
    return output.permute(0, 2, 1)
