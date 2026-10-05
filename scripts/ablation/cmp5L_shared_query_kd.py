"""Shared-student-query hidden-state distillation for cmp5L KQ experiments.

This module is deliberately independent from query-gate and logits/box KD
routes.  The student supplies the complete normal-query initialization to a
frozen teacher decoder; only query hidden states are distilled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class KQSpec:
    group: str
    enabled: bool
    teacher_memory_mode: str
    layers: tuple[int, ...]
    weight: float = 0.5
    output_weight: float = 0.0
    num_queries: int = 300


_KQ_SPECS = {
    "KQ0": KQSpec("KQ0", False, "none", ()),
    "KQ1": KQSpec("KQ1", True, "privileged_np", (3,)),
    "KQ2": KQSpec("KQ2", True, "privileged_np", (0, 1, 2, 3)),
    "KQ3": KQSpec("KQ3", True, "normal", (0, 1, 2, 3)),
    "KQW": KQSpec("KQW", True, "privileged_np", (0, 1, 2, 3), weight=5.0),
    "KQO": KQSpec(
        "KQO", True, "privileged_np", (0, 1, 2, 3),
        weight=0.5, output_weight=50.82931828609598,
    ),
    "KQB": KQSpec(
        "KQB", True, "privileged_np", (0, 1, 2, 3),
        weight=0.5, output_weight=3.643302750228453,
    ),
    "KQR": KQSpec(
        "KQR", True, "privileged_np", (0, 1, 2, 3),
        weight=0.5, output_weight=2.322074686584331,
    ),
    "KQU": KQSpec(
        "KQU", True, "privileged_np", (0, 1, 2, 3),
        weight=0.5, output_weight=6.196734780239296,
    ),
}


def resolve_kq_spec(group: str) -> KQSpec:
    """Return the pre-registered first-round KQ experiment contract."""
    try:
        return _KQ_SPECS[str(group).upper()]
    except KeyError as error:
        raise ValueError(f"unknown KQ group: {group}") from error


def validate_kq_config(config: dict, spec: KQSpec) -> None:
    """Reject drift between the resolved YAML and the registered KQ arm."""
    expected = {
        "enabled": spec.enabled,
        "teacher_memory_mode": spec.teacher_memory_mode,
        "layers": list(spec.layers),
        "weight": spec.weight,
        "num_queries": spec.num_queries,
        "share_student_initial_query": True,
        "share_student_topk": True,
        "detach_teacher": True,
        "loss": "cosine",
        "output_weight": spec.output_weight,
        "output_loss": (
            "sigmoid_soft_bce_grouped" if spec.group == "KQB"
            else "teacher_positive_pairwise_rank" if spec.group == "KQR"
            else "one_way_sigmoid_soft_bce" if spec.group == "KQU"
            else "sigmoid_soft_bce" if spec.output_weight > 0 else "none"
        ),
        "output_layer": 3,
        "output_all_normal_queries": spec.group not in ("KQB", "KQR", "KQU"),
    }
    if spec.group == "KQB":
        expected.update({
            "output_grouping": "hungarian_lesions_top20_far_negatives",
            "negative_topk": 20,
            "negative_max_iou": 0.3,
            "empty_image_output_kd": "zero",
        })
    if spec.group in ("KQR", "KQU"):
        expected.update({
            "output_grouping": (
                "hungarian_lesions_classwise_top20_far_candidates" if spec.group == "KQR"
                else "hungarian_lesions_top20_far_negatives"
            ),
            "negative_topk": 20,
            "negative_max_iou": 0.3,
            "empty_image_output_kd": "zero",
        })
    differences = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if differences:
        raise ValueError(f"KQ config disagrees with {spec.group}: {differences}")


def compose_shared_query_loss(
    detection_loss: torch.Tensor,
    kd_loss: torch.Tensor | None,
    spec: KQSpec,
    *,
    output_kd_loss: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compose detection, hidden KD, and optional output KD objectives."""
    if not spec.enabled:
        if kd_loss is not None or output_kd_loss is not None:
            raise ValueError("KQ0 must not construct a KD loss")
        return detection_loss
    if kd_loss is None:
        raise ValueError(f"{spec.group} requires a KD loss")
    if spec.output_weight > 0 and output_kd_loss is None:
        raise ValueError(f"{spec.group} requires an output KD loss")
    if spec.output_weight == 0 and output_kd_loss is not None:
        raise ValueError(f"{spec.group} must not construct an output KD loss")
    total = detection_loss + spec.weight * kd_loss
    if output_kd_loss is not None:
        total = total + spec.output_weight * output_kd_loss
    return total


@dataclass
class SharedQueryReplay:
    initial_query: torch.Tensor
    initial_reference_unactivated: torch.Tensor
    topk_indices: torch.Tensor
    memory: Sequence[torch.Tensor] | torch.Tensor
    spatial_shapes: object
    attention_mask: torch.Tensor | None = None
    denoising_metadata: object = None
    normal_query_count: int = 300


def normal_query_slice(value: torch.Tensor, normal_query_count: int = 300) -> torch.Tensor:
    """Return only the normal-query suffix, excluding any DN prefix."""
    count = int(normal_query_count)
    if value.ndim < 2 or count <= 0 or value.shape[1] < count:
        raise ValueError("normal query count is incompatible with tensor")
    return value[:, -count:]


def cosine_query_kd(
    student_layers: Sequence[torch.Tensor],
    teacher_layers: Sequence[torch.Tensor],
    *,
    layers: Sequence[int] = (0, 1, 2, 3),
    detach_teacher: bool = True,
):
    """Mean per-layer cosine distance over aligned normal queries."""
    if len(student_layers) != len(teacher_layers):
        raise ValueError("student/teacher layer counts differ")
    if not layers:
        raise ValueError("at least one KD layer is required")
    losses = []
    for index in layers:
        if index < 0 or index >= len(student_layers):
            raise ValueError(f"KD layer {index} is out of range")
        student = student_layers[index]
        teacher = teacher_layers[index]
        if student.shape != teacher.shape or student.ndim != 3:
            raise ValueError("query hidden states must share [B,Q,C] shape")
        teacher = teacher.detach() if detach_teacher else teacher
        student_norm = F.normalize(student, dim=-1)
        teacher_norm = F.normalize(teacher, dim=-1)
        losses.append(1.0 - (student_norm * teacher_norm).sum(dim=-1).mean())
    loss = torch.stack(losses).mean()
    return {
        "loss": loss,
        "layer_losses": losses,
        "layers": [int(index) for index in layers],
    }


def sigmoid_output_kd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    normal_query_count: int = 300,
    detach_teacher: bool = True,
):
    """Full-class soft-label BCE on aligned normal-query slots.

    The Student tensor may contain a DN prefix; the Teacher replay contains
    only normal queries.  Every class dimension participates independently,
    matching the detector's sigmoid classification semantics.
    """
    if student_logits.ndim != 3 or teacher_logits.ndim != 3:
        raise ValueError("classification logits must have [B,Q,C] shape")
    student = normal_query_slice(student_logits, normal_query_count)
    teacher = normal_query_slice(teacher_logits, normal_query_count)
    if student.shape != teacher.shape:
        raise ValueError("aligned Student/Teacher logits must share shape")
    teacher = teacher.detach() if detach_teacher else teacher
    targets = teacher.float().sigmoid()
    loss = F.binary_cross_entropy_with_logits(
        student.float(), targets, reduction="mean"
    )
    return {
        "loss": loss,
        "query_count": int(student.shape[1]),
        "class_count": int(student.shape[2]),
        "element_count": int(student.numel()),
    }


def _pairwise_cxcywh_iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for normalized boxes; selection is detached by the caller."""
    first_xyxy = torch.cat((first[:, :2] - first[:, 2:] / 2, first[:, :2] + first[:, 2:] / 2), dim=-1)
    second_xyxy = torch.cat((second[:, :2] - second[:, 2:] / 2, second[:, :2] + second[:, 2:] / 2), dim=-1)
    left_top = torch.maximum(first_xyxy[:, None, :2], second_xyxy[None, :, :2])
    right_bottom = torch.minimum(first_xyxy[:, None, 2:], second_xyxy[None, :, 2:])
    intersection = (right_bottom - left_top).clamp(min=0).prod(dim=-1)
    first_area = (first_xyxy[:, 2:] - first_xyxy[:, :2]).clamp(min=0).prod(dim=-1)
    second_area = (second_xyxy[:, 2:] - second_xyxy[:, :2]).clamp(min=0).prod(dim=-1)
    union = first_area[:, None] + second_area[None, :] - intersection
    return intersection / union.clamp(min=1e-12)


def final_layer_hungarian_matches(
    matcher,
    outputs: dict,
    targets: Sequence[dict],
    *,
    normal_query_count: int = 300,
):
    """Repeat the criterion's final classification assignment without gradients.

    The criterion uses this same matcher on final `pred_logits`/`pred_boxes`
    before any cross-layer union or DN assignment. Those final tensors must
    already contain exactly the normal-query count.
    """
    logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
    if logits.ndim != 3 or boxes.shape != (*logits.shape[:2], 4):
        raise ValueError("final logits/boxes shapes disagree")
    if logits.shape[1] != normal_query_count:
        raise ValueError("final detection outputs must contain normal queries only")
    with torch.no_grad():
        return matcher({
            "pred_logits": logits.detach(),
            "pred_boxes": boxes.detach(),
        }, targets)["indices"]


def grouped_sigmoid_output_kd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_boxes: torch.Tensor,
    targets: Sequence[dict],
    matches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    normal_query_count: int = 300,
    negative_topk: int = 20,
    negative_max_iou: float = 0.3,
    global_image_count: int | None = None,
    ddp_world_size: int = 1,
) -> dict:
    """KQ-B loss: matched lesions and top far negatives, balanced per image.

    `matches` must be the final-layer Hungarian pairs used by classification
    detection loss, not the cross-layer union used by some box/local terms.
    All selection is detached. DDP scales each rank's image sum by
    world_size/global_image_count so DDP's gradient averaging yields the
    global per-image mean even if local batch sizes differ.
    """
    if student_logits.ndim != 3 or teacher_logits.ndim != 3 or student_boxes.ndim != 3:
        raise ValueError("logits and boxes must have [B,Q,C] and [B,Q,4] shape")
    student = normal_query_slice(student_logits, normal_query_count).float()
    teacher = normal_query_slice(teacher_logits, normal_query_count).detach().float()
    if student.shape != teacher.shape or student_boxes.shape != (*student.shape[:2], 4):
        raise ValueError("student/teacher normal queries or boxes do not align")
    batch_size, query_count, class_count = student.shape
    if len(targets) != batch_size or len(matches) != batch_size:
        raise ValueError("targets and matches must have one entry per image")
    if negative_topk < 0 or not 0 <= negative_max_iou <= 1:
        raise ValueError("invalid negative selection limits")
    denominator = batch_size if global_image_count is None else int(global_image_count)
    if denominator <= 0 or ddp_world_size <= 0:
        raise ValueError("image normalization must be positive")

    zero = student.sum() * 0.0
    positive_sum, negative_sum = zero, zero
    lesion_indices, negative_indices = [], []
    class_counts = {str(index): 0 for index in range(class_count)}
    gt_class_gaps = []
    lesion_probability_samples = []
    teacher_lower_gt_count = 0
    low_score_lesion_count = 0
    empty_count = 0

    for image_index, (target, matched) in enumerate(zip(targets, matches)):
        with torch.no_grad():
            gt_boxes = target["boxes"].detach().float().to(student_boxes.device)
            gt_labels = target["labels"].detach().long().to(student.device)
            if gt_boxes.ndim != 2 or gt_boxes.shape[-1] != 4 or gt_labels.numel() != len(gt_boxes):
                raise ValueError("each target needs aligned boxes [N,4] and labels [N]")
            if not len(gt_boxes):
                empty_count += 1
                lesion_indices.append([])
                negative_indices.append([])
                continue
            query_ids = matched[0].detach().long().to(student.device)
            target_ids = matched[1].detach().long().to(student.device)
            if len(query_ids) != len(target_ids) or len(query_ids.unique()) != len(query_ids):
                raise ValueError("Hungarian matches must be one-to-one")
            if query_ids.numel() and (
                int(query_ids.min()) < 0 or int(query_ids.max()) >= query_count
                or int(target_ids.min()) < 0 or int(target_ids.max()) >= len(gt_boxes)
            ):
                raise ValueError("Hungarian match index is out of range")
            eligible = torch.ones(query_count, dtype=torch.bool, device=student.device)
            eligible[query_ids] = False
            max_iou = _pairwise_cxcywh_iou(
                student_boxes[image_index].detach().float(), gt_boxes,
            ).max(dim=1).values
            eligible &= max_iou < negative_max_iou
            candidate_ids = torch.nonzero(eligible, as_tuple=False).flatten()
            confidence = student[image_index].detach().sigmoid().max(dim=-1).values
            if candidate_ids.numel() and negative_topk:
                order = torch.argsort(confidence[candidate_ids], descending=True, stable=True)
                hard_ids = candidate_ids[order[:negative_topk]]
            else:
                hard_ids = candidate_ids[:0]
            lesion_indices.append(query_ids.tolist())
            negative_indices.append(hard_ids.tolist())
            if set(lesion_indices[-1]) & set(negative_indices[-1]):
                raise RuntimeError("lesion and hard-negative groups overlap")
            if query_ids.numel():
                classes = gt_labels[target_ids]
                student_gt = student[image_index, query_ids, classes].detach().sigmoid()
                teacher_gt = teacher[image_index, query_ids, classes].sigmoid()
                differences = teacher_gt - student_gt
                gt_class_gaps.extend(differences.cpu().tolist())
                teacher_lower_gt_count += int((differences < 0).sum())
                low_score_lesion_count += int((student_gt < 0.5).sum())
                for class_id in classes.tolist():
                    class_counts[str(int(class_id))] += 1
                for query_id, target_id, class_id, student_prob, teacher_prob in zip(
                    query_ids.tolist(), target_ids.tolist(), classes.tolist(),
                    student_gt.tolist(), teacher_gt.tolist(),
                ):
                    if len(lesion_probability_samples) >= 16:
                        break
                    lesion_probability_samples.append({
                        "image_index": image_index,
                        "query_index": int(query_id),
                        "target_index": int(target_id),
                        "class_id": int(class_id),
                        "student_gt_class_probability": float(student_prob),
                        "teacher_gt_class_probability": float(teacher_prob),
                    })
        if query_ids.numel():
            positive_sum = positive_sum + F.binary_cross_entropy_with_logits(
                student[image_index, query_ids], teacher[image_index, query_ids].sigmoid(), reduction="mean"
            )
        if hard_ids.numel():
            negative_sum = negative_sum + F.binary_cross_entropy_with_logits(
                student[image_index, hard_ids], teacher[image_index, hard_ids].sigmoid(), reduction="mean"
            )

    scale = float(ddp_world_size) / denominator
    positive_loss = positive_sum * scale
    negative_loss = negative_sum * scale
    return {
        "loss": 0.5 * (positive_loss + negative_loss),
        "positive_loss": positive_loss,
        "negative_loss": negative_loss,
        "lesion_indices": lesion_indices,
        "negative_indices": negative_indices,
        "lesion_count": sum(map(len, lesion_indices)),
        "negative_count": sum(map(len, negative_indices)),
        "empty_image_count": empty_count,
        "matched_class_counts": class_counts,
        "low_score_lesion_count_below_0_5": low_score_lesion_count,
        "teacher_lower_gt_class_count": teacher_lower_gt_count,
        "teacher_minus_student_gt_class_probability": gt_class_gaps,
        "lesion_probability_samples": lesion_probability_samples,
        "normal_query_count": query_count,
        "class_count": class_count,
        "global_image_count": denominator,
    }


def candidate_ranking_kd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_boxes: torch.Tensor,
    targets: Sequence[dict],
    matches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    normal_query_count: int = 300,
    negative_topk: int = 20,
    negative_max_iou: float = 0.3,
    global_image_count: int | None = None,
    ddp_world_size: int = 1,
) -> dict:
    """KQ-R: GT-class lesion-before-far-candidate ranking on shared slots.

    Inputs are final *post-LQE* detection logits. Selection, matching and
    Teacher targets are detached. DDP scaling compensates gradient averaging.
    """
    student = normal_query_slice(student_logits, normal_query_count).float()
    teacher = normal_query_slice(teacher_logits, normal_query_count).detach().float()
    if student.ndim != 3 or student.shape != teacher.shape or student_boxes.shape != (*student.shape[:2], 4):
        raise ValueError("aligned normal logits/boxes must be [B,Q,C]/[B,Q,4]")
    batch_size, query_count, class_count = student.shape
    if len(targets) != batch_size or len(matches) != batch_size:
        raise ValueError("targets/matches must cover the batch")
    if negative_topk < 0 or not 0 <= negative_max_iou <= 1:
        raise ValueError("invalid negative selection limits")
    denominator = batch_size if global_image_count is None else int(global_image_count)
    if denominator <= 0 or ddp_world_size <= 0:
        raise ValueError("invalid global image normalization")
    zero = student.sum() * 0.0
    image_sum = zero
    matched_counts = {str(index): 0 for index in range(class_count)}
    valid_counts = dict(matched_counts)
    candidate_pairs = valid_pairs = reverse_pairs = empty_images = zero_target_images = 0
    for image_index, (target, (query_ids, target_ids)) in enumerate(zip(targets, matches)):
        with torch.no_grad():
            boxes = target["boxes"].detach().float().to(student_boxes.device)
            labels = target["labels"].detach().long().to(student.device)
            query_ids = query_ids.detach().long().to(student.device)
            target_ids = target_ids.detach().long().to(student.device)
            if boxes.ndim != 2 or boxes.shape[-1] != 4 or len(labels) != len(boxes):
                raise ValueError("target boxes/labels disagree")
            if not len(boxes):
                empty_images += 1
                zero_target_images += 1
                continue
            if len(query_ids) != len(target_ids) or len(query_ids.unique()) != len(query_ids):
                raise ValueError("matches must be one-to-one")
            if len(query_ids) and (int(query_ids.min()) < 0 or int(query_ids.max()) >= query_count
                                   or int(target_ids.min()) < 0 or int(target_ids.max()) >= len(boxes)):
                raise ValueError("match index out of range")
            eligible = torch.ones(query_count, dtype=torch.bool, device=student.device)
            eligible[query_ids] = False
            max_iou = _pairwise_cxcywh_iou(student_boxes[image_index].detach().float(), boxes).max(dim=1).values
            eligible &= max_iou < negative_max_iou
            candidate_ids = eligible.nonzero(as_tuple=False).flatten()
        lesion_losses = []
        for query_id, target_id in zip(query_ids.tolist(), target_ids.tolist()):
            class_id = int(labels[target_id])
            if not 0 <= class_id < class_count:
                raise ValueError("target category out of range")
            matched_counts[str(class_id)] += 1
            if not candidate_ids.numel() or not negative_topk:
                continue
            with torch.no_grad():
                scores = student[image_index, candidate_ids, class_id].detach()
                order = torch.argsort(scores, descending=True, stable=True)
                competitors = candidate_ids[order[:negative_topk]]
                teacher_differences = teacher[image_index, query_id, class_id] - teacher[image_index, competitors, class_id]
                valid = teacher_differences > 0
                candidate_pairs += len(competitors)
                reverse_pairs += int((~valid).sum())
                valid_pairs += int(valid.sum())
                selected = competitors[valid]
                targets_soft = teacher_differences[valid].sigmoid()
            if selected.numel():
                differences = student[image_index, query_id, class_id] - student[image_index, selected, class_id]
                lesion_losses.append(F.binary_cross_entropy_with_logits(differences, targets_soft, reduction="mean"))
                valid_counts[str(class_id)] += 1
        if lesion_losses:
            image_sum = image_sum + torch.stack(lesion_losses).mean()
        else:
            zero_target_images += 1
    loss = image_sum * (float(ddp_world_size) / denominator)
    return {
        "loss": loss,
        "candidate_pair_count": candidate_pairs,
        "valid_pair_count": valid_pairs,
        "reverse_pair_count": reverse_pairs,
        "matched_class_counts": matched_counts,
        "valid_class_counts": valid_counts,
        "empty_image_count": empty_images,
        "zero_target_image_count": zero_target_images,
        "global_image_count": denominator,
        "logit_source": "final_post_lqe_pred_logits",
    }


def one_way_protected_kd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_boxes: torch.Tensor,
    targets: Sequence[dict],
    matches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    normal_query_count: int = 300,
    negative_topk: int = 20,
    negative_max_iou: float = 0.3,
    global_image_count: int | None = None,
    ddp_world_size: int = 1,
) -> dict:
    """KQ-U: one-way GT-class lift and far-negative class suppression.

    Direction masks are detached; denominators precede the masks. Inputs are
    final post-LQE detection logits, not uncorrected classification-head logits.
    """
    student = normal_query_slice(student_logits, normal_query_count).float()
    teacher = normal_query_slice(teacher_logits, normal_query_count).detach().float()
    if student.ndim != 3 or student.shape != teacher.shape or student_boxes.shape != (*student.shape[:2], 4):
        raise ValueError("aligned normal logits/boxes must be [B,Q,C]/[B,Q,4]")
    batch_size, query_count, class_count = student.shape
    if len(targets) != batch_size or len(matches) != batch_size:
        raise ValueError("targets/matches must cover the batch")
    if negative_topk < 0 or not 0 <= negative_max_iou <= 1:
        raise ValueError("invalid negative selection limits")
    denominator = batch_size if global_image_count is None else int(global_image_count)
    if denominator <= 0 or ddp_world_size <= 0:
        raise ValueError("invalid global image normalization")
    zero = student.sum() * 0.0
    positive_sum, negative_sum = zero, zero
    matched_counts = {str(index): 0 for index in range(class_count)}
    active_counts = dict(matched_counts)
    lesion_count = lesion_active = negative_count = negative_active = empty_images = zero_target_images = 0
    probability_samples = []
    for image_index, (target, (query_ids, target_ids)) in enumerate(zip(targets, matches)):
        with torch.no_grad():
            boxes = target["boxes"].detach().float().to(student_boxes.device)
            labels = target["labels"].detach().long().to(student.device)
            query_ids = query_ids.detach().long().to(student.device)
            target_ids = target_ids.detach().long().to(student.device)
            if boxes.ndim != 2 or boxes.shape[-1] != 4 or len(labels) != len(boxes):
                raise ValueError("target boxes/labels disagree")
            if not len(boxes):
                empty_images += 1
                zero_target_images += 1
                continue
            if len(query_ids) != len(target_ids) or len(query_ids.unique()) != len(query_ids):
                raise ValueError("matches must be one-to-one")
            if len(query_ids) and (int(query_ids.min()) < 0 or int(query_ids.max()) >= query_count
                                   or int(target_ids.min()) < 0 or int(target_ids.max()) >= len(boxes)):
                raise ValueError("match index out of range")
            eligible = torch.ones(query_count, dtype=torch.bool, device=student.device)
            eligible[query_ids] = False
            max_iou = _pairwise_cxcywh_iou(student_boxes[image_index].detach().float(), boxes).max(dim=1).values
            eligible &= max_iou < negative_max_iou
            candidate_ids = eligible.nonzero(as_tuple=False).flatten()
            confidence = student[image_index].detach().sigmoid().max(dim=-1).values
            if candidate_ids.numel() and negative_topk:
                order = torch.argsort(confidence[candidate_ids], descending=True, stable=True)
                hard_ids = candidate_ids[order[:negative_topk]]
            else:
                hard_ids = candidate_ids[:0]
        has_target = False
        if query_ids.numel():
            classes = labels[target_ids]
            if int(classes.min()) < 0 or int(classes.max()) >= class_count:
                raise ValueError("target category out of range")
            lesion_count += len(query_ids)
            student_gt = student[image_index, query_ids, classes]
            teacher_gt = teacher[image_index, query_ids, classes].sigmoid()
            with torch.no_grad():
                active = teacher_gt > student_gt.detach().sigmoid()
            per_lesion = F.binary_cross_entropy_with_logits(student_gt, teacher_gt, reduction="none")
            positive_sum = positive_sum + (per_lesion * active).sum() / len(query_ids)
            lesion_active += int(active.sum())
            for class_id, is_active, student_prob, teacher_prob in zip(
                classes.tolist(), active.tolist(), student_gt.detach().sigmoid().tolist(), teacher_gt.tolist()
            ):
                matched_counts[str(int(class_id))] += 1
                active_counts[str(int(class_id))] += int(is_active)
                if len(probability_samples) < 16:
                    probability_samples.append({
                        "image_index": image_index, "class_id": int(class_id),
                        "student_gt_class_probability": float(student_prob),
                        "teacher_gt_class_probability": float(teacher_prob),
                        "active": bool(is_active),
                    })
            has_target = True
        if hard_ids.numel():
            negative_count += len(hard_ids)
            student_negative = student[image_index, hard_ids]
            teacher_negative = teacher[image_index, hard_ids].sigmoid()
            with torch.no_grad():
                active = teacher_negative < student_negative.detach().sigmoid()
            per_dimension = F.binary_cross_entropy_with_logits(student_negative, teacher_negative, reduction="none")
            negative_sum = negative_sum + (per_dimension * active).sum() / (len(hard_ids) * class_count)
            negative_active += int(active.sum())
            has_target = True
        if not has_target:
            zero_target_images += 1
    scale = float(ddp_world_size) / denominator
    positive_loss, negative_loss = positive_sum * scale, negative_sum * scale
    return {
        "loss": 0.5 * (positive_loss + negative_loss),
        "positive_loss": positive_loss,
        "negative_loss": negative_loss,
        "lesion_count": lesion_count,
        "lesion_active_count": lesion_active,
        "negative_count": negative_count,
        "negative_active_dimension_count": negative_active,
        "matched_class_counts": matched_counts,
        "active_class_counts": active_counts,
        "empty_image_count": empty_images,
        "zero_target_image_count": zero_target_images,
        "global_image_count": denominator,
        "lesion_probability_samples": probability_samples,
        "logit_source": "final_post_lqe_pred_logits",
    }


def shared_query_alignment_audit(
    student: SharedQueryReplay,
    teacher: SharedQueryReplay,
):
    """Audit exact query/reference/index sharing before decoder replay."""
    fields = {
        "initial_query": student.initial_query,
        "initial_reference_unactivated": student.initial_reference_unactivated,
        "topk_indices": student.topk_indices,
    }
    teacher_fields = {
        "initial_query": teacher.initial_query,
        "initial_reference_unactivated": teacher.initial_reference_unactivated,
        "topk_indices": teacher.topk_indices,
    }
    result = {}
    for name, first in fields.items():
        second = teacher_fields[name]
        if first.shape != second.shape:
            raise ValueError(f"shared field shape mismatch: {name}")
        difference = (first.detach() - second.detach()).abs()
        difference_float = difference.float()
        finite_difference = difference_float[torch.isfinite(difference_float)]
        result[name] = {
            "shape": list(first.shape),
            "max_abs": float(finite_difference.max()) if finite_difference.numel() else 0.0,
            "mean_abs": float(finite_difference.mean()) if finite_difference.numel() else 0.0,
            "nonfinite_difference_count": int((~torch.isfinite(difference_float)).sum()),
            "exact": bool(torch.equal(first.detach(), second.detach())),
        }
    if student.normal_query_count != teacher.normal_query_count:
        raise ValueError("normal query counts differ")
    result["normal_query_count"] = int(student.normal_query_count)
    return result


def layer_query_audit(student_layers, teacher_layers):
    """Return shape, norm and cosine summaries for aligned decoder outputs."""
    if len(student_layers) != len(teacher_layers):
        raise ValueError("student/teacher layer counts differ")
    rows = []
    for index, (student, teacher) in enumerate(zip(student_layers, teacher_layers)):
        if student.shape != teacher.shape or student.ndim != 3:
            raise ValueError(f"layer {index} hidden-state shape mismatch")
        student_f = student.float()
        teacher_f = teacher.detach().float()
        cosine = F.cosine_similarity(student_f, teacher_f, dim=-1, eps=1e-8)
        rows.append({
            "layer": index,
            "shape": list(student.shape),
            "student_norm_mean": float(student_f.norm(dim=-1).mean()),
            "student_norm_std": float(student_f.norm(dim=-1).std(unbiased=False)),
            "teacher_norm_mean": float(teacher_f.norm(dim=-1).mean()),
            "teacher_norm_std": float(teacher_f.norm(dim=-1).std(unbiased=False)),
            "cosine_mean": float(cosine.mean()),
            "cosine_std": float(cosine.std(unbiased=False)),
            "max_abs": float((student_f - teacher_f).abs().max()),
        })
    return rows
