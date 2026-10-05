"""Pure contracts for cmp5L QG query selection and behavior distillation."""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn.functional as F


def compose_qg_loss(
    detection_loss,
    selection_loss,
    behavior_loss,
    *,
    group,
    lambda_select=1.0,
    lambda_behavior=0.5,
):
    """Compose the pre-registered QG0-QG3 objective."""
    if group not in {"QG0", "QG1", "QG2", "QG3"}:
        raise ValueError(f"unknown QG group: {group}")
    weighted_selection = selection_loss * float(lambda_select)
    weighted_behavior = behavior_loss * float(lambda_behavior)
    total = detection_loss
    if group != "QG0":
        total = total + weighted_selection
    if group in {"QG2", "QG3"}:
        total = total + weighted_behavior
    return {
        "total": total,
        "detection": detection_loss,
        "selection": selection_loss,
        "behavior": behavior_loss,
        "weighted_selection": weighted_selection,
        "weighted_behavior": weighted_behavior,
    }


def slice_normal_query_layers(layers, *, normal_queries):
    """Remove the training-only DN prefix from decoder-layer logits."""
    normal_queries = int(normal_queries)
    if normal_queries <= 0:
        raise ValueError("normal query count must be positive")
    result = []
    for layer in layers:
        if layer.ndim != 3 or layer.shape[1] < normal_queries:
            raise ValueError("normal query count exceeds decoder layer shape")
        result.append(layer[:, -normal_queries:])
    return result


@contextmanager
def temporary_query_gate_state(transformer_decoder, enabled):
    """Set QG state for one forward and restore it even when forward fails."""
    previous = bool(getattr(transformer_decoder, "query_gate_enabled", False))
    transformer_decoder.query_gate_enabled = bool(enabled)
    try:
        yield transformer_decoder
    finally:
        transformer_decoder.query_gate_enabled = previous


def max_named_buffer_change(before, named_buffers):
    """Return a dtype-safe maximum change over a named-buffer snapshot."""
    maximum = 0.0
    observed = set()
    for name, value in named_buffers:
        if name not in before:
            raise KeyError(f"unexpected buffer after forward: {name}")
        observed.add(name)
        reference = before[name]
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise RuntimeError(f"buffer metadata changed: {name}")
        if not value.numel():
            continue
        if value.dtype == torch.bool:
            difference = float(torch.logical_xor(value, reference).any())
        else:
            difference = float((value - reference).abs().max())
        maximum = max(maximum, difference)
    missing = sorted(set(before) - observed)
    if missing:
        raise KeyError(f"buffers disappeared after forward: {missing[:10]}")
    return maximum


def attach_query_gate_head(model, *, hidden_dim=64, prior_probability=0.01):
    """Register the QG head before EMA and optimizer construction."""
    from engine.edgecrafter.decoder import QueryGateHead

    ec_transformer = getattr(model, "decoder", None)
    transformer_decoder = getattr(ec_transformer, "decoder", None)
    if transformer_decoder is None or not hasattr(transformer_decoder, "hidden_dim"):
        raise TypeError("expected an EC detector with decoder.decoder.hidden_dim")
    if getattr(transformer_decoder, "query_gate_head", None) is not None:
        raise RuntimeError("query gate head is already installed")
    head = QueryGateHead(
        in_dim=int(transformer_decoder.hidden_dim),
        hidden_dim=int(hidden_dim),
        prior_probability=float(prior_probability),
    )
    try:
        head = head.to(next(model.parameters()).device)
    except StopIteration:
        pass
    transformer_decoder.query_gate_head = head
    transformer_decoder.query_gate_enabled = True
    return head


def _paired_bernoulli_kl(student_logits, teacher_logits, pairs):
    if pairs.ndim != 2 or pairs.shape[1] != 3:
        raise ValueError("pairs must be [N,3] as batch/student/teacher indices")
    if not pairs.shape[0]:
        return student_logits.sum() * 0.0, 0
    pairs = pairs.to(device=student_logits.device, dtype=torch.long)
    batch, student_index, teacher_index = pairs.unbind(dim=1)
    student = student_logits[batch, student_index]
    teacher = teacher_logits.detach()[batch, teacher_index]
    probability = teacher.sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(student, probability, reduction="none")
    teacher_entropy = F.binary_cross_entropy_with_logits(teacher, probability, reduction="none")
    local_sum = (cross_entropy - teacher_entropy).sum()
    global_count = _global_count(student.new_tensor(pairs.shape[0], dtype=torch.long))
    mean = _distributed_mean(local_sum, global_count * student.shape[-1])
    return mean, int(global_count.item())


def bernoulli_behavior_kd(
    student_layers,
    teacher_layers,
    *,
    positive_pairs,
    competitive_pairs,
):
    """Pool-balanced Bernoulli KL, averaged over decoder layers."""
    if not student_layers or len(student_layers) != len(teacher_layers):
        raise ValueError("student and teacher must provide the same nonzero layer count")
    per_layer = []
    positive_losses = []
    competitive_losses = []
    positive_count = competitive_count = 0
    for student, teacher in zip(student_layers, teacher_layers):
        if student.shape != teacher.shape or student.ndim != 3:
            raise ValueError("paired logits must have matching [B,Q,C] shape")
        positive_loss, positive_count = _paired_bernoulli_kl(student, teacher, positive_pairs)
        competitive_loss, competitive_count = _paired_bernoulli_kl(
            student, teacher, competitive_pairs
        )
        present = int(positive_count > 0) + int(competitive_count > 0)
        if present == 2:
            layer_loss = 0.5 * (positive_loss + competitive_loss)
        elif positive_count:
            layer_loss = positive_loss
        elif competitive_count:
            layer_loss = competitive_loss
        else:
            layer_loss = student.sum() * 0.0
        per_layer.append(layer_loss)
        positive_losses.append(positive_loss)
        competitive_losses.append(competitive_loss)
    return {
        "loss": torch.stack(per_layer).mean(),
        "per_layer": per_layer,
        "positive_loss_per_layer": positive_losses,
        "competitive_loss_per_layer": competitive_losses,
        "positive_count": positive_count,
        "competitive_count": competitive_count,
    }


def _cxcywh_iou(left, right):
    left_xyxy = torch.cat((left[..., :2] - left[..., 2:] / 2, left[..., :2] + left[..., 2:] / 2), dim=-1)
    right_xyxy = torch.cat((right[..., :2] - right[..., 2:] / 2, right[..., :2] + right[..., 2:] / 2), dim=-1)
    top_left = torch.maximum(left_xyxy[:, None, :2], right_xyxy[None, :, :2])
    bottom_right = torch.minimum(left_xyxy[:, None, 2:], right_xyxy[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(-1)
    left_area = (left_xyxy[:, 2:] - left_xyxy[:, :2]).clamp(min=0).prod(-1)
    right_area = (right_xyxy[:, 2:] - right_xyxy[:, :2]).clamp(min=0).prod(-1)
    return intersection / (left_area[:, None] + right_area[None, :] - intersection).clamp(min=1e-12)


def match_selection_queries(
    teacher_query,
    teacher_reference,
    student_query,
    student_reference,
    *,
    teacher_gt_to_query,
    student_gt_to_query,
    min_cosine=0.5,
    min_reference_iou=0.3,
):
    """GT-identity-first teacher/student mapping; never assumes query index."""
    from scipy.optimize import linear_sum_assignment

    tensors = (teacher_query, teacher_reference, student_query, student_reference)
    if any(value.ndim != 2 for value in tensors):
        raise ValueError("query and reference tensors must be rank two")
    pairs, used_teacher, used_student = [], set(), set()
    cosine_all = F.normalize(teacher_query.detach().float(), dim=-1) @ F.normalize(student_query.detach().float(), dim=-1).T
    iou_all = _cxcywh_iou(teacher_reference.detach().float(), student_reference.detach().float())
    for gt_index in sorted(set(teacher_gt_to_query) & set(student_gt_to_query)):
        teacher_index = int(teacher_gt_to_query[gt_index])
        student_index = int(student_gt_to_query[gt_index])
        pairs.append({
            "teacher_query": teacher_index,
            "student_query": student_index,
            "source": "gt_identity",
            "gt_index": int(gt_index),
            "cosine": float(cosine_all[teacher_index, student_index]),
            "reference_iou": float(iou_all[teacher_index, student_index]),
        })
        used_teacher.add(teacher_index)
        used_student.add(student_index)

    teacher_left = [i for i in range(teacher_query.shape[0]) if i not in used_teacher]
    student_left = [i for i in range(student_query.shape[0]) if i not in used_student]
    if teacher_left and student_left:
        cosine = cosine_all[teacher_left][:, student_left]
        iou = iou_all[teacher_left][:, student_left]
        cost = (1.0 - cosine) + 2.0 * (1.0 - iou)
        rows, columns = linear_sum_assignment(cost.cpu().numpy())
        for row, column in zip(rows.tolist(), columns.tolist()):
            teacher_index = teacher_left[row]
            student_index = student_left[column]
            similarity = float(cosine_all[teacher_index, student_index])
            reference_iou = float(iou_all[teacher_index, student_index])
            if similarity < float(min_cosine) or reference_iou < float(min_reference_iou):
                continue
            pairs.append({
                "teacher_query": teacher_index,
                "student_query": student_index,
                "source": "init_token_reference",
                "gt_index": None,
                "cosine": similarity,
                "reference_iou": reference_iou,
            })
    return sorted(pairs, key=lambda item: (item["teacher_query"], item["student_query"]))


def match_competitive_candidates(
    student_logits,
    student_boxes,
    teacher_logits,
    teacher_boxes,
    *,
    positive_pairs,
    top_k=20,
    score_threshold=0.05,
    min_iou=0.30,
):
    """Pre-registered high-score, same-class, spatial one-to-one pool."""
    from scipy.optimize import linear_sum_assignment

    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 3:
        raise ValueError("student and teacher logits must be [B,Q,C] and match")
    if student_boxes.shape != teacher_boxes.shape or student_boxes.shape[:2] != student_logits.shape[:2]:
        raise ValueError("student and teacher boxes must be [B,Q,4] and match logits")
    device = student_logits.device
    positive_pairs = positive_pairs.to(device=device, dtype=torch.long)
    excluded_student = {(int(b), int(q)) for b, q, _ in positive_pairs.tolist()}
    excluded_teacher = {(int(b), int(q)) for b, _, q in positive_pairs.tolist()}
    student_prob = student_logits.detach().float().sigmoid()
    teacher_prob = teacher_logits.detach().float().sigmoid()
    student_score, student_class = student_prob.max(dim=-1)
    teacher_score, teacher_class = teacher_prob.max(dim=-1)
    pairs = []
    records = []
    student_candidate_count = teacher_candidate_count = 0
    for batch in range(student_logits.shape[0]):
        def candidates(scores, classes, excluded):
            values = []
            for query in range(scores.shape[0]):
                if (batch, query) in excluded or float(scores[query]) < float(score_threshold):
                    continue
                values.append(query)
            values.sort(key=lambda query: float(scores[query]), reverse=True)
            return values[: int(top_k)]

        student_candidates = candidates(student_score[batch], student_class[batch], excluded_student)
        teacher_candidates = candidates(teacher_score[batch], teacher_class[batch], excluded_teacher)
        student_candidate_count += len(student_candidates)
        teacher_candidate_count += len(teacher_candidates)
        for top_class in range(student_logits.shape[-1]):
            student_group = [q for q in student_candidates if int(student_class[batch, q]) == top_class]
            teacher_group = [q for q in teacher_candidates if int(teacher_class[batch, q]) == top_class]
            if not student_group or not teacher_group:
                continue
            iou = _cxcywh_iou(student_boxes[batch, student_group].detach().float(), teacher_boxes[batch, teacher_group].detach().float())
            rows, columns = linear_sum_assignment((1.0 - iou).cpu().numpy())
            for row, column in zip(rows.tolist(), columns.tolist()):
                value = float(iou[row, column])
                if value < float(min_iou):
                    continue
                student_query = student_group[row]
                teacher_query = teacher_group[column]
                pairs.append([batch, student_query, teacher_query])
                records.append({
                    "batch": batch,
                    "student_query": student_query,
                    "teacher_query": teacher_query,
                    "top_class": top_class,
                    "student_score": float(student_score[batch, student_query]),
                    "teacher_score": float(teacher_score[batch, teacher_query]),
                    "iou": value,
                })
    pair_tensor = torch.tensor(pairs, dtype=torch.long, device=device).reshape(-1, 3)
    return {
        "pairs": pair_tensor,
        "records": records,
        "student_candidate_count": student_candidate_count,
        "teacher_candidate_count": teacher_candidate_count,
    }


def broadcast_soft_gate_coeff(
    probability,
    *,
    total_queries,
    normal_queries,
    heads,
    points,
):
    """Build `[B,Q,H,P]` all-point coefficients; DN prefix is identity."""
    if probability.ndim != 2 or probability.shape[1] != int(normal_queries):
        raise ValueError("gate probability must be [B, normal_queries]")
    if int(total_queries) < int(normal_queries):
        raise ValueError("total_queries cannot be smaller than normal_queries")
    normal = 1.0 - 0.8 * probability
    prefix = probability.new_ones(
        probability.shape[0], int(total_queries) - int(normal_queries)
    )
    all_queries = torch.cat((prefix, normal), dim=1)
    return all_queries[:, :, None, None].expand(
        probability.shape[0], int(total_queries), int(heads), int(points)
    )


def _global_count(local_count: torch.Tensor) -> torch.Tensor:
    count = local_count.detach().clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(count)
    return count


def _distributed_mean(local_sum: torch.Tensor, global_count: torch.Tensor) -> torch.Tensor:
    if not int(global_count.item()):
        return local_sum * 0.0
    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    return local_sum * float(world) / global_count.to(local_sum.dtype)


def balanced_selection_bce(logits, labels, valid_mask):
    """Balanced protect/suppress BCE with globally correct DDP normalization.

    When both groups exist globally they receive equal 0.5 weight.  If only one
    group exists, its mean receives weight 1.  With no valid labels the returned
    zero remains attached to ``logits`` so ordinary detector loss can backprop.
    """
    if logits.shape != labels.shape or logits.shape != valid_mask.shape:
        raise ValueError("selection logits, labels and validity mask must align")
    labels = labels.to(dtype=logits.dtype)
    valid_mask = valid_mask.to(dtype=torch.bool)
    element = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    protect = valid_mask & (labels < 0.5)
    suppress = valid_mask & (labels >= 0.5)
    protect_count = _global_count(protect.sum())
    suppress_count = _global_count(suppress.sum())
    protect_loss = _distributed_mean((element * protect).sum(), protect_count)
    suppress_loss = _distributed_mean((element * suppress).sum(), suppress_count)
    present = int(protect_count.item() > 0) + int(suppress_count.item() > 0)
    if present == 2:
        loss = 0.5 * (protect_loss + suppress_loss)
    elif int(protect_count.item()) > 0:
        loss = protect_loss
    elif int(suppress_count.item()) > 0:
        loss = suppress_loss
    else:
        loss = logits.sum() * 0.0
    return {
        "loss": loss,
        "protect_loss": protect_loss,
        "suppress_loss": suppress_loss,
        "protect_count": int(protect_count.item()),
        "suppress_count": int(suppress_count.item()),
    }
