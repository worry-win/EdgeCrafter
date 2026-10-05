"""Pure contracts for KQ teacher-quality and no-step gradient diagnostics."""

from __future__ import annotations

import math

import torch

from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy


def restore_trainable_parameter_mask(target: torch.nn.Module, reference: torch.nn.Module):
    """Apply the training model's parameter mask to an EMA-weight model copy."""
    target_parameters = dict(target.named_parameters())
    reference_parameters = dict(reference.named_parameters())
    if target_parameters.keys() != reference_parameters.keys():
        missing = sorted(reference_parameters.keys() - target_parameters.keys())
        extra = sorted(target_parameters.keys() - reference_parameters.keys())
        raise ValueError(f"parameter schemas differ: missing={missing}, extra={extra}")
    trainable = 0
    frozen = 0
    for name, parameter in target_parameters.items():
        enabled = bool(reference_parameters[name].requires_grad)
        parameter.requires_grad_(enabled)
        trainable += int(enabled)
        frozen += int(not enabled)
    return {
        "trainable_parameter_tensors": trainable,
        "frozen_parameter_tensors": frozen,
        "total_parameter_tensors": len(target_parameters),
    }


def _box_iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    if first.numel() == 0 or second.numel() == 0:
        return first.new_zeros((first.shape[0], second.shape[0]))
    lt = torch.maximum(first[:, None, :2], second[None, :, :2])
    rb = torch.minimum(first[:, None, 2:], second[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_first = (first[:, 2:] - first[:, :2]).clamp(min=0).prod(-1)
    area_second = (second[:, 2:] - second[:, :2]).clamp(min=0).prod(-1)
    return inter / (area_first[:, None] + area_second[None, :] - inter).clamp(min=1e-9)


def fixed_student_query_groups(
    student_logits: torch.Tensor,
    student_boxes: torch.Tensor,
    targets,
    *,
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
):
    """Freeze query strata from Student-normal predictions.

    TP/FP/ordinary semantics intentionally match the historical NP mechanism
    audit.  These masks must be reused for both frozen-teacher arms; they are
    never recomputed from teacher predictions.
    """
    if student_logits.ndim != 3 or student_boxes.shape[:2] != student_logits.shape[:2]:
        raise ValueError("expected logits [B,Q,C] and boxes [B,Q,4]")
    batch, queries = student_logits.shape[:2]
    if len(targets) != batch:
        raise ValueError("target batch does not match predictions")
    scores, classes = student_logits.sigmoid().max(-1)
    tp = torch.zeros((batch, queries), dtype=torch.bool, device=student_logits.device)
    high_fp = torch.zeros_like(tp)
    matched_gt = torch.full((batch, queries), -1, dtype=torch.long, device=student_logits.device)
    matched_iou = student_logits.new_zeros((batch, queries))
    matched_class = torch.full_like(matched_gt, -1)
    for index, target in enumerate(targets):
        gt_boxes = target["boxes"].as_subclass(torch.Tensor).to(student_boxes.device)
        gt_labels = target["labels"].to(student_logits.device)
        if gt_boxes.numel():
            iou = _box_iou(box_cxcywh_to_xyxy(student_boxes[index]), box_cxcywh_to_xyxy(gt_boxes))
            best_iou, best_gt = iou.max(-1)
            labels = gt_labels[best_gt]
            matched_gt[index] = best_gt
            matched_iou[index] = best_iou
            matched_class[index] = labels
            tp[index] = (
                (scores[index] >= score_threshold)
                & (best_iou >= iou_threshold)
                & (classes[index] == labels)
            )
        high_fp[index] = (scores[index] >= score_threshold) & ~tp[index]
    ordinary = ~(tp | high_fp)
    return {
        "all": torch.ones_like(tp),
        "tp": tp,
        "high_score_fp": high_fp,
        "ordinary_unmatched": ordinary,
        "student_score": scores,
        "student_class": classes,
        "matched_gt": matched_gt,
        "matched_gt_class": matched_class,
        "matched_iou": matched_iou,
    }


def grouped_loss_accounting(per_query_loss: torch.Tensor, mask: torch.Tensor):
    """Report a group mean and its contribution under the original denominator."""
    if per_query_loss.shape != mask.shape:
        raise ValueError("loss and group mask shapes differ")
    selected = per_query_loss[mask]
    group_sum = selected.sum()
    total_sum = per_query_loss.sum()
    denominator = per_query_loss.numel()
    return {
        "count": int(mask.sum()),
        "global_count": int(denominator),
        "group_sum": float(group_sum.detach()),
        "group_mean": float(selected.mean().detach()) if selected.numel() else None,
        "global_denominator_contribution": float((group_sum / denominator).detach()),
        "fraction_of_total_loss": (
            float((group_sum / total_sum).detach())
            if float(total_sum.detach()) != 0.0 else None
        ),
    }


def gradient_pair_stats(det_gradients, kd_gradients):
    """Norm/cosine audit that distinguishes missing from exact-zero gradients."""
    if len(det_gradients) != len(kd_gradients):
        raise ValueError("gradient lists do not share parameter layout")
    det_missing = all(value is None for value in det_gradients)
    kd_missing = all(value is None for value in kd_gradients)
    det_parts, kd_parts = [], []
    for det_value, kd_value in zip(det_gradients, kd_gradients):
        if det_value is None and kd_value is None:
            continue
        if det_value is None:
            det_value = torch.zeros_like(kd_value)
        if kd_value is None:
            kd_value = torch.zeros_like(det_value)
        if det_value.shape != kd_value.shape:
            raise ValueError("paired gradients have different shapes")
        det_parts.append(det_value.detach().float().reshape(-1))
        kd_parts.append(kd_value.detach().float().reshape(-1))
    det = torch.cat(det_parts) if det_parts else None
    kd = torch.cat(kd_parts) if kd_parts else None
    det_norm = None if det_missing else float(det.norm())
    kd_norm = None if kd_missing else float(kd.norm())

    def status(value, norm):
        if value is None:
            return "missing"
        if not math.isfinite(norm):
            return "nonfinite"
        return "zero" if norm == 0.0 else "nonzero"

    det_status = "missing" if det_missing else status(det, det_norm)
    kd_status = "missing" if kd_missing else status(kd, kd_norm)
    cosine = None
    if det_status == kd_status == "nonzero":
        cosine = float(torch.dot(det, kd) / (det.norm() * kd.norm()))
    return {
        "det_status": det_status,
        "kd_status": kd_status,
        "det_norm": det_norm,
        "kd_norm": kd_norm,
        "kd_over_det": kd_norm / det_norm if det_status == kd_status == "nonzero" else None,
        "cosine": cosine,
    }
