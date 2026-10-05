"""Shared, GT-free predictor utilities for cmp5L BG-dependency experiments."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, using zero-based ranks."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or left.std() == 0 or right.std() == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction - target
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "pearson": _safe_corr(target, prediction),
        "spearman": _safe_corr(_rankdata(target), _rankdata(prediction)),
    }


def binary_metrics(
    target: np.ndarray,
    score: np.ndarray,
    threshold: float = 0.5,
    high_threshold: float = 0.8,
) -> Dict[str, float]:
    """Dependency-free ROC-AUC, average precision, and gate metrics."""
    target = np.asarray(target, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    positives = target == 1
    negatives = ~positives
    n_pos, n_neg = int(positives.sum()), int(negatives.sum())

    ranks = _rankdata(score) + 1.0
    roc_auc = (
        (ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        if n_pos and n_neg
        else 0.0
    )

    order = np.argsort(-score, kind="mergesort")
    ordered_y = target[order]
    cum_tp = np.cumsum(ordered_y)
    precision_curve = cum_tp / np.arange(1, target.size + 1)
    pr_auc = float(precision_curve[ordered_y == 1].mean()) if n_pos else 0.0

    def threshold_metrics(cutoff: float):
        predicted = score >= cutoff
        tp = int(np.logical_and(predicted, positives).sum())
        fp = int(np.logical_and(predicted, negatives).sum())
        fn = int(np.logical_and(~predicted, positives).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    precision, recall, f1 = threshold_metrics(threshold)
    high_precision, high_recall, _ = threshold_metrics(high_threshold)
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_high_confidence": high_precision,
        "recall_high_confidence": high_recall,
        "threshold": float(threshold),
        "high_threshold": float(high_threshold),
        "positive_rate": float(positives.mean()),
    }


def point_suppression_coeff(
    far_bg_probability: torch.Tensor,
    query_dependency: Optional[torch.Tensor] = None,
    enabled: bool = True,
    strength: float = 0.8,
) -> torch.Tensor:
    """Return λ=1-0.8*g (or 1-0.8*d*g) without GT-derived inputs."""
    if not enabled:
        return torch.ones_like(far_bg_probability)
    gate = far_bg_probability
    if query_dependency is not None:
        while query_dependency.ndim < gate.ndim:
            query_dependency = query_dependency.unsqueeze(-1)
        gate = gate * query_dependency
    return (1.0 - strength * gate).clamp(min=1.0 - strength, max=1.0)
