"""Numpy-only diagnosis of cmp5L lesion-level decoder behavior dumps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ARMS = ("dinov2s", "dinov2b", "lw_xlarge", "mae_vitb", "ecvits_official")
QUERY = {"q_iou": 0, "q_cls": 1, "q_global": 2, "q_detection": 3,
         "q_hungarian": 4, "q_random": 5}


def summarize(values, seed=42, bootstrap=2000):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "mean": None, "median": None, "iqr": [None, None],
                "ci95": [None, None], "std": None}
    rng = np.random.default_rng(seed)
    means = np.empty(bootstrap, dtype=np.float64)
    for start in range(0, bootstrap, 200):
        count = min(200, bootstrap - start)
        sample = rng.integers(0, len(values), size=(count, len(values)))
        means[start:start + count] = values[sample].mean(1)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "iqr": [float(v) for v in np.quantile(values, [0.25, 0.75])],
        "ci95": [float(v) for v in np.quantile(means, [0.025, 0.975])],
        "std": float(values.std()),
    }


def paired_summary(a, b, seed=42):
    delta = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    result = summarize(delta, seed)
    finite = delta[np.isfinite(delta)]
    result["effect_size_paired_dz"] = (
        float(finite.mean() / finite.std(ddof=1)) if len(finite) > 1 and finite.std(ddof=1) else None
    )
    return result


def balanced_group_difference(a, b, seed=42, bootstrap=2000):
    """Equal-size bootstrap difference for two independent lesion groups."""
    a = np.asarray(a, np.float64); a = a[np.isfinite(a)]
    b = np.asarray(b, np.float64); b = b[np.isfinite(b)]
    n = min(len(a), len(b))
    if not n:
        return {"n_per_group": 0, "mean_difference": None, "ci95": [None, None],
                "effect_size_cohen_d": None}
    rng = np.random.default_rng(seed)
    differences = np.empty(bootstrap)
    for i in range(bootstrap):
        differences[i] = rng.choice(a, n, replace=True).mean() - rng.choice(b, n, replace=True).mean()
    pooled = np.sqrt(((a.var(ddof=1) if len(a) > 1 else 0) +
                      (b.var(ddof=1) if len(b) > 1 else 0)) / 2)
    return {
        "n_per_group": int(n),
        "mean_difference": float(a.mean() - b.mean()),
        "ci95": [float(v) for v in np.quantile(differences, [0.025, 0.975])],
        "effect_size_cohen_d": float((a.mean() - b.mean()) / pooled) if pooled else None,
    }


def cxcywh_to_xyxy(boxes):
    boxes = np.asarray(boxes)
    result = np.empty_like(boxes)
    result[..., :2] = boxes[..., :2] - boxes[..., 2:] / 2
    result[..., 2:] = boxes[..., :2] + boxes[..., 2:] / 2
    return result


def aligned_iou(boxes, gt):
    boxes = np.asarray(boxes)
    gt = np.asarray(gt)
    left_top = np.maximum(boxes[..., :2], gt[..., :2])
    right_bottom = np.minimum(boxes[..., 2:], gt[..., 2:])
    intersection = np.clip(right_bottom - left_top, 0, None).prod(-1)
    area_a = np.clip(boxes[..., 2:] - boxes[..., :2], 0, None).prod(-1)
    area_b = np.clip(gt[..., 2:] - gt[..., :2], 0, None).prod(-1)
    return intersection / np.clip(area_a + area_b - intersection, 1e-9, None)


def cka(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    x -= x.mean(0, keepdims=True)
    y -= y.mean(0, keepdims=True)
    cross = x.T @ y
    denom = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    return float((cross * cross).sum() / denom) if denom else None


def rankdata(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def rsa(x, y, max_samples=250, seed=42):
    rng = np.random.default_rng(seed)
    if len(x) > max_samples:
        ids = rng.choice(len(x), max_samples, replace=False)
        x, y = x[ids], y[ids]
    dx = np.linalg.norm(x[:, None] - x[None], axis=-1)
    dy = np.linalg.norm(y[:, None] - y[None], axis=-1)
    tri = np.triu_indices(len(x), 1)
    return float(np.corrcoef(rankdata(dx[tri]), rankdata(dy[tri]))[0, 1])


def effective_rank(matrix):
    matrix = np.asarray(matrix, np.float64)
    matrix = matrix - matrix.mean(0, keepdims=True)
    singular = np.linalg.svd(matrix, compute_uv=False)
    energy = singular * singular
    return float(energy.sum() ** 2 / np.clip((energy * energy).sum(), 1e-12, None))


def set_chamfer(points_a, points_b):
    distance = np.linalg.norm(points_a[:, None] - points_b[None], axis=-1)
    return float(0.5 * (distance.min(1).mean() + distance.min(0).mean()))


def weighted_histogram_js(points_a, weights_a, points_b, weights_b):
    edges = np.linspace(-1.5, 1.5, 10)
    hist_a, _, _ = np.histogram2d(
        points_a[:, 0], points_a[:, 1], bins=(edges, edges), weights=weights_a
    )
    hist_b, _, _ = np.histogram2d(
        points_b[:, 0], points_b[:, 1], bins=(edges, edges), weights=weights_b
    )
    hist_a = (hist_a.reshape(-1) + 1e-9); hist_a /= hist_a.sum()
    hist_b = (hist_b.reshape(-1) + 1e-9); hist_b /= hist_b.sum()
    middle = 0.5 * (hist_a + hist_b)
    return float(0.5 * ((hist_a * np.log(hist_a / middle)).sum()
                        + (hist_b * np.log(hist_b / middle)).sum()))


def _spatial_metrics(data, query_index):
    locations = data["sampling_locations"][:, query_index].astype(np.float32)
    weights = data["attention_weights"][:, query_index].astype(np.float32)
    gt = data["gt_box_norm"].astype(np.float32)
    x0, y0, x1, y1 = [gt[:, i, None, None, None] for i in range(4)]
    x, y = locations[..., 0], locations[..., 1]
    inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    width = np.clip(x1 - x0, 1e-6, None)
    height = np.clip(y1 - y0, 1e-6, None)
    edge_distance = np.minimum.reduce([
        np.abs(x - x0) / width, np.abs(x - x1) / width,
        np.abs(y - y0) / height, np.abs(y - y1) / height,
    ])
    boundary = inside & (edge_distance <= 0.1)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    context = (
        (x >= cx - width) & (x <= cx + width)
        & (y >= cy - height) & (y <= cy + height) & ~inside
    )
    diagonal = np.sqrt(width * width + height * height)
    center_distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / diagonal
    entropy = -(np.clip(weights, 1e-9, 1) * np.log(np.clip(weights, 1e-9, 1))).sum(-1)
    entropy /= np.log(weights.shape[-1])
    return {
        "inside_ratio": inside.mean((-1, -2)),
        "boundary_ratio": boundary.mean((-1, -2)),
        "context_ratio": context.mean((-1, -2)),
        "attention_inside_mass": (weights * inside).sum(-1).mean(-1),
        "attention_boundary_mass": (weights * boundary).sum(-1).mean(-1),
        "attention_context_mass": (weights * context).sum(-1).mean(-1),
        "attention_entropy": entropy.mean(-1),
        "attention_concentration": weights.max(-1).mean(-1),
        "center_distance": center_distance.mean((-1, -2)),
        "boundary_distance": edge_distance.mean((-1, -2)),
        "inside_by_head": inside.mean(-1),
        "mass_by_head": (weights * inside).sum(-1),
        "inside_points": inside,
        "locations_relative": np.stack(((x - cx) / width, (y - cy) / height), -1),
    }


def _level_metrics(data, spatial, points=(3, 6, 3)):
    result = []
    start = 0
    for level, count in enumerate(points):
        stop = start + count
        weights = data["attention_weights"][:, QUERY["q_hungarian"], :, :, start:stop]
        inside = spatial["inside_points"][..., start:stop]
        result.append({
            "level": level,
            "inside_ratio": summarize(inside.mean((-1, -2)).reshape(-1)),
            "attention_mass": summarize(weights.sum(-1).mean(-1).reshape(-1)),
        })
        start = stop
    return result


def _trajectory(data):
    boxes = cxcywh_to_xyxy(data["refined_boxes"][:, QUERY["q_hungarian"]])
    gt = data["gt_box_norm"][:, None]
    iou = aligned_iou(boxes, gt)
    centers = (boxes[..., :2] + boxes[..., 2:]) / 2
    gt_center = (gt[..., :2] + gt[..., 2:]) / 2
    gt_size = np.clip(gt[..., 2:] - gt[..., :2], 1e-6, None)
    size = boxes[..., 2:] - boxes[..., :2]
    return {
        "iou": iou,
        "delta_iou": np.diff(iou, axis=1, prepend=iou[:, :1]),
        "center_error": np.linalg.norm((centers - gt_center) / gt_size, axis=-1),
        "size_error": np.linalg.norm((size - gt_size) / gt_size, axis=-1),
    }


def query_prototype_metrics(responsible, background, success):
    """Within-model distances to correct-query and random-query prototypes."""
    responsible = np.asarray(responsible, np.float32)
    background = np.asarray(background, np.float32)
    success = np.asarray(success, bool)
    if not success.any():
        correct = responsible.mean(0)
    else:
        correct = responsible[success].mean(0)
    background_prototype = background.mean(0)
    distance_correct = np.linalg.norm(responsible - correct[None], axis=-1)
    distance_background = np.linalg.norm(responsible - background_prototype[None], axis=-1)
    return {
        "distance_to_correct_prototype": distance_correct,
        "distance_to_background_prototype": distance_background,
        "prototype_margin": distance_background - distance_correct,
    }


def _query_metrics(data, success):
    stages = data["query_stages"][:, QUERY["q_hungarian"]].astype(np.float32)
    outputs = stages[:, :, -1]
    random_outputs = data["query_stages"][:, QUERY["q_random"], :, -1].astype(np.float32)
    movement = np.linalg.norm(np.diff(outputs, axis=1, prepend=outputs[:, :1]), axis=-1)
    stage_change = np.linalg.norm(stages[:, :, [2, 4, 5]] - stages[:, :, [0, 2, 4]], axis=-1)
    result = {"norm": np.linalg.norm(outputs, axis=-1), "layer_movement": movement,
              "self_cross_ffn_change": stage_change, "representations": outputs}
    result.update(query_prototype_metrics(outputs, random_outputs, success))
    return result


def _ffn_metrics(data):
    activation = data["ffn_activation"][:, QUERY["q_hungarian"]].astype(np.float32)
    magnitude = np.abs(activation)
    probability = magnitude / np.clip(magnitude.sum(-1, keepdims=True), 1e-9, None)
    entropy = -(probability * np.log(np.clip(probability, 1e-12, 1))).sum(-1) / np.log(activation.shape[-1])
    ordered = np.sort(magnitude * magnitude, axis=-1)[..., ::-1]
    top_count = max(1, activation.shape[-1] // 10)
    top_fraction = ordered[..., :top_count].sum(-1) / np.clip(ordered.sum(-1), 1e-9, None)
    cumulative = np.cumsum(ordered, axis=-1) / np.clip(ordered.sum(-1, keepdims=True), 1e-9, None)
    result = {"norm": np.linalg.norm(activation, axis=-1),
            "near_zero_fraction": (magnitude < 1e-4).mean(-1),
            "entropy": entropy, "top10pct_energy": top_fraction,
            "active50_fraction": (np.argmax(cumulative >= 0.5, axis=-1) + 1) / activation.shape[-1],
            "active90_fraction": (np.argmax(cumulative >= 0.9, axis=-1) + 1) / activation.shape[-1],
            "activation": activation}
    if "ffn_grad_importance" in data:
        importance = data["ffn_grad_importance"].astype(np.float32)
        valid = np.isfinite(importance).any(axis=-1)
        total = np.nansum(importance, axis=-1)
        probability_i = importance / np.where(total[..., None] > 0, total[..., None], np.nan)
        importance_norm = np.sqrt(np.nansum(importance * importance, axis=-1))
        importance_entropy = -np.nansum(
            probability_i * np.log(np.clip(probability_i, 1e-12, 1)), axis=-1
        ) / np.log(importance.shape[-1])
        importance_norm[~valid] = np.nan
        importance_entropy[~valid] = np.nan
        result["gradient_importance_norm"] = importance_norm
        result["gradient_importance_entropy"] = importance_entropy
    return result


def _auc(labels, scores):
    labels = np.asarray(labels, bool)
    pos, neg = labels.sum(), (~labels).sum()
    if not pos or not neg:
        return None
    ranks = rankdata(scores) + 1
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def logistic_probe(features, labels, seed=42):
    x = np.asarray(features, np.float64)
    y = np.asarray(labels, np.float64)
    x = np.nan_to_num(x, nan=np.nanmedian(x, axis=0), posinf=0, neginf=0)
    mean, std = x.mean(0), x.std(0)
    x = (x - mean) / np.where(std > 1e-9, std, 1)
    rng = np.random.default_rng(seed)
    aucs, coefs = [], []
    fold_id = np.empty(len(y), dtype=np.int8)
    permutation = rng.permutation(len(y))
    fold_id[permutation] = np.arange(len(y)) % 5
    for fold in range(5):
        test = fold_id == fold
        train = ~test
        w = np.zeros(x.shape[1])
        bias = 0.0
        for _ in range(600):
            logits = np.clip(x[train] @ w + bias, -30, 30)
            prob = 1 / (1 + np.exp(-logits))
            error = prob - y[train]
            w -= 0.05 * ((x[train].T @ error) / train.sum() + 0.01 * w)
            bias -= 0.05 * error.mean()
        scores = x[test] @ w + bias
        aucs.append(_auc(y[test].astype(bool), scores))
        coefs.append(w)
    univariate = []
    for column in range(x.shape[1]):
        auc = _auc(y.astype(bool), x[:, column])
        univariate.append(max(auc, 1.0 - auc) if auc is not None else None)
    return {"auc_mean": float(np.mean(aucs)), "auc_folds": aucs,
            "mean_abs_standardized_coefficient": np.abs(coefs).mean(0).tolist(),
            "univariate_auc_orientation_free": univariate}


def _group_summaries(metrics, groups, seed):
    result = {}
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        result[group] = {
            name: [summarize(values[mask, layer], seed + layer)
                   for layer in range(values.shape[1])]
            for name, values in metrics.items() if values.ndim == 2
        }
    return result


def _permutation_null(data, spatial, seed=42):
    rng = np.random.default_rng(seed)
    area = data["gt_area"]
    bins = np.digitize(area, np.quantile(area, [0.25, 0.5, 0.75]))
    perm = np.arange(len(area))
    for bucket in range(4):
        ids = np.flatnonzero(bins == bucket)
        perm[ids] = rng.permutation(ids)
    locations = data["sampling_locations"][:, QUERY["q_hungarian"]].astype(np.float32)
    shuffled_gt = data["gt_box_norm"][perm]
    x, y = locations[..., 0], locations[..., 1]
    x0, y0, x1, y1 = [shuffled_gt[:, i, None, None, None] for i in range(4)]
    inside = ((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)).mean((-1, -2))
    shuffled_iou = aligned_iou(
        cxcywh_to_xyxy(data["refined_boxes"][:, QUERY["q_hungarian"]]),
        shuffled_gt[:, None],
    )
    return {"sampling_inside": inside, "box_iou": shuffled_iou,
            "observed_sampling_inside": spatial["inside_ratio"]}


def analyze(dump_dir, split, seed=42):
    dump_dir = Path(dump_dir)
    data = {arm: dict(np.load(dump_dir / f"{arm}_{split}.npz", allow_pickle=True)) for arm in ARMS}
    reference = data[ARMS[0]]
    for arm in ARMS[1:]:
        if not np.array_equal(reference["ann_id"], data[arm]["ann_id"]):
            raise ValueError("arm dumps are not lesion-aligned")
    groups = reference["group"]
    result = {"n": len(groups), "groups": {g: int((groups == g).sum()) for g in sorted(set(groups))},
              "arms": {}, "teacher_student": {}, "cross_model": {}, "null_baselines": {}}
    derived = {}
    for arm_index, arm in enumerate(ARMS):
        item = data[arm]
        spatial = _spatial_metrics(item, QUERY["q_hungarian"])
        random_spatial = _spatial_metrics(item, QUERY["q_random"])
        trajectory = _trajectory(item)
        query = _query_metrics(item, item["arm_success"][:, arm_index])
        ffn = _ffn_metrics(item)
        gt_class = item["gt_class"].astype(int)
        q_iou_logits = item["lqe_logits"][:, QUERY["q_iou"], -1]
        q_cls_logits = item["lqe_logits"][:, QUERY["q_cls"], -1]
        q_hung_logits = item["lqe_logits"][:, QUERY["q_hungarian"], -1]
        margin = q_hung_logits[np.arange(len(item["ann_id"])), gt_class] - np.max(
            np.where(np.arange(q_hung_logits.shape[1])[None] == gt_class[:, None], -np.inf, q_hung_logits), axis=1
        )
        binding = {
            "q_iou_equals_q_cls": item["query_ids"][:, 0] == item["query_ids"][:, 1],
            "q_iou_equals_q_hungarian": item["query_ids"][:, 0] == item["query_ids"][:, 4],
            "q_iou_gt_logit": q_iou_logits[np.arange(len(gt_class)), gt_class],
            "q_cls_final_iou": item["final_iou"][:, QUERY["q_cls"]],
            "q_hungarian_margin": margin,
            "q_iou_high_box_low_class": (
                (item["final_iou"][:, QUERY["q_iou"]] >= 0.5)
                & (q_iou_logits[np.arange(len(gt_class)), gt_class] < 0.0)
            ),
            "q_cls_high_class_low_box": (
                (q_cls_logits[np.arange(len(gt_class)), gt_class] >= 0.0)
                & (item["final_iou"][:, QUERY["q_cls"]] < 0.5)
            ),
            "lqe_logit_delta_hungarian": (
                item["lqe_logits"][:, QUERY["q_hungarian"], -1][np.arange(len(gt_class)), gt_class]
                - item["raw_logits"][:, QUERY["q_hungarian"], -1][np.arange(len(gt_class)), gt_class]
            ),
        }
        result["arms"][arm] = {
            "trajectory_by_group": _group_summaries(trajectory, groups, seed),
            "spatial_by_group": _group_summaries({k: v for k, v in spatial.items() if v.ndim == 2}, groups, seed),
            "query_by_group": _group_summaries({k: v for k, v in query.items() if v.ndim == 2}, groups, seed),
            "ffn_by_group": _group_summaries({k: v for k, v in ffn.items() if v.ndim == 2}, groups, seed),
            "binding": {name: summarize(value, seed) for name, value in binding.items()},
            "level_metrics": _level_metrics(item, spatial),
            "head_inside_mass_sorted": {
                group: [
                    np.sort(spatial["mass_by_head"][groups == group, layer], axis=1).mean(0).tolist()
                    for layer in range(spatial["mass_by_head"].shape[1])
                ] for group in sorted(set(groups))
            },
            "ffn_effective_rank": {
                group: [effective_rank(ffn["activation"][groups == group, layer])
                        for layer in range(ffn["activation"].shape[1])]
                for group in sorted(set(groups)) if (groups == group).sum() > 1
            },
            "balanced_group_differences": {
                contrast: {
                    "final_iou": balanced_group_difference(
                        trajectory["iou"][groups == "agreement_easy", -1],
                        trajectory["iou"][groups == contrast, -1], seed,
                    ),
                    "sampling_inside": balanced_group_difference(
                        spatial["inside_ratio"][groups == "agreement_easy", -1],
                        spatial["inside_ratio"][groups == contrast, -1], seed,
                    ),
                    "attention_inside_mass": balanced_group_difference(
                        spatial["attention_inside_mass"][groups == "agreement_easy", -1],
                        spatial["attention_inside_mass"][groups == contrast, -1], seed,
                    ),
                    "prototype_margin": balanced_group_difference(
                        query["prototype_margin"][groups == "agreement_easy", -1],
                        query["prototype_margin"][groups == contrast, -1], seed,
                    ),
                }
                for contrast in ("rank_fixable", "common_failure")
                if (groups == contrast).any()
            },
        }
        null = _permutation_null(item, spatial, seed)
        area_edges = np.quantile(item["gt_area"], [0.25, 0.5, 0.75])
        area_bin = np.digitize(item["gt_area"], area_edges)
        result["null_baselines"][arm] = {
            "responsible_minus_random_inside": [
                paired_summary(spatial["inside_ratio"][:, layer], random_spatial["inside_ratio"][:, layer], seed + layer)
                for layer in range(spatial["inside_ratio"].shape[1])
            ],
            "responsible_minus_permuted_gt_inside": [
                paired_summary(null["observed_sampling_inside"][:, layer], null["sampling_inside"][:, layer], seed + layer)
                for layer in range(spatial["inside_ratio"].shape[1])
            ],
            "box_iou_observed_minus_permuted": [
                paired_summary(trajectory["iou"][:, layer], null["box_iou"][:, layer], seed + layer)
                for layer in range(trajectory["iou"].shape[1])
            ],
            "area_controlled_responsible_minus_random_inside": {
                f"quartile_{bucket + 1}": [
                    paired_summary(
                        spatial["inside_ratio"][area_bin == bucket, layer],
                        random_spatial["inside_ratio"][area_bin == bucket, layer],
                        seed + layer,
                    ) for layer in range(spatial["inside_ratio"].shape[1])
                ] for bucket in range(4)
            },
            "score_topk": {
                "k": [1, 5, 10],
                "sampling_inside": [[summarize(item["score_topk_sampling_inside"][:, k, layer], seed)
                                     for layer in range(item["score_topk_sampling_inside"].shape[2])]
                                    for k in range(3)],
                "box_iou_max": [summarize(item["score_topk_box_iou_max"][:, k], seed) for k in range(3)],
            },
        }
        derived[arm] = {"spatial": spatial, "trajectory": trajectory, "query": query,
                        "ffn": ffn, "binding": binding}

    student_success = reference["arm_success"][:, 0]
    for teacher_index, teacher in enumerate(ARMS[1:], 1):
        mask = ~student_success & reference["arm_success"][:, teacher_index]
        comparison = {}
        for family, names in {
            "trajectory": ("iou", "center_error", "size_error"),
            "spatial": ("inside_ratio", "attention_inside_mass", "attention_entropy", "center_distance"),
            "query": ("norm", "layer_movement", "distance_to_correct_prototype",
                      "distance_to_background_prototype", "prototype_margin"),
            "ffn": tuple(name for name in (
                "norm", "near_zero_fraction", "entropy", "top10pct_energy",
                "gradient_importance_norm", "gradient_importance_entropy",
            ) if name in derived[teacher]["ffn"] and name in derived[ARMS[0]]["ffn"]),
        }.items():
            comparison[family] = {
                name: [paired_summary(derived[teacher][family][name][mask, layer],
                                      derived[ARMS[0]][family][name][mask, layer], seed + layer)
                       for layer in range(derived[teacher][family][name].shape[1])]
                for name in names
            }
        per_level = []
        start = 0
        for level, count in enumerate((3, 6, 3)):
            stop = start + count
            teacher_inside = derived[teacher]["spatial"]["inside_points"][..., start:stop].mean((-1, -2))
            student_inside = derived[ARMS[0]]["spatial"]["inside_points"][..., start:stop].mean((-1, -2))
            teacher_mass = (
                data[teacher]["attention_weights"][:, QUERY["q_hungarian"], ..., start:stop]
                * derived[teacher]["spatial"]["inside_points"][..., start:stop]
            ).sum(-1).mean(-1)
            student_mass = (
                data[ARMS[0]]["attention_weights"][:, QUERY["q_hungarian"], ..., start:stop]
                * derived[ARMS[0]]["spatial"]["inside_points"][..., start:stop]
            ).sum(-1).mean(-1)
            per_level.append({
                "level": level,
                "inside_ratio_teacher_minus_student": [
                    paired_summary(teacher_inside[mask, layer], student_inside[mask, layer], seed + layer)
                    for layer in range(teacher_inside.shape[1])
                ],
                "attention_inside_mass_teacher_minus_student": [
                    paired_summary(teacher_mass[mask, layer], student_mass[mask, layer], seed + layer)
                    for layer in range(teacher_mass.shape[1])
                ],
            })
            start = stop
        result["teacher_student"][teacher] = {
            "n_pairs": int(mask.sum()), **comparison, "spatial_by_level": per_level,
        }

    for teacher in ARMS[1:]:
        pair = f"dinov2s__{teacher}"
        result["cross_model"][pair] = {}
        for group in sorted(set(groups)):
            mask = groups == group
            count = int(mask.sum())
            if count < 3:
                continue
            limit = min(count, 100)
            ids = np.random.default_rng(seed).choice(np.flatnonzero(mask), limit, replace=False)
            result["cross_model"][pair][group] = [
                {"layer": layer,
                 "cka": cka(derived["dinov2s"]["query"]["representations"][ids, layer],
                            derived[teacher]["query"]["representations"][ids, layer]),
                 "rsa": rsa(derived["dinov2s"]["query"]["representations"][ids, layer],
                            derived[teacher]["query"]["representations"][ids, layer], seed=seed),
                 "sampling_chamfer": summarize([
                     set_chamfer(
                         derived["dinov2s"]["spatial"]["locations_relative"][i, layer].reshape(-1, 2),
                         derived[teacher]["spatial"]["locations_relative"][i, layer].reshape(-1, 2),
                     ) for i in ids
                 ], seed + layer),
                 "attention_histogram_js": summarize([
                     weighted_histogram_js(
                         derived["dinov2s"]["spatial"]["locations_relative"][i, layer].reshape(-1, 2),
                         data["dinov2s"]["attention_weights"][i, QUERY["q_hungarian"], layer].reshape(-1),
                         derived[teacher]["spatial"]["locations_relative"][i, layer].reshape(-1, 2),
                         data[teacher]["attention_weights"][i, QUERY["q_hungarian"], layer].reshape(-1),
                     ) for i in ids
                 ], seed + layer),
                 "sorted_attention_weight_l1": summarize([
                     np.abs(
                         np.sort(data["dinov2s"]["attention_weights"][i, QUERY["q_hungarian"], layer].reshape(-1))
                         - np.sort(data[teacher]["attention_weights"][i, QUERY["q_hungarian"], layer].reshape(-1))
                     ).mean() for i in ids
                 ], seed + layer),
                 "levels": [
                     {"level": level,
                      "sampling_chamfer": summarize([
                          set_chamfer(
                              derived["dinov2s"]["spatial"]["locations_relative"][i, layer, :, start:stop].reshape(-1, 2),
                              derived[teacher]["spatial"]["locations_relative"][i, layer, :, start:stop].reshape(-1, 2),
                          ) for i in ids
                      ], seed + layer + level),
                      "attention_histogram_js": summarize([
                          weighted_histogram_js(
                              derived["dinov2s"]["spatial"]["locations_relative"][i, layer, :, start:stop].reshape(-1, 2),
                              data["dinov2s"]["attention_weights"][i, QUERY["q_hungarian"], layer, :, start:stop].reshape(-1),
                              derived[teacher]["spatial"]["locations_relative"][i, layer, :, start:stop].reshape(-1, 2),
                              data[teacher]["attention_weights"][i, QUERY["q_hungarian"], layer, :, start:stop].reshape(-1),
                          ) for i in ids
                      ], seed + layer + level)}
                     for level, (start, stop) in enumerate(((0, 3), (3, 9), (9, 12)))
                 ]}
                for layer in range(derived[teacher]["query"]["representations"].shape[1])
            ]

    student = derived["dinov2s"]
    feature_names = (
        "mean_inside", "final_attention_mass", "final_attention_entropy",
        "final_center_distance", "query_movement", "class_margin",
        "binding_iou_cls", "ffn_norm", "ffn_entropy",
        "ffn_top10_energy", "ffn_active90_fraction",
    )
    features = np.column_stack([
        student["spatial"]["inside_ratio"].mean(1),
        student["spatial"]["attention_inside_mass"][:, -1],
        student["spatial"]["attention_entropy"][:, -1],
        student["spatial"]["center_distance"][:, -1],
        student["query"]["layer_movement"].sum(1),
        student["binding"]["q_hungarian_margin"],
        student["binding"]["q_iou_equals_q_cls"].astype(float),
        student["ffn"]["norm"][:, -1],
        student["ffn"]["entropy"][:, -1],
        student["ffn"]["top10pct_energy"][:, -1],
        student["ffn"]["active90_fraction"][:, -1],
    ])
    probe = logistic_probe(features, student_success, seed)
    probe["features"] = feature_names
    probe["ranked_features"] = [feature_names[i] for i in np.argsort(probe["mean_abs_standardized_coefficient"])[::-1]]
    probe["ranked_univariate_features"] = [
        feature_names[i] for i in np.argsort(probe["univariate_auc_orientation_free"])[::-1]
    ]
    result["student_success_probe"] = probe
    return result


def render_markdown(result):
    lines = ["# cmp5L Internal Behavior — Computed Results", "",
             f"Subset: {result['n']} lesions. Groups: `{result['groups']}`.", "",
             "## Predictive probe", "",
             f"Five-fold AUC: {result['student_success_probe']['auc_mean']:.3f}. Ranked variables: "
             + ", ".join(result["student_success_probe"]["ranked_features"]), "",
             "## Teacher-correct / student-wrong pairs", ""]
    for teacher, item in result["teacher_student"].items():
        lines.append(f"- {teacher}: n={item['n_pairs']}")
    lines.extend(["", "Full layer/group statistics, bootstrap intervals, null baselines, CKA/RSA, "
                  "and effect sizes are stored in the adjacent JSON file.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--markdown-out")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = analyze(args.dump_dir, args.split, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.markdown_out:
        Path(args.markdown_out).write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result))


if __name__ == "__main__":
    main()
