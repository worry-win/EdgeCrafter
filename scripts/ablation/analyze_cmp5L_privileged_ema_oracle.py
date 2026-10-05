"""Analyze paired GT-conditioned feature-privilege oracle results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LESION_METRICS = (
    "q_iou_equals_q_cls",
    "q_iou_equals_q_hungarian",
    "q_cls_equals_q_hungarian",
    "gt_logit_q_iou",
    "wrong_logit_q_iou",
    "class_margin_q_iou",
    "has_iou_05_candidate",
    "joint_success_iou05_score05",
    "max_iou",
    "iou_q_cls",
    "iou_q_hungarian",
    "rank_q_iou_final_score",
    "rank_q_iou_gt_class",
    "top1_best_iou",
    "top5_best_iou",
    "top1_top5_iou_gap",
    "normal_q_iou_current_iou",
    "normal_q_iou_gt_logit",
    "normal_q_iou_wrong_logit",
    "normal_q_iou_class_margin",
    "normal_q_iou_final_rank",
    "normal_q_iou_gt_class_rank",
)

AUX_METRICS = (
    "sampling_inside",
    "attention_foreground_mass",
    "attention_context_mass",
    "query_update_norm",
    "query_norm",
    "ffn_activation_norm",
    "ffn_activation_entropy",
    "ffn_top10pct_energy",
)

IMAGE_METRICS = (
    "all_logit_mean",
    "all_query_score_mean",
    "background_query_score_mean",
    "positive_query_score_mean",
    "background_fp_count_05",
    "background_fp_count_01",
    "top1_query_score",
)


def summarize(values, seed=42, bootstrap=2000):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "mean": None, "median": None, "iqr": [None, None],
                "ci95": [None, None], "std": None}
    rng = np.random.default_rng(seed)
    means = np.empty(bootstrap)
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


def paired_summary(current, normal, seed=42, bootstrap=2000):
    delta = np.asarray(current, np.float64) - np.asarray(normal, np.float64)
    result = summarize(delta, seed, bootstrap)
    finite = delta[np.isfinite(delta)]
    result["effect_size_paired_dz"] = (
        float(finite.mean() / finite.std(ddof=1))
        if len(finite) > 1 and finite.std(ddof=1) else None
    )
    return result


def _lesion_block(records, condition, seed, bootstrap):
    result = {
        metric: summarize(
            [record["conditions"][condition][metric] for record in records],
            seed, bootstrap,
        ) for metric in LESION_METRICS
    }
    result["delta_vs_normal"] = {
        metric: paired_summary(
            [record["conditions"][condition][metric] for record in records],
            [record["conditions"]["normal"][metric] for record in records],
            seed, bootstrap,
        ) for metric in LESION_METRICS
    }
    result["auxiliary_q_iou"] = {}
    for metric in AUX_METRICS:
        current = np.asarray([
            record["conditions"][condition]["auxiliary_q_iou"][metric]
            for record in records
        ], np.float64)
        normal = np.asarray([
            record["conditions"]["normal"]["auxiliary_q_iou"][metric]
            for record in records
        ], np.float64)
        result["auxiliary_q_iou"][metric] = {
            "value": [summarize(current[:, layer], seed + layer, bootstrap)
                      for layer in range(current.shape[1])],
            "delta_vs_normal": [paired_summary(
                current[:, layer], normal[:, layer], seed + layer, bootstrap
            ) for layer in range(current.shape[1])],
        }
    return result


def _image_delta_block(images, condition, is_negative, seed, bootstrap):
    selected = [row for row in images if row["condition"] == condition
                and bool(row["is_negative"]) == is_negative]
    normal_by_id = {
        row["image_id"]: row for row in images
        if row["condition"] == "normal" and bool(row["is_negative"]) == is_negative
    }
    result = {}
    for metric in IMAGE_METRICS:
        current, normal = [], []
        for row in selected:
            reference = normal_by_id[row["image_id"]]
            if row.get(metric) is not None and reference.get(metric) is not None:
                current.append(row[metric]); normal.append(reference[metric])
        result[metric] = paired_summary(current, normal, seed, bootstrap)
    return result


def analyze_payload(payload, seed=42, bootstrap=2000):
    lesions = payload["lesions"]
    conditions = [item["name"] for item in payload["meta"]["conditions"]]
    groups = sorted({record["group"] for record in lesions})
    result = {
        "meta": payload["meta"],
        "n_lesions": len(lesions),
        "groups": {group: sum(record["group"] == group for record in lesions) for group in groups},
        "conditions": {},
        "shortcut": {},
    }
    for condition in conditions:
        result["conditions"][condition] = {
            "overall": _lesion_block(lesions, condition, seed, bootstrap),
            "groups": {
                group: _lesion_block(
                    [record for record in lesions if record["group"] == group],
                    condition, seed, bootstrap,
                ) for group in groups
            },
        }
        if condition != "normal":
            result["shortcut"][condition] = {
                "positive_images": _image_delta_block(
                    payload["images"], condition, False, seed, bootstrap
                ),
                "negative_images": _image_delta_block(
                    payload["images"], condition, True, seed, bootstrap
                ),
            }
    return result


def render_markdown(result):
    lines = [
        "# Privileged EMA Oracle — Computed Results",
        "",
        f"Lesions: {result['n_lesions']}; groups: `{result['groups']}`.",
        "",
        "| Condition | Δ q_iou=q_cls | Δ GT logit(q_iou) | Δ margin | Δ final rank | Δ max IoU |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition, block in result["conditions"].items():
        delta = block["overall"]["delta_vs_normal"]
        values = [
            delta["q_iou_equals_q_cls"]["mean"],
            delta["gt_logit_q_iou"]["mean"],
            delta["class_margin_q_iou"]["mean"],
            delta["rank_q_iou_final_score"]["mean"],
            delta["max_iou"]["mean"],
        ]
        lines.append("| " + condition + " | " + " | ".join(
            "—" if value is None else f"{value:+.4f}" for value in values
        ) + " |")
    lines.extend(["", "Full group/layer/bootstrap statistics are in the adjacent JSON file.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--markdown-out")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = analyze_payload(payload, args.seed, args.bootstrap)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_markdown(result)
    if args.markdown_out:
        Path(args.markdown_out).write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
