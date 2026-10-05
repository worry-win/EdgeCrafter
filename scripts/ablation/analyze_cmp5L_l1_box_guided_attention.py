"""Aggregate full L1-box guided decoder runs and explain TP transitions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ENDPOINTS = (
    "q_iou_equals_q_cls",
    "gt_logit_q_iou",
    "iou_q_cls",
    "rank_q_iou_final_score",
    "top1_top5_iou_gap",
    "class_margin_q_iou",
)


def _mean(values):
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def _rank(values, query):
    return 1 + sum(float(value) > float(values[query]) for value in values)


def _index(payload):
    return {int(record["ann_id"]): record["metrics"] for record in payload["lesions"]}


def summarize_transitions(baseline, guided):
    base = _index(baseline)
    new = _index(guided)
    gained, lost = [], []
    for ann_id, normal in base.items():
        condition = new[ann_id]
        if not normal["joint_success_iou05_score05"] and condition["joint_success_iou05_score05"]:
            query = int(condition["q_detection"])
            normal_margin = normal["query_gt_logit"][query] - normal["query_wrong_logit"][query]
            guided_margin = condition["query_gt_logit"][query] - condition["query_wrong_logit"][query]
            row = {
                "ann_id": ann_id,
                "guided_query": query,
                "same_as_normal_detection_query": query == int(normal["q_detection"]),
                "same_as_normal_best_iou_query": query == int(normal["q_iou"]),
                "normal_final_rank": _rank(normal["query_final_score"], query),
                "guided_final_rank": _rank(condition["query_final_score"], query),
                "normal_iou": normal["query_iou"][query],
                "guided_iou": condition["query_iou"][query],
                "normal_iou_ge_05": normal["query_iou"][query] >= 0.5,
                "normal_class_margin": normal_margin,
                "guided_class_margin": guided_margin,
                "margin_negative_to_positive": normal_margin < 0 <= guided_margin,
            }
            for layer in (2, 3):
                for region in ("foreground", "context"):
                    key = f"l{layer}_{region}_mass"
                    row[f"delta_{key}"] = condition[key][query] - normal[key][query]
            gained.append(row)
        elif normal["joint_success_iou05_score05"] and not condition["joint_success_iou05_score05"]:
            lost.append(ann_id)

    return {
        "gained_count": len(gained),
        "lost_count": len(lost),
        "net_count": len(gained) - len(lost),
        "same_detection_query_fraction": _mean(row["same_as_normal_detection_query"] for row in gained),
        "same_best_iou_query_fraction": _mean(row["same_as_normal_best_iou_query"] for row in gained),
        "normal_iou_ge_05_fraction": _mean(row["normal_iou_ge_05"] for row in gained),
        "margin_negative_to_positive_fraction": _mean(row["margin_negative_to_positive"] for row in gained),
        "mean_normal_final_rank": _mean(row["normal_final_rank"] for row in gained),
        "mean_guided_final_rank": _mean(row["guided_final_rank"] for row in gained),
        "mean_rank_change_guided_minus_normal": _mean(
            row["guided_final_rank"] - row["normal_final_rank"] for row in gained
        ),
        "mean_normal_iou": _mean(row["normal_iou"] for row in gained),
        "mean_guided_iou": _mean(row["guided_iou"] for row in gained),
        "mean_normal_class_margin": _mean(row["normal_class_margin"] for row in gained),
        "mean_guided_class_margin": _mean(row["guided_class_margin"] for row in gained),
        "mean_delta_l2_foreground_mass": _mean(row["delta_l2_foreground_mass"] for row in gained),
        "mean_delta_l2_context_mass": _mean(row["delta_l2_context_mass"] for row in gained),
        "mean_delta_l3_foreground_mass": _mean(row["delta_l3_foreground_mass"] for row in gained),
        "mean_delta_l3_context_mass": _mean(row["delta_l3_context_mass"] for row in gained),
        "gained_records": gained,
        "lost_ann_ids": lost,
    }


def _binding_summary(payload):
    metrics = [record["metrics"] for record in payload["lesions"]]
    return {key: _mean(metric[key] for metric in metrics) for key in ENDPOINTS}


def render_markdown(summary):
    lines = [
        "# L1 predicted-box guided L2/L3 attention",
        "",
        "| Condition | AP | AP50 | AP75 | Precision | Recall | F1 | AR@100 | ΔAP50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in summary["conditions"]:
        row = condition["row"]
        lines.append(
            f"| {row['condition']} | {row['map']:.4f} | {row['map50']:.4f} | "
            f"{row['map75']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['ar100']:.4f} | {condition['delta_map50']:+.4f} |"
        )
    lines.extend([
        "",
        "| Guided condition | New correct | Lost correct | Net | Same detection query | Normal IoU≥.5 | Margin −→+ | Rank Δ | Δ L2 FG mass | Δ L3 FG mass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for condition in summary["conditions"][1:]:
        transition = condition["transitions"]
        lines.append(
            f"| {condition['row']['condition']} | {transition['gained_count']} | "
            f"{transition['lost_count']} | {transition['net_count']:+d} | "
            f"{transition['same_detection_query_fraction'] or 0:.3f} | "
            f"{transition['normal_iou_ge_05_fraction'] or 0:.3f} | "
            f"{transition['margin_negative_to_positive_fraction'] or 0:.3f} | "
            f"{transition['mean_rank_change_guided_minus_normal'] or 0:+.2f} | "
            f"{transition['mean_delta_l2_foreground_mass'] or 0:+.4f} | "
            f"{transition['mean_delta_l3_foreground_mass'] or 0:+.4f} |"
        )
    return "\n".join(lines) + "\n"


def main(args):
    payloads = [json.loads(Path(path).read_text()) for path in args.inputs]
    baseline = next(payload for payload in payloads if payload["meta"]["condition"] == "baseline")
    base_ap50 = baseline["row"]["map50"]
    ordered_names = [
        "baseline", "soft_l2", "soft_l3", "soft_l2_l3",
        "hard_l2", "hard_l3", "hard_l2_l3",
    ]
    by_name = {payload["meta"]["condition"]: payload for payload in payloads}
    if set(by_name) != set(ordered_names):
        raise RuntimeError(f"condition mismatch: {sorted(by_name)}")
    conditions = []
    for name in ordered_names:
        payload = by_name[name]
        item = {
            "row": payload["row"],
            "delta_map50": payload["row"]["map50"] - base_ap50,
            "binding": _binding_summary(payload),
        }
        if name != "baseline":
            item["transitions"] = summarize_transitions(baseline, payload)
        conditions.append(item)
    summary = {
        "meta": {
            "n_images": baseline["meta"]["n_images"],
            "n_lesions": baseline["meta"]["n_lesions"],
            "checkpoint": baseline["meta"]["checkpoint"],
            "checkpoint_weights": baseline["meta"]["checkpoint_weights"],
        },
        "conditions": conditions,
    }
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    markdown = render_markdown(summary)
    Path(args.markdown_out).write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--markdown-out", required=True)
    main(parser.parse_args())
