"""Summarize GT-attention and Oracle-gated L1 experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EC_ROOT))

from scripts.ablation.analyze_cmp5L_l1_box_guided_attention import summarize_transitions


BINDING_KEYS = (
    "q_iou_equals_q_cls", "gt_logit_q_iou", "iou_q_cls",
    "rank_q_iou_final_score", "top1_top5_iou_gap", "class_margin_q_iou",
)


def _load(path):
    return json.loads(Path(path).read_text())


def _index(payload):
    return {int(record["ann_id"]): record["metrics"] for record in payload["lesions"]}


def _mean(values):
    values = list(values)
    return sum(float(value) for value in values) / len(values)


def mechanism_summary(baseline, condition):
    base = _index(baseline)
    guided = _index(condition)
    binding = {
        key: _mean(metric[key] for metric in guided.values())
        for key in BINDING_KEYS
    }
    binding_delta = {
        key: binding[key] - _mean(metric[key] for metric in base.values())
        for key in BINDING_KEYS
    }
    attention = {key: [] for key in ("l2_fg", "l2_context", "l3_fg", "l3_context")}
    gated_q_iou = []
    for ann_id, metric in guided.items():
        normal = base[ann_id]
        query = int(metric["q_iou"])
        attention["l2_fg"].append(metric["l2_foreground_mass"][query] - normal["l2_foreground_mass"][query])
        attention["l2_context"].append(metric["l2_context_mass"][query] - normal["l2_context_mass"][query])
        attention["l3_fg"].append(metric["l3_foreground_mass"][query] - normal["l3_foreground_mass"][query])
        attention["l3_context"].append(metric["l3_context_mass"][query] - normal["l3_context_mass"][query])
        if "oracle_gate" in metric:
            gated_q_iou.append(metric["oracle_gate"][query])
    return {
        "binding": binding,
        "binding_delta": binding_delta,
        "attention_delta_at_guided_q_iou": {key: _mean(values) for key, values in attention.items()},
        "guided_q_iou_gate_fraction": _mean(gated_q_iou) if gated_q_iou else None,
        "transitions": summarize_transitions(baseline, condition),
    }


def _condition_item(group, payload, baseline):
    row = payload["row"]
    return {
        "group": group,
        "condition": row["condition"],
        "row": row,
        "delta": {
            key: row[key] - baseline["row"][key]
            for key in ("map", "map50", "map75", "precision", "recall", "f1", "ar100")
        },
        "gated_fraction": payload["meta"].get("gated_fraction"),
        "mechanism": mechanism_summary(baseline, payload),
    }


def _table(lines, title, items):
    lines.extend([
        f"## {title}", "",
        "| Condition | AP | AP50 | AP75 | Precision | Recall | F1 | AR | ΔAP50 | ΔF1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for item in items:
        row, delta = item["row"], item["delta"]
        lines.append(
            f"| {item['condition']} | {row['map']:.4f} | {row['map50']:.4f} | {row['map75']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['ar100']:.4f} | "
            f"{delta['map50']:+.4f} | {delta['f1']:+.4f} |"
        )
    lines.append("")


def render_markdown(summary):
    base = summary["baseline"]
    lines = [
        "# Attention Locality Oracle Decision", "",
        "| Baseline | AP | AP50 | AP75 | Precision | Recall | F1 | AR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| unchanged | {base['map']:.4f} | {base['map50']:.4f} | {base['map75']:.4f} | "
        f"{base['precision']:.4f} | {base['recall']:.4f} | {base['f1']:.4f} | {base['ar100']:.4f} |",
        "",
    ]
    _table(lines, "Ungated L1", summary["groups"]["ungated"])
    _table(lines, "GT-attention Oracle", summary["groups"]["gt_attention"])
    _table(lines, "Gate A: localization", summary["groups"]["gate_a"])
    _table(lines, "Gate B: localization + classification", summary["groups"]["gate_b"])
    lines.extend([
        "## Fixed score=0.5 TP/FP/FN", "",
        "| Mode | TP | FP | FN | ΔTP | ΔFP | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for item in summary["fixed_threshold_counts"]:
        lines.append(
            f"| {item['mode']} | {item['tp']} | {item['fp']} | {item['fn']} | "
            f"{item['delta_tp']:+d} | {item['delta_fp']:+d} | {item['precision']:.4f} | "
            f"{item['recall']:.4f} | {item['f1']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main(args):
    baseline = _load(args.baseline)
    groups = {}
    for group, paths in (
        ("ungated", args.ungated),
        ("gt_attention", args.gt_attention),
        ("gate_a", args.gate_a),
        ("gate_b", args.gate_b),
    ):
        payloads = [_load(path) for path in paths]
        groups[group] = [_condition_item(group, payload, baseline) for payload in payloads]
        groups[group].sort(key=lambda item: item["condition"])
    counts = [_load(path) for path in args.counts]
    base_counts = next(item for item in counts if item["mode"] == "baseline")
    for item in counts:
        item["delta_tp"] = int(item["tp"] - base_counts["tp"])
        item["delta_fp"] = int(item["fp"] - base_counts["fp"])
        item["delta_fn"] = int(item["fn"] - base_counts["fn"])
    counts.sort(key=lambda item: item["mode"])
    summary = {
        "meta": {
            "n_images": baseline["meta"]["n_images"],
            "n_lesions": baseline["meta"]["n_lesions"],
            "checkpoint": baseline["meta"]["checkpoint"],
            "checkpoint_weights": baseline["meta"]["checkpoint_weights"],
            "actual_layers": {"L2": 1, "L3": 2, "L4_unchanged": 3},
        },
        "baseline": baseline["row"],
        "groups": groups,
        "fixed_threshold_counts": counts,
    }
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    markdown = render_markdown(summary)
    Path(args.markdown_out).write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--ungated", nargs="+", required=True)
    parser.add_argument("--gt-attention", nargs="+", required=True)
    parser.add_argument("--gate-a", nargs="+", required=True)
    parser.add_argument("--gate-b", nargs="+", required=True)
    parser.add_argument("--counts", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--markdown-out", required=True)
    main(parser.parse_args())
