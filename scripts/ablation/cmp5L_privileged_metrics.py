"""Small, dependency-free output helpers for privileged full evaluation."""


def metrics_row(condition, stats, n_images):
    pr = stats["yolo_f1_iou50"]
    return {
        "condition": condition,
        "n_images": int(n_images),
        "map50": float(stats["coco_eval_bbox"][1]),
        "precision": float(pr["precision"]),
        "recall": float(pr["recall"]),
        "f1": float(pr["f1"]),
        "macro_f1_iou50": float(stats["macro_f1_iou50"]),
    }


def render_markdown(rows):
    normal = next(row for row in rows if row["condition"] == "normal")
    lines = [
        "# cmp5L Privileged Full-Test Metrics",
        "",
        "| Condition | mAP50 | ΔmAP50 | Precision | Recall | F1 | ΔF1 | Macro-F1@50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['condition']} | {row['map50']:.4f} | "
            f"{row['map50'] - normal['map50']:+.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} | "
            f"{row['f1'] - normal['f1']:+.4f} | {row['macro_f1_iou50']:.4f} |"
        )
    return "\n".join(lines) + "\n"
