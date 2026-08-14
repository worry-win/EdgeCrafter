from typing import Mapping


def format_yolo_per_class_metrics_table(
    yolo_per_class: Mapping,
) -> str:
    if not yolo_per_class:
        return None

    header = '[YOLO Per Class]'
    columns = (
        f"{'class':<18}"
        f"{'P':>8}"
        f"{'R':>8}"
        f"{'F1@.50':>8}"
        f"{'F1@.95':>8}"
        f"{'F1@50:95':>10}"
        f"{'mAP@50':>9}"
        f"{'Conf':>9}"
    )

    rows = []
    for cat_id, metrics in sorted(
        yolo_per_class.items(),
        key=lambda item: int(item[0]),
    ):
        label = f"{int(cat_id)}: {metrics.get('name', cat_id)}"
        rows.append(
            f"{label:<18}"
            f"{float(metrics.get('precision', 0.0)):>8.4f}"
            f"{float(metrics.get('recall', 0.0)):>8.4f}"
            f"{float(metrics.get('f1', 0.0)):>8.4f}"
            f"{float(metrics.get('f1_iou95', 0.0)):>8.4f}"
            f"{float(metrics.get('f1_iou50_95', 0.0)):>10.4f}"
            f"{float(metrics.get('map50', 0.0)):>9.4f}"
            f"{float(metrics.get('confidence', 0.0)):>9.4f}"
        )

    return '\n'.join([header, columns, *rows])
