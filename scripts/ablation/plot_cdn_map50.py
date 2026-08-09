#!/usr/bin/env python3
"""Plot the per-epoch AP50 comparison for EC-full and no-CDN."""

import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "docs" / "assets"
RUNS = {
    "EC-full": ROOT
    / "outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_ec_full_ignore9/log.txt",
    "no-CDN": ROOT
    / "outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_no_cdn_ignore9/log.txt",
}
COLORS = {
    "EC-full": "#0072B2",
    "no-CDN": "#D55E00",
    "no-FDR decode": "#009E73",
    "no-GO-DDF": "#CC79A7",
    "no-FDR + no-GO-DDF": "#E69F00",
    "no-Mosaic + no-MAL": "#D55E00",
}


def load_ap50(path):
    values = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            bbox_metrics = record.get("test_coco_eval_bbox")
            if bbox_metrics is None or len(bbox_metrics) < 2:
                continue
            values[int(record["epoch"])] = float(bbox_metrics[1])
    if not values:
        raise ValueError(f"No AP50 values found in {path}")
    return sorted(values.items())


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def text_size(draw, value, text_font):
    box = draw.textbbox((0, 0), value, font=text_font)
    return box[2] - box[0], box[3] - box[1]


def draw_dashed_line(draw, xy, fill, width=2, dash=10, gap=8):
    x1, y1, x2, y2 = xy
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    position = 0.0
    while position < length:
        end = min(position + dash, length)
        draw.line(
            (
                x1 + dx * position,
                y1 + dy * position,
                x1 + dx * end,
                y1 + dy * end,
            ),
            fill=fill,
            width=width,
        )
        position += dash + gap


def write_csv(series, path):
    by_name = {name: dict(values) for name, values in series.items()}
    epochs = sorted({epoch for values in series.values() for epoch, _ in values})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        names = list(series)
        writer.writerow(["epoch"] + [name.lower().replace(" ", "_").replace("+", "plus") + "_ap50" for name in names])
        for epoch in epochs:
            writer.writerow([epoch] + [by_name[name].get(epoch, "") for name in names])


def plot(series, path, title, subtitle, source):
    width, height = 1800, 1080
    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image)

    left, top, right, bottom = 165, 185, 1735, 865
    plot_width, plot_height = right - left, bottom - top
    max_epoch = max(epoch for values in series.values() for epoch, _ in values)
    x_max = int(math.ceil(max_epoch / 10.0) * 10)
    max_ap50 = max(value for values in series.values() for _, value in values)
    y_max = math.ceil((max_ap50 + 0.025) / 0.05) * 0.05

    def point(epoch, value):
        x = left + epoch / x_max * plot_width
        y = bottom - value / y_max * plot_height
        return x, y

    draw.text((left, 48), title, fill="#17212B", font=font(48, True))
    draw.text(
        (left, 112),
        subtitle,
        fill="#5D6873",
        font=font(25),
    )

    tick_font = font(21)
    for tick in range(0, int(round(y_max * 100)) + 1, 5):
        value = tick / 100.0
        _, y = point(0, value)
        draw.line((left, y, right, y), fill="#E3E8ED", width=2)
        label = f"{value:.2f}"
        label_width, label_height = text_size(draw, label, tick_font)
        draw.text((left - label_width - 20, y - label_height / 2), label, fill="#58636E", font=tick_font)

    for epoch in range(0, x_max + 1, 10):
        x, _ = point(epoch, 0)
        draw.line((x, top, x, bottom), fill="#F0F3F6", width=2)
        label = str(epoch)
        label_width, _ = text_size(draw, label, tick_font)
        draw.text((x - label_width / 2, bottom + 18), label, fill="#58636E", font=tick_font)

    draw.line((left, top, left, bottom), fill="#76818C", width=3)
    draw.line((left, bottom, right, bottom), fill="#76818C", width=3)

    axis_font = font(25, True)
    x_label = "Epoch"
    x_width, _ = text_size(draw, x_label, axis_font)
    draw.text(((left + right - x_width) / 2, bottom + 70), x_label, fill="#29333D", font=axis_font)
    y_label = "AP50"
    y_layer = Image.new("RGBA", (200, 70), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_layer)
    y_draw.text((0, 0), y_label, fill="#29333D", font=axis_font)
    y_layer = y_layer.rotate(90, expand=True)
    image.paste(y_layer, (31, int((top + bottom - y_layer.height) / 2)), y_layer)

    names = list(series)
    legend_x = left + 28
    legend_y = top + 22
    for row, name in enumerate(names):
        y = legend_y + row * 50
        draw.line((legend_x, y + 13, legend_x + 62, y + 13), fill=COLORS[name], width=7)
        draw.ellipse((legend_x + 25, y + 5, legend_x + 41, y + 21), fill=COLORS[name])
        draw.text((legend_x + 82, y), name, fill="#26313B", font=font(23, True))

    for row, (name, values) in enumerate(series.items()):
        coordinates = [point(epoch, value) for epoch, value in values]
        draw.line(coordinates, fill=COLORS[name], width=6, joint="curve")
        for x, y in coordinates:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=COLORS[name])

        best_epoch, best_value = max(values, key=lambda item: item[1])
        best_x, best_y = point(best_epoch, best_value)
        draw.ellipse(
            (best_x - 11, best_y - 11, best_x + 11, best_y + 11),
            fill="#FFFFFF",
            outline=COLORS[name],
            width=5,
        )
        annotation = f"{name} best: {best_value:.3f} @ {best_epoch}"
        annotation_font = font(21, True)
        annotation_width, annotation_height = text_size(draw, annotation, annotation_font)
        annotation_x = min(best_x + 18, right - annotation_width - 12)
        annotation_y = top + 7 + row * 45
        draw.rounded_rectangle(
            (
                annotation_x - 9,
                annotation_y - 6,
                annotation_x + annotation_width + 9,
                annotation_y + annotation_height + 7,
            ),
            radius=5,
            fill="#FFFFFF",
            outline=COLORS[name],
            width=2,
        )
        draw.text((annotation_x, annotation_y), annotation, fill=COLORS[name], font=annotation_font)

        last_epoch, last_value = values[-1]
        last_x, last_y = point(last_epoch, last_value)
        draw_dashed_line(draw, (last_x, last_y, last_x, bottom), COLORS[name], width=2, dash=8, gap=7)

    footer_font = font(20)
    draw.text(
        (left, 1018),
        source,
        fill="#6A747E",
        font=footer_font,
    )
    image.save(path, optimize=True)


def main():
    series = {name: load_ap50(path) for name, path in RUNS.items()}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUTPUT_DIR / "cdn_map50_training_curve.png"
    csv_path = OUTPUT_DIR / "cdn_map50_training_curve.csv"
    plot(
        series,
        plot_path,
        "EC-full vs no-CDN: AP50 during training",
        "Strict 9-class setting | one evaluation per epoch | raw values (no smoothing)",
        "Source: outputs/ablation/.../log.txt | Metric: test_coco_eval_bbox[1] (COCO AP50)",
    )
    write_csv(series, csv_path)

    for name, values in series.items():
        best_epoch, best_value = max(values, key=lambda item: item[1])
        print(
            f"{name}: {len(values)} epochs (0-{values[-1][0]}), "
            f"best AP50={best_value:.6f} at epoch {best_epoch}"
        )
    print(plot_path)
    print(csv_path)


if __name__ == "__main__":
    main()
