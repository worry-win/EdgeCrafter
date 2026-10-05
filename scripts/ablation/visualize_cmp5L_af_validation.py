"""Create representative A–F validation side-by-side detection visualizations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from scripts.ablation.analyze_cmp5L_af_validation import load_predictions


CLASS_NAMES = {0: "solid", 1: "cyst", 2: "node", 3: "duct"}
CLASS_COLORS = {0: "#ff5a5f", 1: "#00a8e8", 2: "#ffb400", 3: "#c77dff"}


def _draw_panel(image, title, annotations, predictions, score_threshold, duct_only):
    panel = image.copy().convert("RGB")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, panel.width, 24), fill=(0, 0, 0))
    draw.text((6, 6), title, fill="white", font=font)
    for annotation in annotations:
        if duct_only and int(annotation["category_id"]) != 3:
            continue
        x, y, width, height = annotation["bbox"]
        draw.rectangle((x, y, x + width, y + height), outline="#45ff72", width=3)
        draw.text((x + 2, y + 2), f"GT:{CLASS_NAMES[int(annotation['category_id'])]}", fill="#45ff72", font=font)
    for prediction in predictions:
        if prediction["score"] < score_threshold:
            continue
        category = int(prediction["category_id"])
        if duct_only and category != 3:
            continue
        x, y, width, height = prediction["bbox"]
        color = CLASS_COLORS[category]
        draw.rectangle((x, y, x + width, y + height), outline=color, width=2)
        draw.text(
            (x + 2, max(y - 10, 25)),
            f"{CLASS_NAMES[category]} q{prediction['query_id']} {prediction['score']:.2f}",
            fill=color,
            font=font,
        )
    return panel


def visualize(args):
    result_dir = Path(args.result_dir)
    analysis = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
    ground_truth = json.loads(Path(args.ann_file).read_text(encoding="utf-8"))
    image_by_id = {int(image["id"]): image for image in ground_truth["images"]}
    annotations_by_image = {int(image_id): [] for image_id in image_by_id}
    for annotation in ground_truth["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    all_ids = list(image_by_id)
    predictions = {
        condition: load_predictions(result_dir / f"predictions_{condition}.jsonl", all_ids)
        for condition in "ABCDEF"
    }
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    index = []
    for comparison, record in analysis["representatives"].items():
        left, right = record["left"], record["right"]
        for event, image_ids in record.items():
            if not event.endswith("_image_ids"):
                continue
            duct_only = event.startswith("duct_")
            for rank, image_id in enumerate(image_ids, start=1):
                image_info = image_by_id[int(image_id)]
                image_path = Path(args.img_folder) / image_info["file_name"]
                if not image_path.exists():
                    image_path = Path(args.img_folder) / Path(image_info["file_name"]).name
                image = Image.open(image_path)
                left_panel = _draw_panel(
                    image,
                    f"{left} | image {image_id} | {event}",
                    annotations_by_image[int(image_id)],
                    predictions[left][int(image_id)],
                    args.score_threshold,
                    duct_only,
                )
                right_panel = _draw_panel(
                    image,
                    f"{right} | image {image_id} | {event}",
                    annotations_by_image[int(image_id)],
                    predictions[right][int(image_id)],
                    args.score_threshold,
                    duct_only,
                )
                canvas = Image.new("RGB", (left_panel.width + right_panel.width, max(left_panel.height, right_panel.height)), "white")
                canvas.paste(left_panel, (0, 0))
                canvas.paste(right_panel, (left_panel.width, 0))
                filename = f"{comparison}_{event}_{rank}_image{image_id}.png"
                canvas.save(output_dir / filename)
                index.append({
                    "comparison": comparison,
                    "event": event,
                    "image_id": int(image_id),
                    "file_name": filename,
                    "source_image": str(image_path),
                })
    (output_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"visualizations": len(index), "out_dir": str(output_dir)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--img-folder", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    visualize(parser.parse_args())
