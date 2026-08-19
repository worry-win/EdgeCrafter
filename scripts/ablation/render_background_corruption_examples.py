import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import tv_tensors

from ecdetseg.engine.data.transforms import GTExcludedBackgroundCorruption


ANN_FILE = Path("/cobot/Data/Lesion_det/det_liver/annotations/Lesion/Ignore_Delete-Image/train.json")
IMAGE_ROOT = Path("/cobot/Data/Lesion_det/det_liver/img")
OUTPUT_FILE = Path(sys.argv[1])
TILE_SIZE = 300
HEADER_HEIGHT = 38
ROW_GAP = 8


def load_font(size):
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


FONT = load_font(14)
SMALL_FONT = load_font(12)
COLORS = [
    (255, 91, 91),
    (255, 191, 71),
    (112, 214, 148),
    (80, 190, 255),
    (176, 132, 255),
    (255, 121, 196),
    (174, 206, 82),
]


with ANN_FILE.open(encoding="utf-8") as handle:
    coco = json.load(handle)

categories = {int(item["id"]): item["name"] for item in coco["categories"]}
annotations_by_image = defaultdict(list)
for annotation in coco["annotations"]:
    annotations_by_image[int(annotation["image_id"])].append(annotation)

# Greedily cover categories, then fill to ten with compact, annotated examples.
candidate_images = [
    image for image in coco["images"]
    if annotations_by_image[int(image["id"])] and len(annotations_by_image[int(image["id"])]) <= 5
]
selected = []
covered = set()
for image in candidate_images:
    labels = {int(item["category_id"]) for item in annotations_by_image[int(image["id"])]}
    if labels - covered:
        selected.append(image)
        covered.update(labels)
    if len(selected) == 10:
        break
for image in candidate_images:
    if len(selected) == 10:
        break
    if image not in selected:
        selected.append(image)


def resize_with_boxes(image_info, annotations):
    image = Image.open(IMAGE_ROOT / image_info["file_name"]).convert("RGB")
    original_width, original_height = image.size
    image = image.resize((640, 640), Image.Resampling.BILINEAR)
    scale_x, scale_y = 640 / original_width, 640 / original_height
    boxes = []
    labels = []
    for annotation in annotations:
        x, y, width, height = annotation["bbox"]
        boxes.append([
            x * scale_x,
            y * scale_y,
            (x + width) * scale_x,
            (y + height) * scale_y,
        ])
        labels.append(int(annotation["category_id"]))
    return image, boxes, labels


def augment(image, boxes, mode, seed):
    tensor = tv_tensors.Image(torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float() / 255.0)
    target = {
        "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(640, 640)),
        "labels": torch.zeros(len(boxes), dtype=torch.long),
    }
    torch.manual_seed(seed)
    transform = GTExcludedBackgroundCorruption(
        mode=mode,
        p=1.0,
        area_range=(0.05, 0.15),
        aspect_ratio_range=(0.5, 2.0),
        box_margin=4,
        max_trials=50,
        noise_std=0.15,
    )
    output, _ = transform((tensor, target))
    changed = (output.as_subclass(torch.Tensor) != tensor.as_subclass(torch.Tensor)).any(dim=0)
    coordinates = changed.nonzero()
    rectangle = None
    if len(coordinates):
        top_left = coordinates.min(dim=0).values
        bottom_right = coordinates.max(dim=0).values + 1
        rectangle = (int(top_left[1]), int(top_left[0]), int(bottom_right[1]), int(bottom_right[0]))
    array = (output.as_subclass(torch.Tensor).permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
    return Image.fromarray(array), rectangle


def annotated_tile(image, boxes, labels, corruption_rectangle=None, footer=None):
    tile = image.resize((TILE_SIZE, TILE_SIZE), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(tile)
    scale = TILE_SIZE / 640
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = [value * scale for value in box]
        color = COLORS[label % len(COLORS)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text = f"{label}: {categories[label]}"
        text_box = draw.textbbox((x1, y1), text, font=SMALL_FONT)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_y = max(0, y1 - text_height - 4)
        draw.rectangle((x1, text_y, min(TILE_SIZE, x1 + text_width + 4), text_y + text_height + 4), fill=color)
        draw.text((x1 + 2, text_y + 1), text, fill=(20, 20, 20), font=SMALL_FONT)
    if corruption_rectangle is not None:
        rectangle = tuple(value * scale for value in corruption_rectangle)
        draw.rectangle(rectangle, outline=(0, 255, 255), width=4)
    if footer:
        draw.rectangle((0, TILE_SIZE - 22, TILE_SIZE, TILE_SIZE), fill=(0, 0, 0, 180))
        draw.text((5, TILE_SIZE - 20), footer, fill=(255, 255, 255), font=SMALL_FONT)
    return tile


canvas_width = TILE_SIZE * 3
canvas_height = HEADER_HEIGHT + len(selected) * (TILE_SIZE + ROW_GAP)
canvas = Image.new("RGB", (canvas_width, canvas_height), (24, 27, 34))
header = ImageDraw.Draw(canvas)
for column, title in enumerate(("Original + GT", "Gaussian noise + GT", "Mean mask + GT")):
    header.text((column * TILE_SIZE + 8, 9), title, fill=(245, 245, 245), font=FONT)

for row, image_info in enumerate(selected):
    annotations = annotations_by_image[int(image_info["id"])]
    original, boxes, labels = resize_with_boxes(image_info, annotations)
    noise, noise_rectangle = augment(original, boxes, "noise", seed=10_000 + row)
    mask, mask_rectangle = augment(original, boxes, "mask", seed=10_000 + row)
    footer = f"#{row + 1}  {Path(image_info['file_name']).name[:28]}"
    tiles = [
        annotated_tile(original, boxes, labels, footer=footer),
        annotated_tile(noise, boxes, labels, noise_rectangle, footer=footer),
        annotated_tile(mask, boxes, labels, mask_rectangle, footer=footer),
    ]
    top = HEADER_HEIGHT + row * (TILE_SIZE + ROW_GAP)
    for column, tile in enumerate(tiles):
        canvas.paste(tile, (column * TILE_SIZE, top))

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
canvas.save(OUTPUT_FILE, format="JPEG", quality=78, optimize=True, progressive=True)
print({
    "output": str(OUTPUT_FILE),
    "size": canvas.size,
    "bytes": OUTPUT_FILE.stat().st_size,
    "selected": len(selected),
    "covered_categories": sorted(covered),
    "categories": categories,
})
