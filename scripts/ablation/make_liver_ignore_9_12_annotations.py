#!/usr/bin/env python3
"""Create COCO annotations restricted to the nine RF-DETR liver classes."""

import json
from pathlib import Path


ROOT = Path("/cobot/Data/Lesion_det/det_liver/annotations")
KEEP_MAX_CATEGORY_ID = 8

for split in ("train", "valid", "test"):
    source = ROOT / f"{split}.json"
    target = ROOT / f"{split}_ignore_9_12.json"
    data = json.loads(source.read_text())
    data["categories"] = [cat for cat in data["categories"] if cat["id"] <= KEEP_MAX_CATEGORY_ID]
    data["annotations"] = [ann for ann in data["annotations"] if ann["category_id"] <= KEEP_MAX_CATEGORY_ID]
    target.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    print(f"{target}: {len(data['images'])} images, {len(data['annotations'])} annotations, {len(data['categories'])} categories")
