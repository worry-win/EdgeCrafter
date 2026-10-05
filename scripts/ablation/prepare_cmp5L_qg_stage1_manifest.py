"""Create the pre-registered 12-image QG Stage-1 train manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def build(coco):
    images = {int(item["id"]): item for item in coco["images"]}
    annotations = {image_id: [] for image_id in images}
    for item in coco["annotations"]:
        annotations.setdefault(int(item["image_id"]), []).append(item)
    classes = sorted({int(item["category_id"]) for item in coco["annotations"]})
    if len(classes) < 4:
        raise ValueError(f"expected four classes, found {classes}")
    selected = []
    strata = []

    def add(stratum, candidates, count):
        for image_id in sorted(candidates):
            if image_id in selected:
                continue
            selected.append(image_id)
            strata.append(stratum)
            if sum(name == stratum for name in strata) == count:
                return
        raise ValueError(f"could not select {count} unique images for {stratum}")

    add("empty", [image_id for image_id in images if not annotations[image_id]], 3)
    for category_id in classes[:4]:
        add(f"class_{category_id}", [
            image_id for image_id in images
            if any(int(item["category_id"]) == category_id for item in annotations[image_id])
        ], 2)
    hard = [
        image_id for image_id in images
        if len(annotations[image_id]) > 1
        or any(float(item["bbox"][2] * item["bbox"][3]) <= 0.01 * images[image_id]["width"] * images[image_id]["height"] for item in annotations[image_id])
    ]
    add("hard", hard, 1)
    if len(selected) != 12:
        raise AssertionError(f"manifest has {len(selected)} images")
    rows = [
        {
            "image_id": image_id,
            "file_name": images[image_id]["file_name"],
            "stratum": strata[index],
            "gt_count": len(annotations[image_id]),
            "category_ids": sorted({int(item["category_id"]) for item in annotations[image_id]}),
        }
        for index, image_id in enumerate(selected)
    ]
    manifest = {
        "schema_version": 1,
        "selection_rule": "sorted image ids: 3 empty, 2 per first four category ids, 1 hard; unique first assignment",
        "source_ann_file": "train.json",
        "image_count": 12,
        "images": rows,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    subset = dict(coco)
    subset["images"] = [images[image_id] for image_id in selected]
    subset["annotations"] = [item for image_id in selected for item in annotations[image_id]]
    return manifest, subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--subset-ann-file", required=True)
    args = parser.parse_args()
    coco = json.loads(Path(args.ann_file).read_text())
    manifest, subset = build(coco)
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n")
    Path(args.subset_ann_file).write_text(json.dumps(subset))
    print(json.dumps({"image_ids": [row["image_id"] for row in manifest["images"]], "sha256": manifest["sha256"]}))


if __name__ == "__main__":
    main()
