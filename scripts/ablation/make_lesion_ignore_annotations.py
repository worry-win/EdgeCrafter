#!/usr/bin/env python3
"""Create strict COCO splits with one category removed and labels remapped."""

import argparse
import json
from pathlib import Path


SPLITS = ("train", "valid", "test")


def convert_split(source: Path, target: Path, ignore_id: int) -> None:
    data = json.loads(source.read_text(encoding="utf-8"))
    categories = sorted(data["categories"], key=lambda category: int(category["id"]))
    source_ids = [int(category["id"]) for category in categories]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(f"{source}: duplicate category IDs")
    if ignore_id not in source_ids:
        raise ValueError(f"{source}: ignore category {ignore_id} is not declared")

    retained = [category for category in categories if int(category["id"]) != ignore_id]
    category_map = {int(category["id"]): new_id for new_id, category in enumerate(retained)}
    unknown_ids = {
        int(annotation["category_id"])
        for annotation in data["annotations"]
        if int(annotation["category_id"]) not in source_ids
    }
    if unknown_ids:
        raise ValueError(f"{source}: annotations use undeclared categories {sorted(unknown_ids)}")

    data["categories"] = [
        {**category, "id": category_map[int(category["id"])], "source_category_id": int(category["id"])}
        for category in retained
    ]
    data["annotations"] = [
        {**annotation, "category_id": category_map[int(annotation["category_id"])]}
        for annotation in data["annotations"]
        if int(annotation["category_id"]) != ignore_id
    ]
    target.write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        f"{target}: {len(data['images'])} images, {len(data['annotations'])} annotations, "
        f"{len(data['categories'])} classes"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--ignore-id", type=int, required=True)
    args = parser.parse_args()

    annotation_root = args.dataset_root / "annotations"
    for split in SPLITS:
        convert_split(
            annotation_root / f"{split}.json",
            annotation_root / f"{split}_ignore_{args.ignore_id}_remap.json",
            args.ignore_id,
        )


if __name__ == "__main__":
    main()
