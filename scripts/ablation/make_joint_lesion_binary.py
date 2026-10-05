"""Map the 17 joint-dataset lesion classes to one detection foreground class."""

import json
import os
import sys
from pathlib import Path


def main(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must differ")
    destination.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        source_file = source / f"instances_{split}.json"
        with source_file.open() as handle:
            data = json.load(handle)
        category_ids = {category["id"] for category in data["categories"]}
        if category_ids != set(range(17)):
            raise ValueError(f"{source_file}: expected category IDs 0..16")
        for annotation in data["annotations"]:
            if annotation["category_id"] not in category_ids:
                raise ValueError(f"{source_file}: unknown annotation category")
            annotation["category_id"] = 0
        data["categories"] = [{"id": 0, "name": "lesion"}]
        target_file = destination / source_file.name
        temporary = target_file.with_suffix(".json.tmp")
        with temporary.open("w") as handle:
            json.dump(data, handle, ensure_ascii=False)
        os.replace(temporary, target_file)
        print(f"{split}: {len(data['images'])} images, {len(data['annotations'])} lesion boxes -> {target_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(f"Usage: {sys.argv[0]} SOURCE_ANNOTATIONS DESTINATION_ANNOTATIONS")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
