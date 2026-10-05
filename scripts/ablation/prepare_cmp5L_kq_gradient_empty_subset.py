"""Create a deterministic empty-image train subset before gradient diagnostics."""

import argparse
import json
from pathlib import Path


def main(args):
    source = json.loads(Path(args.ann_file).read_text(encoding="utf-8"))
    annotated = {int(item["image_id"]) for item in source["annotations"]}
    selected = [item for item in sorted(source["images"], key=lambda row: int(row["id"])) if int(item["id"]) not in annotated][:args.images]
    if len(selected) != args.images:
        raise RuntimeError(f"requested {args.images} empty images, found {len(selected)}")
    selected_ids = {int(item["id"]) for item in selected}
    subset = {
        **{key: value for key, value in source.items() if key not in ("images", "annotations")},
        "images": selected,
        "annotations": [item for item in source["annotations"] if int(item["image_id"]) in selected_ids],
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or Path(args.manifest).exists():
        raise RuntimeError("refusing to overwrite prelocked gradient subset")
    output.write_text(json.dumps(subset, ensure_ascii=False) + "\n", encoding="utf-8")
    Path(args.manifest).write_text(json.dumps({
        "rule": "first sorted train image IDs with zero annotations",
        "source": args.ann_file,
        "image_ids": sorted(selected_ids),
        "images": len(selected_ids),
        "annotations": len(subset["annotations"]),
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--images", type=int, default=32)
    main(parser.parse_args())
