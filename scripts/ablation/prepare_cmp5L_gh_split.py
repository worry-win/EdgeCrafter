"""Create the pre-result calibration/evaluation image split for cmp5L H."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scripts.ablation.evaluate_cmp5L_gh_validation import stratified_image_split


def main(args):
    annotation_path = Path(args.ann_file)
    output_path = Path(args.out)
    if output_path.exists():
        raise FileExistsError(output_path)
    dataset = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotations = defaultdict(list)
    for annotation in dataset["annotations"]:
        annotations[int(annotation["image_id"])].append(annotation)
    image_by_id = {int(image["id"]): image for image in dataset["images"]}
    image_ids = sorted(image_by_id)
    positive_ids = {image_id for image_id in image_ids if annotations[image_id]}
    duct_ids = {
        image_id
        for image_id in positive_ids
        if any(int(annotation["category_id"]) == 3 for annotation in annotations[image_id])
    }
    split = stratified_image_split(image_ids, positive_ids, duct_ids, seed=args.seed)

    def record(image_id):
        if image_id not in positive_ids:
            stratum = "empty"
        elif image_id in duct_ids:
            stratum = "positive_with_duct"
        else:
            stratum = "positive_without_duct"
        return {
            "image_id": int(image_id),
            "file_name": image_by_id[image_id]["file_name"],
            "stratum": stratum,
        }

    payload = {
        "scope": "cmp5L H development-data calibration/evaluation control split",
        "seed": int(args.seed),
        "unit": "image",
        "patient_grouping": False,
        "patient_grouping_reason": (
            "annotation has no patient field; filename UUID heuristic covers only 1658/2975 images"
        ),
        "warning": (
            "predictor, historical threshold, and research decisions already used this validation; "
            "this split is not an independent generalization estimate"
        ),
        "strata": ["empty", "positive_with_duct", "positive_without_duct"],
        "calibration": [record(image_id) for image_id in split["calibration"]],
        "evaluation": [record(image_id) for image_id in split["evaluation"]],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    counts = {
        subset: {
            stratum: sum(item["stratum"] == stratum for item in payload[subset])
            for stratum in payload["strata"]
        }
        for subset in ("calibration", "evaluation")
    }
    print(json.dumps({"path": str(output_path), "counts": counts}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260919)
    main(parser.parse_args())

