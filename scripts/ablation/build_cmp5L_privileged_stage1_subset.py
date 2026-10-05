"""Build the locked, annotation-only cmp5L Stage-1 train subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.ablation.cmp5L_privileged_decision_kd import (
    build_stage1_subset_manifest,
    subset_coco_from_manifest,
)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--subset-ann", required=True)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument(
        "--counts-json",
        help="Optional exact stratum-count object; defaults to the locked 640-image design",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    subset_path = Path(args.subset_ann)
    for path in (manifest_path, subset_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite locked artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    source = json.loads(Path(args.ann_file).read_text(encoding="utf-8"))
    requested = json.loads(args.counts_json) if args.counts_json else None
    manifest = build_stage1_subset_manifest(source, seed=args.seed, requested=requested)
    manifest["source_annotation"] = str(Path(args.ann_file))
    manifest["source_annotation_sha256"] = sha256_path(args.ann_file)
    subset = subset_coco_from_manifest(source, manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    subset_path.write_text(
        json.dumps(subset, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "manifest": str(manifest_path),
        "subset_ann": str(subset_path),
        "images": len(subset["images"]),
        "annotations": len(subset["annotations"]),
        "manifest_sha256": sha256_path(manifest_path),
        "subset_sha256": sha256_path(subset_path),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
