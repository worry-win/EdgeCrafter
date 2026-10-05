"""Create a new QG-owned 640-image enriched train diagnostic subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ablation.cmp5L_privileged_decision_kd import (  # noqa: E402
    build_stage1_subset_manifest,
    subset_coco_from_manifest,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--subset-ann-file", required=True)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    coco = json.loads(Path(args.ann_file).read_text())
    manifest = build_stage1_subset_manifest(coco, seed=args.seed)
    subset = subset_coco_from_manifest(coco, manifest)
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n")
    Path(args.subset_ann_file).write_text(json.dumps(subset))
    print(json.dumps({
        "images": len(manifest["images"]),
        "manifest_sha256": manifest["manifest_sha256"],
        "strata": manifest["requested"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
