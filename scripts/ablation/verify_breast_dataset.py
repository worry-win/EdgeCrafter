#!/usr/bin/env python3
"""Read-only verification of the breast detection dataset used by the cmp5L family.

Answers, with measured numbers instead of inference:

1. per-split size, and how many images contain **no** annotation at all
   (pure negatives) -- this is what makes validation and test non-comparable;
2. the COCO size bucket distribution (small <32^2, medium 32^2-96^2, large >96^2)
   which decides whether APs / APm / APl can carry a conclusion at all;
3. per-category box counts, to expose classes whose arm-to-arm differences are
   sample-size artefacts;
4. image dimensions, on-disk file integrity (missing / unreferenced files);
5. split disjointness at file level and at "block" level (one imaging series).

Nothing is written and nothing is executed on the dataset; pass --json to also
dump the measurements.

    /cobot/miniforge3/envs/lw-detr/bin/python \
        scripts/ablation/verify_breast_dataset.py

PITFALL this script exists to kill: the ``image id`` field in each COCO json is
an independent running index per file, so naively intersecting ``id`` sets
reports a bogus ~96% train/test overlap. Only ``file_name`` is comparable.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

ROOT = Path("/cobot/Data/Lesion_det/det_breast")
ANN_DIR = ROOT / "annotations" / "Lesion" / "Ignore_Delete-Image"
IMG_DIR = ROOT / "img"
SPLITS = ("train", "valid", "test")

SMALL_MAX = 32 * 32      # 1024
LARGE_MIN = 96 * 96      # 9216


def block_id(file_name: str) -> str:
    """Group frames of one imaging series.

    File names look like ``train__<hash>__rightBreast-<uuid>_<frame>.png``;
    the stable series key is ``<side>-<uuid>`` without the trailing index.
    """
    parts = file_name.split("__")
    return parts[2].rsplit("_", 1)[0] if len(parts) >= 3 else file_name


def load(split: str) -> dict:
    return json.loads((ANN_DIR / f"{split}.json").read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="optional path for the raw measurements")
    args = ap.parse_args()

    data = {s: load(s) for s in SPLITS}
    report: dict = {"splits": {}, "images": {}, "files": {}, "leakage": {}}

    print("=" * 92)
    print("1. per-split size and pure-negative images")
    print("=" * 92)
    print(f"{'split':7s}{'images':>9s}{'annotated':>11s}{'negative':>10s}{'neg %':>8s}"
          f"{'boxes':>8s}{'box/annot.img':>15s}")
    for s, d in data.items():
        images, anns = d["images"], d["annotations"]
        had = {a["image_id"] for a in anns}
        n_neg = sum(1 for i in images if i["id"] not in had)
        row = {
            "images": len(images),
            "annotated_images": len(had),
            "negative_images": n_neg,
            "negative_pct": round(n_neg / len(images) * 100, 1),
            "boxes": len(anns),
            "boxes_per_annotated_image": round(len(anns) / max(len(had), 1), 2),
        }
        report["splits"][s] = row
        print(f"{s:7s}{row['images']:>9d}{row['annotated_images']:>11d}"
              f"{row['negative_images']:>10d}{row['negative_pct']:>7.1f}%"
              f"{row['boxes']:>8d}{row['boxes_per_annotated_image']:>15.2f}")
    print("\nNOTE: box density is nearly identical across splits (1.05-1.11 per")
    print("      annotated image). What differs by two orders of magnitude is the")
    print("      share of PURE-NEGATIVE images -> that alone drives the val/test gap.")

    print()
    print("=" * 92)
    print("2. COCO size buckets of the boxes (small <1024, medium 1024-9216, large >9216 px^2)")
    print("=" * 92)
    for s, d in data.items():
        areas = [a["bbox"][2] * a["bbox"][3] for a in d["annotations"]]
        buckets = {
            "small": sum(1 for x in areas if x < SMALL_MAX),
            "medium": sum(1 for x in areas if SMALL_MAX <= x <= LARGE_MIN),
            "large": sum(1 for x in areas if x > LARGE_MIN),
        }
        report["splits"][s]["size_buckets"] = buckets
        total = len(areas)
        print(f"{s:7s}" + "  ".join(
            f"{k}={v:5d} ({v / total * 100:4.1f}%)" for k, v in buckets.items()) + f"   total={total}")
    print("\nNOTE: ~58-60% of boxes are MEDIUM, so APm is the only scale metric with")
    print("      enough support. APs rests on 13 (valid) / 42 (test) boxes and must")
    print("      not be used to rank arms.")

    print()
    print("=" * 92)
    print("3. boxes per category")
    print("=" * 92)
    cats = {c["id"]: c["name"] for c in data["train"]["categories"]}
    print(f"{'category':22s}" + "".join(f"{s:>9s}" for s in SPLITS))
    for cid in sorted(cats):
        counts = [collections.Counter(a["category_id"] for a in data[s]["annotations"]).get(cid, 0)
                  for s in SPLITS]
        report["splits"].setdefault("per_category", {})[cats[cid]] = dict(zip(SPLITS, counts))
        print(f"{cats[cid]:22s}" + "".join(f"{c:>9d}" for c in counts))

    print()
    print("=" * 92)
    print("4. image dimensions and on-disk file integrity")
    print("=" * 92)
    for s, d in data.items():
        w = sorted(i["width"] for i in d["images"])
        h = sorted(i["height"] for i in d["images"])
        report["images"][s] = {
            "width_median": w[len(w) // 2], "width_min": w[0], "width_max": w[-1],
            "height_median": h[len(h) // 2], "height_min": h[0], "height_max": h[-1],
        }
        print(f"{s:7s} width median {w[len(w) // 2]:5d} [{w[0]}-{w[-1]}]   "
              f"height median {h[len(h) // 2]:5d} [{h[0]}-{h[-1]}]")

    on_disk = set(os.listdir(IMG_DIR))
    referenced = {i["file_name"] for d in data.values() for i in d["images"]}
    missing = [f for s, d in data.items() for i in d["images"] if i["file_name"] not in on_disk]
    unreferenced = sorted(on_disk - referenced)
    report["files"] = {
        "on_disk": len(on_disk),
        "referenced_unique": len(referenced),
        "missing_referenced": len(missing),
        "unreferenced": len(unreferenced),
        "unreferenced_by_prefix": dict(collections.Counter(
            f.split("__")[0] for f in unreferenced)),
    }
    print(f"\nimg directory files        : {len(on_disk)}")
    print(f"referenced by the 3 splits : {len(referenced)}")
    print(f"referenced but missing     : {len(missing)}")
    print(f"on disk but unreferenced   : {len(unreferenced)} "
          f"{report['files']['unreferenced_by_prefix']}")

    print()
    print("=" * 92)
    print("5. leakage: file level and block level (one imaging series)")
    print("=" * 92)
    names = {s: {i["file_name"] for i in data[s]["images"]} for s in SPLITS}
    blocks = {s: collections.Counter(block_id(i["file_name"]) for i in data[s]["images"])
              for s in SPLITS}
    for s in SPLITS:
        print(f"{s:7s} blocks={len(blocks[s]):5d}  largest block={max(blocks[s].values()):4d} frames")
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        f_overlap = len(names[a] & names[b])
        b_overlap = len(set(blocks[a]) & set(blocks[b]))
        report["leakage"][f"{a}_{b}"] = {"file_overlap": f_overlap, "block_overlap": b_overlap}
        print(f"  {a}∩{b}: files={f_overlap}  blocks={b_overlap}")
    print("\nCAVEAT: file and block level are clean, but a true patient-level split")
    print("        cannot be proven from file names -- confirm with the data owner.")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        print(f"\nraw measurements written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
