"""Dump low-threshold, per-query detection records for one cmp5L arm.

Why this exists
---------------
The usual COCO evaluation only keeps the top-``num_top_queries`` (query, class)
pairs above a high confidence, and it discards the raw logits. That makes it
impossible to answer the question this project actually cares about:

    did model M fail to *perceive* lesion L, or did it perceive it but with low
    confidence?

So this script keeps **every one of the 300 decoder queries** for every image
(no score threshold at all by default -- ``--score-floor 0.0``), together with
the full raw class logit vector, the absolute-pixel box, the query index and the
matching against the ground-truth boxes. Downstream analysis can then impose any
(IoU, score) operating point after the fact.

Everything is written as one compressed ``.npz`` plus a ``meta.json`` per
(arm, split). No pandas/pyarrow is required -- the cluster env only has numpy.

Records are laid out image-major and concatenated; ``rec_ptr`` gives the record
slice of image i, ``gt_ptr`` the ground-truth slice. Both are int64 arrays of
length n_images + 1, so the classic "ragged array" indexing applies.

Usage
-----
    python scripts/ablation/dump_cmp5L_predictions.py \
        --config ../ecdetseg/configs/ecdet/ecdet_l_dinov2s_breast_cmp5L_363_100e_es20.yml \
        --checkpoint outputs/ablation/<run>/best.pth \
        --ann-file /cobot/Data/.../valid.json \
        --img-folder /cobot/Data/Lesion_det/det_breast/img \
        --split-name valid --model-name dinov2s \
        --output outputs/ablation/cmp5L_predictions/dinov2s_valid.npz
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision

# ecdetseg/ is NOT a python package (no __init__.py) and the repo's own entry
# points import via `from engine.x import y` with ecdetseg/ on sys.path. Follow
# that convention rather than importing `ecdetseg.engine.*`, which would create
# a second, duplicate copy of every module.
EC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EC_ROOT / "ecdetseg"))

from engine.core import YAMLConfig  # noqa: E402
from engine.misc import dist_utils  # noqa: E402
from engine.solver import TASKS  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ann-file", required=True)
    p.add_argument("--img-folder", required=True)
    p.add_argument("--split-name", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--score-floor", type=float, default=0.0,
                   help="keep records with max-class sigmoid >= this. 0.0 keeps all 300 queries.")
    p.add_argument("--limit", type=int, default=0, help="only the first N images (debug)")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--amp", dest="amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--weights", choices=["ema", "model"], default="ema",
                   help="the reported metrics came from the EMA weights, so that is the default")
    return p.parse_args()


def xyxy_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between [Na,4] and [Nb,4] xyxy boxes."""
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return np.zeros((na, nb), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / np.maximum(union, 1e-9)).astype(np.float32)


def load_ground_truth(ann_file: str):
    """Return (per_image_gt, file_name_by_id, categories)."""
    with open(ann_file, encoding="utf-8") as fh:
        data = json.load(fh)

    cats = {c["id"]: c.get("name", str(c["id"])) for c in data.get("categories", [])}
    per_img = {}
    for ann in data["annotations"]:
        # Mirror ConvertCocoPolysToMask: drop crowd annotations, clamp to image,
        # and discard degenerate boxes.
        if ann.get("iscrowd", 0) != 0:
            continue
        x, y, w, h = ann["bbox"]
        per_img.setdefault(ann["image_id"], []).append(
            (int(ann["id"]), int(ann["category_id"]), float(x), float(y), float(w), float(h))
        )

    gts = {}
    names = {}
    for img in data["images"]:
        iid = img["id"]
        names[iid] = str(img.get("file_name", ""))
        iw, ih = float(img["width"]), float(img["height"])
        rows = []
        for ann_id, cat, x, y, w, h in per_img.get(iid, []):
            x0, y0 = max(x, 0.0), max(y, 0.0)
            x1, y1 = min(x + w, iw), min(y + h, ih)
            if x1 <= x0 or y1 <= y0:
                continue
            rows.append((ann_id, cat, x0, y0, x1, y1, w * h))
        gts[iid] = rows

    return gts, names, cats


def main():
    args = parse_args()
    t0 = time.time()

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (EC_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("FATAL: --device cuda but torch.cuda.is_available() is False "
              "(broken node). Refusing to silently run on CPU.", file=sys.stderr)
        return 2

    # ---- config ------------------------------------------------------------
    # Deep-merge overrides; merge_dict recurses into dicts and replaces lists, so
    # the validation transforms are preserved verbatim.
    cfg = YAMLConfig(args.config, **{
        "val_dataloader": {
            "dataset": {"ann_file": args.ann_file, "img_folder": args.img_folder},
            "num_workers": args.num_workers,
            "total_batch_size": args.batch_size,
            "shuffle": False,
            "drop_last": False,
        },
        "num_classes": 4,
        "remap_mscoco_category": False,
    })

    # A checkpoint contains the complete detector; do not re-fetch an external
    # backbone before applying it (same rule as train.py under -r/-t).
    for backbone_name in ("ViTAdapter", "DinoV2Adapter"):
        if backbone_name in cfg.yaml_cfg:
            cfg.yaml_cfg[backbone_name]["skip_load_backbone"] = True

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver._setup()
    print(f"[init] loading checkpoint {args.checkpoint}", flush=True)
    solver.load_resume_state(args.checkpoint)

    module = solver.ema.module if (args.weights == "ema" and solver.ema is not None) else solver.model
    module = dist_utils.de_parallel(module)
    module.eval()
    print(f"[init] using weights={args.weights}", flush=True)

    loader = cfg.val_dataloader
    n_img_total = len(loader.dataset)
    print(f"[data] {args.split_name}: {n_img_total} images, batch={args.batch_size}, "
          f"workers={args.num_workers}", flush=True)

    gts, file_names_by_id, cats = load_ground_truth(args.ann_file)
    n_class = int(cfg.yaml_cfg.get("num_classes", 4))

    # ---- accumulators ------------------------------------------------------
    rec_image, rec_query, rec_score, rec_label = [], [], [], []
    rec_logit, rec_box = [], []
    rec_gt_any_id, rec_gt_any_iou, rec_gt_any_cls = [], [], []
    rec_gt_cls_id, rec_gt_cls_iou = [], []
    rec_ptr = [0]

    img_ids, img_names, img_w, img_h, img_ngt = [], [], [], [], []
    gt_boxes, gt_cls, gt_ann, gt_area, gt_img = [], [], [], [], []

    seen = 0
    with torch.no_grad():
        for samples, targets in loader:
            if args.limit and seen >= args.limit:
                break
            if not isinstance(targets, (list, tuple)):
                raise TypeError(
                    "expected the val collate_fn to return a list of target dicts, "
                    f"got {type(targets).__name__}"
                )
            samples = samples.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=(args.amp and device.type == "cuda")):
                outputs = module(samples)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            logits = outputs["pred_logits"].float()          # [B,Q,C]
            boxes = outputs["pred_boxes"].float()            # [B,Q,4] cxcywh, normalized
            if logits.dim() == 2:
                logits = logits.unsqueeze(0)
            if boxes.dim() == 2:
                boxes = boxes.unsqueeze(0)

            b, q, c = logits.shape
            xyxy = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
            logit_np = logits.cpu().numpy()
            prob_np = 1.0 / (1.0 + np.exp(-logit_np))
            score_np = prob_np.max(axis=-1)
            label_np = prob_np.argmax(axis=-1).astype(np.int8)

            for bi, tgt in enumerate(targets):
                if args.limit and seen >= args.limit:
                    break
                image_id = int(tgt["image_id"].item()) if torch.is_tensor(tgt["image_id"]) else int(tgt["image_id"])
                ow, oh = [int(v) for v in (tgt["orig_size"].tolist() if torch.is_tensor(tgt["orig_size"])
                                           else tgt["orig_size"])]

                gt_rows = gts.get(image_id, [])
                img_ids.append(image_id)
                img_names.append(file_names_by_id.get(image_id, ""))
                img_w.append(ow)
                img_h.append(oh)
                img_ngt.append(len(gt_rows))

                for ann_id, cat, x0, y0, x1, y1, area in gt_rows:
                    gt_boxes.append((x0, y0, x1, y1))
                    gt_cls.append(cat)
                    gt_ann.append(ann_id)
                    gt_area.append(area)
                    gt_img.append(image_id)

                # absolute pixels in the ORIGINAL image frame
                pb = xyxy[bi].copy()
                pb[:, 0::2] *= ow
                pb[:, 1::2] *= oh

                keep = np.where(score_np[bi] >= args.score_floor)[0]
                if len(keep) == 0:
                    rec_ptr.append(rec_ptr[-1])
                    seen += 1
                    continue

                gb = np.asarray([(r[2], r[3], r[4], r[5]) for r in gt_rows], dtype=np.float32) \
                    if gt_rows else np.zeros((0, 4), dtype=np.float32)
                gcls = np.asarray([r[1] for r in gt_rows], dtype=np.int64)
                iou = xyxy_iou_matrix(pb[keep], gb)          # [K, G]

                if iou.shape[1] > 0:
                    any_j = iou.argmax(axis=1)
                    any_iou = iou[np.arange(len(keep)), any_j]
                    gid_any = np.asarray([r[0] for r in gt_rows], dtype=np.int64)[any_j]
                    gc_any = gcls[any_j].astype(np.int8)
                    same = iou * (gcls[None, :] == label_np[bi][keep][:, None])
                    cls_j = same.argmax(axis=1)
                    cls_iou = same[np.arange(len(keep)), cls_j]
                    gid_cls = np.asarray([r[0] for r in gt_rows], dtype=np.int64)[cls_j]
                    cls_iou = np.where(cls_iou > 0, cls_iou, 0.0)
                    gid_cls = np.where(cls_iou > 0, gid_cls, -1)
                else:
                    any_iou = np.zeros(len(keep), dtype=np.float32)
                    gid_any = np.full(len(keep), -1, dtype=np.int64)
                    gc_any = np.full(len(keep), -1, dtype=np.int8)
                    cls_iou = np.zeros(len(keep), dtype=np.float32)
                    gid_cls = np.full(len(keep), -1, dtype=np.int64)

                rec_image.append(np.full(len(keep), image_id, dtype=np.int64))
                rec_query.append(keep.astype(np.int16))
                rec_score.append(score_np[bi][keep].astype(np.float32))
                rec_label.append(label_np[bi][keep])
                rec_logit.append(logit_np[bi][keep].astype(np.float32))
                rec_box.append(pb[keep].astype(np.float32))
                rec_gt_any_id.append(gid_any.astype(np.int32))
                rec_gt_any_iou.append(any_iou.astype(np.float32))
                rec_gt_any_cls.append(gc_any)
                rec_gt_cls_id.append(gid_cls.astype(np.int32))
                rec_gt_cls_iou.append(cls_iou.astype(np.float32))
                rec_ptr.append(rec_ptr[-1] + len(keep))
                seen += 1

            if args.log_every and seen % args.log_every < args.batch_size:
                print(f"  [{seen}/{n_img_total if not args.limit else args.limit}] "
                      f"{time.time() - t0:.1f}s", flush=True)

    def cat_or_empty(lst, dtype, shape):
        return np.concatenate(lst) if lst else np.zeros((0,) + shape, dtype=dtype)

    payload = {
        "rec_ptr": np.asarray(rec_ptr, dtype=np.int64),
        "image_ids": np.asarray(img_ids, dtype=np.int64),
        "orig_w": np.asarray(img_w, dtype=np.int32),
        "orig_h": np.asarray(img_h, dtype=np.int32),
        "n_gt": np.asarray(img_ngt, dtype=np.int32),
        "file_names": np.asarray(img_names, dtype=object),
        "rec_query_id": cat_or_empty(rec_query, np.int16, ()),
        "rec_score": cat_or_empty(rec_score, np.float32, ()),
        "rec_label": cat_or_empty(rec_label, np.int8, ()),
        "rec_logit": cat_or_empty(rec_logit, np.float32, (n_class,)),
        "rec_box": cat_or_empty(rec_box, np.float32, (4,)),
        "rec_gt_any_id": cat_or_empty(rec_gt_any_id, np.int32, ()),
        "rec_gt_any_iou": cat_or_empty(rec_gt_any_iou, np.float32, ()),
        "rec_gt_any_cls": cat_or_empty(rec_gt_any_cls, np.int8, ()),
        "rec_gt_cls_id": cat_or_empty(rec_gt_cls_id, np.int32, ()),
        "rec_gt_cls_iou": cat_or_empty(rec_gt_cls_iou, np.float32, ()),
        "gt_ptr": np.concatenate([[0], np.cumsum([len(gts.get(i, [])) for i in img_ids])]).astype(np.int64)
        if img_ids else np.zeros(1, dtype=np.int64),
        "gt_box": np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4),
        "gt_cls": np.asarray(gt_cls, dtype=np.int8),
        "gt_ann_id": np.asarray(gt_ann, dtype=np.int64),
        "gt_area": np.asarray(gt_area, dtype=np.float32),
        "gt_image_id": np.asarray(gt_img, dtype=np.int64),
    }
    np.savez_compressed(out_path, **payload)

    n_rec = int(payload["rec_score"].shape[0])
    meta = {
        "model_name": args.model_name,
        "split": args.split_name,
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "weights": args.weights,
        "ann_file": args.ann_file,
        "img_folder": args.img_folder,
        "score_floor": args.score_floor,
        "num_classes": n_class,
        "class_names": {int(k): v for k, v in cats.items()},
        "n_images": len(img_ids),
        "n_records": n_rec,
        "n_gt": int(payload["gt_box"].shape[0]),
        "queries_per_image": None if n_rec == 0 else int(q),
        "batch_size": args.batch_size,
        "amp": bool(args.amp and device.type == "cuda"),
        "limit": args.limit,
        "elapsed_sec": round(time.time() - t0, 2),
        "note": ("rec_* arrays are image-major and ragged; slice image i with "
                 "rec_ptr[i]:rec_ptr[i+1]. rec_logit is the RAW pre-sigmoid logit "
                 "vector, rec_score is max(sigmoid(logit)), rec_label its argmax. "
                 "rec_box is absolute xyxy in original-image pixels. "
                 "rec_gt_any_* is the best-IoU GT regardless of class; "
                 "rec_gt_cls_* restricts to GTs whose class equals rec_label."),
    }
    with open(out_path.with_suffix(".meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print(f"[done] {args.model_name}/{args.split_name}: {len(img_ids)} images, "
          f"{n_rec} records, {payload['gt_box'].shape[0]} GT, "
          f"{out_path.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
