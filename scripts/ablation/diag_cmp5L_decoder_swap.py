"""P0-2: the 5x5 decoder swap.  D_j( H_i( E_i(x) ) ).

Feasibility (verified, not assumed): for every arm the backbone and the neck both
emit three 256-channel maps at strides 8/16/32 -- (256,80,80) / (256,40,40) /
(256,20,20) at 640x640. Every arm's projector is ``proj_dim: 256`` and the shared
neck is ``in_channels: [256,256,256]``, and all five decoders come from the same
ECTransformer contract (4 layers / 300 queries / 3 levels / [3,6,3]). So a swapped
neck output is dimensionally and spatially a drop-in: no reshape, no resampling.

What the cell measures.  D_j(Z_i) mixes arm i's *features* with arm j's *decision
head*.  Read the 5x5 matrix as:

  diagonal        reference: D_i(Z_i), must reproduce that arm's own AP50
  off-diagonal    AP(D_j(Z_i)) - AP(D_i(Z_i))

  roughly flat  -> the neck already produces a decoder-agnostic common space
                   => neck-level multi-teacher KD is natural (Case A)
  moderate drop -> spaces are partly aligned; a teacher-specific adapter is needed
                   (Case B) -- and a per-channel mean/std re-standardisation of Z_i
                   to arm j's statistics should recover a good part of the drop.
                   If it does, the gap is a *distribution shift*, not a semantics
                   mismatch, which is exactly what a light adapter fixes.
  collapse      -> same architecture != same feature semantics; align features
                   before any parameter merging (Case C)

Output is written in the SAME npz schema as the main low-threshold dumps, so
``analyze_cmp5L_complementarity.py`` can consume a swap directory directly
(per-class AP50, hit matrices, perception ceilings all become available for hybrid
cells for free). To do that, point --out at a directory and read it with
``--arms {feat}__{dec} ...``.

Usage (one cell per invocation -> trivially an array job)
---------------------------------------------------------
    python scripts/ablation/diag_cmp5L_decoder_swap.py \
        --feat-arm dinov2s --feat-config ... --feat-checkpoint ... \
        --dec-arm  mae_vitb --dec-config ... --dec-checkpoint ... \
        --ann-file <test.json> --img-folder ... --split-name test \
        --out outputs/ablation/cmp5L_swap_predictions/dinov2s__mae_vitb_test.npz
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from pycocotools.coco import COCO

EC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EC_ROOT / "ecdetseg"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.core import YAMLConfig          # noqa: E402
from engine.solver import TASKS             # noqa: E402
from engine.misc import dist_utils          # noqa: E402


def xyxy_iou_matrix(a, b):
    """IoU between [Na,4] and [Nb,4] xyxy boxes (same implementation as the dumper)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return (inter / np.clip(aa[:, None] + bb[None, :] - inter, 1e-9, None)).astype(np.float32)


def load_ground_truth(ann_file):
    coco = COCO(ann_file)
    gts, names = {}, {}
    for img in coco.dataset["images"]:
        names[img["id"]] = img.get("file_name", "")
    for ann in coco.dataset["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        x, y, w, h = ann["bbox"]
        if w <= 1 or h <= 1:
            continue
        gts.setdefault(ann["image_id"], []).append(
            (int(ann["id"]), int(ann["category_id"]), float(x), float(y),
             float(x + w), float(y + h), float(w * h)))
    return gts, names


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-arm", required=True)
    ap.add_argument("--feat-config", required=True)
    ap.add_argument("--feat-checkpoint", required=True)
    ap.add_argument("--dec-arm", required=True)
    ap.add_argument("--dec-config", required=True)
    ap.add_argument("--dec-checkpoint", required=True)
    ap.add_argument("--ann-file", required=True)
    ap.add_argument("--img-folder", required=True)
    ap.add_argument("--split-name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights", default="ema")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--score-floor", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args()


def build_module(config, checkpoint, weights, device, num_workers, batch_size,
                 ann_file, img_folder):
    cfg = YAMLConfig(config, **{
        "val_dataloader": {
            "dataset": {"ann_file": ann_file, "img_folder": img_folder},
            "num_workers": num_workers,
            "total_batch_size": batch_size,
            "shuffle": False,
            "drop_last": False,
        },
        "num_classes": 4,
        "remap_mscoco_category": False,
    })
    for bn in ("ViTAdapter", "DinoV2Adapter"):
        if bn in cfg.yaml_cfg:
            cfg.yaml_cfg[bn]["skip_load_backbone"] = True
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver._setup()
    solver.load_resume_state(checkpoint)
    module = solver.ema.module if (weights == "ema" and solver.ema is not None) else solver.model
    module = dist_utils.de_parallel(module)
    module.eval()
    return solver, module


def main():
    args = parse_args()
    t0 = time.time()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (EC_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("FATAL: CUDA unavailable; refusing to run on CPU.", file=sys.stderr)
        return 2

    # the loader (and therefore the exact validation transforms / image order) is
    # taken from the FEATURE arm -- the swap changes the head, not the input pipeline
    solver_f, module_f = build_module(args.feat_config, args.feat_checkpoint,
                                      args.weights, device, args.num_workers,
                                      args.batch_size, args.ann_file, args.img_folder)
    _, module_d = build_module(args.dec_config, args.dec_checkpoint,
                               args.weights, device, args.num_workers,
                               args.batch_size, args.ann_file, args.img_folder)

    module_f.decoder = module_d.decoder          # the swap
    module_f.eval()
    print(f"[init] feat={args.feat_arm}  dec={args.dec_arm}  weights={args.weights}",
          flush=True)

    loader = solver_f.cfg.val_dataloader
    gts, names_by_id = load_ground_truth(args.ann_file)
    n_class = int(solver_f.cfg.yaml_cfg.get("num_classes", 4))

    rec_ptr = [0]
    rec_query, rec_score, rec_label, rec_logit, rec_box = [], [], [], [], []
    rec_gt_any_id, rec_gt_any_iou, rec_gt_any_cls = [], [], []
    rec_gt_cls_id, rec_gt_cls_iou = [], []
    img_ids, img_names, img_w, img_h, img_ngt = [], [], [], [], []
    gt_boxes, gt_cls, gt_ann, gt_area, gt_img = [], [], [], [], []

    seen = 0
    with torch.no_grad():
        for samples, targets in loader:
            if args.limit and seen >= args.limit:
                break
            x = samples.to(device, non_blocking=True)
            B = module_f.backbone(x)
            Z = module_f.encoder(B)
            outputs = module_f.decoder(Z, None)      # D_j(Z_i) -- the hybrid forward
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            logits = outputs["pred_logits"].float()
            boxes = outputs["pred_boxes"].float()
            if logits.dim() == 2:
                logits = logits.unsqueeze(0)
            if boxes.dim() == 2:
                boxes = boxes.unsqueeze(0)

            xyxy = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
            logit_np = logits.cpu().numpy()
            prob_np = 1.0 / (1.0 + np.exp(-logit_np))
            score_np = prob_np.max(axis=-1)
            label_np = prob_np.argmax(axis=-1).astype(np.int8)

            for bi, tgt in enumerate(targets):
                if args.limit and seen >= args.limit:
                    break
                image_id = int(tgt["image_id"].item()) if torch.is_tensor(tgt["image_id"]) \
                    else int(tgt["image_id"])
                ow, oh = [int(v) for v in (tgt["orig_size"].tolist()
                                           if torch.is_tensor(tgt["orig_size"])
                                           else tgt["orig_size"])]
                gt_rows = gts.get(image_id, [])
                img_ids.append(image_id)
                img_names.append(names_by_id.get(image_id, ""))
                img_w.append(ow)
                img_h.append(oh)
                img_ngt.append(len(gt_rows))
                for ann_id, cat, x0, y0, x1, y1, area in gt_rows:
                    gt_boxes.append((x0, y0, x1, y1))
                    gt_cls.append(cat)
                    gt_ann.append(ann_id)
                    gt_area.append(area)
                    gt_img.append(image_id)

                pb = xyxy[bi].copy()
                pb[:, 0::2] *= ow
                pb[:, 1::2] *= oh
                keep = np.where(score_np[bi] >= args.score_floor)[0]
                if len(keep) == 0:
                    rec_ptr.append(rec_ptr[-1])
                    seen += 1
                    continue

                gb = np.asarray([(r[2], r[3], r[4], r[5]) for r in gt_rows],
                                dtype=np.float32) if gt_rows else np.zeros((0, 4), np.float32)
                gcls = np.asarray([r[1] for r in gt_rows], dtype=np.int64)
                iou = xyxy_iou_matrix(pb[keep], gb)
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
                    any_iou = np.zeros(len(keep), np.float32)
                    gid_any = np.full(len(keep), -1, np.int64)
                    gc_any = np.full(len(keep), -1, np.int8)
                    cls_iou = np.zeros(len(keep), np.float32)
                    gid_cls = np.full(len(keep), -1, np.int64)

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
        "gt_ptr": np.concatenate([[0], np.cumsum([len(gts.get(i, [])) for i in img_ids])])
        .astype(np.int64) if img_ids else np.zeros(1, dtype=np.int64),
        "gt_box": np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4),
        "gt_cls": np.asarray(gt_cls, dtype=np.int8),
        "gt_ann_id": np.asarray(gt_ann, dtype=np.int64),
        "gt_area": np.asarray(gt_area, dtype=np.float32),
        "gt_image_id": np.asarray(gt_img, dtype=np.int64),
    }
    np.savez_compressed(out_path, **payload)

    meta = {
        "cell": f"{args.feat_arm}__{args.dec_arm}",
        "feature_arm": args.feat_arm, "decoder_arm": args.dec_arm,
        "split": args.split_name, "weights": args.weights,
        "feat_config": args.feat_config, "feat_checkpoint": args.feat_checkpoint,
        "dec_config": args.dec_config, "dec_checkpoint": args.dec_checkpoint,
        "ann_file": args.ann_file, "score_floor": args.score_floor,
        "n_images": len(img_ids), "n_records": int(payload["rec_score"].shape[0]),
        "note": ("hybrid forward: D_dec(Z_feat); Z captured from the feature arm's "
                 "HybridEncoder output; identical 256ch/stride8-16-32 contract"),
        "seconds": round(time.time() - t0, 1),
    }
    with open(str(out_path).replace(".npz", ".meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"[done] {args.feat_arm}__{args.dec_arm} {args.split_name}: "
          f"{len(img_ids)} images, {meta['n_records']} records, "
          f"{time.time() - t0:.0f}s -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
