"""P0-1 / P0-3: dump per-stage representations (backbone output + HybridEncoder output).

One forward pass per image yields three feature maps of shape (256, 80, 80) /
(256, 40, 40) / (256, 20, 20) at 640x640 -- verified identical for all five cmp5L
arms, because every arm's projector is ``proj_dim: 256`` and the shared neck is
``in_channels: [256, 256, 256]``. So cross-arm comparison needs no resampling and
the features are spatially aligned box-by-box.

Per image we keep two views, both at every level:

  glob_*_l{0,1,2}   adaptive-avg-pooled (--pool-grid x --pool-grid) map, flattened
                    -> answers "did the neck push heterogeneous backbones into a
                       common space?"  (global CKA, samples = images)
  roi_*_l{0,1,2}    roi_align 1x1 inside every ground-truth box
                    -> answers "on THIS lesion, how similar are student and
                       teacher?"  (lesion-conditioned CKA, samples = lesions)

Keeping ROI features per lesion is what makes the analysis disagreement-aware:
the caller can restrict the sample set to the easy / disagree / hard groups from
cmp5L_score_ranking_pergt_*.csv and ask whether the feature gap is concentrated
on exactly the lesions where the decisions differ.

Usage (1 GPU per arm, one task per arm)
---------------------------------------
    python scripts/ablation/diag_cmp5L_stage_features.py \
        --config ecdetseg/configs/ecdet/ecdet_l_dinov2s_breast_cmp5L_363_100e_es20.yml \
        --checkpoint outputs/ablation/ecdet_l_dinov2s_.../best.pth \
        --ann-file <valid.json> --img-folder /cobot/Data/Lesion_det/det_breast/img \
        --split-name valid --arm dinov2s \
        --out outputs/ablation/cmp5L_stage_features/dinov2s_valid.npz
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from pycocotools.coco import COCO

EC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EC_ROOT / "ecdetseg"))

from engine.core import YAMLConfig          # noqa: E402
from engine.solver import TASKS             # noqa: E402
from engine.misc import dist_utils          # noqa: E402

LEVELS = (0, 1, 2)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ann-file", required=True)
    ap.add_argument("--img-folder", required=True)
    ap.add_argument("--arm", required=True, help="arm tag, e.g. dinov2s")
    ap.add_argument("--split-name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights", default="ema")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--pool-grid", type=int, default=4,
                    help="global view: adaptive-avg-pool each level to g x g")
    ap.add_argument("--roi-size", type=int, default=1,
                    help="ROI view: roi_align output size (1 -> a 256-d vector/level)")
    ap.add_argument("--limit", type=int, default=0, help="debug: only N images")
    return ap.parse_args()


def load_ground_truth(ann_file):
    """image_id -> list of (ann_id, cat, x0, y0, x1, y1, area), original pixel coords."""
    coco = COCO(ann_file)
    gts = {}
    for ann in coco.dataset["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        x, y, w, h = ann["bbox"]
        if w <= 1 or h <= 1:
            continue
        gts.setdefault(ann["image_id"], []).append(
            (int(ann["id"]), int(ann["category_id"]), float(x), float(y),
             float(x + w), float(y + h), float(w * h)))
    return gts


def main():
    args = parse_args()
    t0 = time.time()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (EC_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("FATAL: --device cuda but torch.cuda.is_available() is False. "
              "Refusing to silently run the whole split on CPU.", file=sys.stderr)
        return 2

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
    for bn in ("ViTAdapter", "DinoV2Adapter"):
        if bn in cfg.yaml_cfg:
            cfg.yaml_cfg[bn]["skip_load_backbone"] = True

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver._setup()
    solver.load_resume_state(args.checkpoint)
    module = solver.ema.module if (args.weights == "ema" and solver.ema is not None) else solver.model
    module = dist_utils.de_parallel(module)
    module.eval()

    loader = cfg.val_dataloader
    gts = load_ground_truth(args.ann_file)
    print(f"[init] arm={args.arm} split={args.split_name} "
          f"images={len(loader.dataset)} pool_grid={args.pool_grid} "
          f"roi_size={args.roi_size}", flush=True)

    # ---- accumulators ------------------------------------------------------
    img_ids, img_names = [], []
    glob_b = {lv: [] for lv in LEVELS}
    glob_z = {lv: [] for lv in LEVELS}
    roi_b = {lv: [] for lv in LEVELS}
    roi_z = {lv: [] for lv in LEVELS}
    gt_ann_l, gt_cls_l, gt_box640_l, gt_area_l, gt_img_l = [], [], [], [], []
    gt_ptr = [0]

    seen = 0
    with torch.no_grad():
        for samples, targets in loader:
            if args.limit and seen >= args.limit:
                break
            x = samples.to(device, non_blocking=True)
            B = module.backbone(x)                     # 3 x [b,256,H,W]
            Z = module.encoder(B)                      # 3 x [b,256,H,W]
            B = list(B) if isinstance(B, (list, tuple)) else [B]
            Z = list(Z) if isinstance(Z, (list, tuple)) else [Z]

            pb = [F.adaptive_avg_pool2d(t.float(), args.pool_grid)
                  .flatten(1).cpu() for t in B]
            pz = [F.adaptive_avg_pool2d(t.float(), args.pool_grid)
                  .flatten(1).cpu() for t in Z]

            batch_gt = []
            for bi, tgt in enumerate(targets):
                iid = int(tgt["image_id"].item()) if torch.is_tensor(tgt["image_id"]) \
                    else int(tgt["image_id"])
                ow, oh = [int(v) for v in (tgt["orig_size"].tolist()
                                           if torch.is_tensor(tgt["orig_size"])
                                           else tgt["orig_size"])]
                rows = gts.get(iid, [])
                img_ids.append(iid)
                # filename from the dataset itself (kept for traceability)
                img_names.append(os.path.basename(
                    tgt.get("file_name", "") if isinstance(tgt, dict) else ""))
                # scale original-pixel boxes into the 640x640 network input
                sx, sy = 640.0 / ow, 640.0 / oh
                scaled = []
                for ann_id, cat, x0, y0, x1, y1, area in rows:
                    scaled.append((ann_id, cat, x0 * sx, y0 * sy,
                                   min(x1 * sx, 639.0), min(y1 * sy, 639.0), area))
                batch_gt.append(scaled)
                for ann_id, cat, x0, y0, x1, y1, area in scaled:
                    gt_ann_l.append(ann_id)
                    gt_cls_l.append(cat)
                    gt_box640_l.append((x0, y0, x1, y1))
                    gt_area_l.append(area)
                    gt_img_l.append(iid)
                gt_ptr.append(gt_ptr[-1] + len(scaled))

            for lv in LEVELS:
                glob_b[lv].append(pb[lv].to(torch.float16))
                glob_z[lv].append(pz[lv].to(torch.float16))

            # ROI pooling: one row per GT box, per level, on the same device
            for lv in LEVELS:
                boxes_lv, keep_rows = [], []
                for bi, scaled in enumerate(batch_gt):
                    for r, (ann_id, cat, x0, y0, x1, y1, area) in enumerate(scaled):
                        if x1 - x0 < 1.0 or y1 - y0 < 1.0:
                            continue
                        boxes_lv.append([float(bi), x0, y0, x1, y1])
                        keep_rows.append((bi, r))
                if boxes_lv:
                    bx = torch.tensor(boxes_lv, dtype=torch.float32, device=device)
                    stride = 640.0 / B[lv].shape[-1]
                    fb = torchvision.ops.roi_align(
                        B[lv].float(), bx, output_size=args.roi_size,
                        spatial_scale=1.0 / stride, sampling_ratio=2, aligned=True)
                    fz = torchvision.ops.roi_align(
                        Z[lv].float(), bx, output_size=args.roi_size,
                        spatial_scale=1.0 / stride, sampling_ratio=2, aligned=True)
                    roi_b[lv].append(fb.flatten(1).cpu().to(torch.float16))
                    roi_z[lv].append(fz.flatten(1).cpu().to(torch.float16))

            seen += len(targets)
            if seen % 200 < len(targets):
                print(f"  [{args.arm}] {seen} images  {time.time() - t0:.0f}s", flush=True)

    payload = {
        "image_ids": np.asarray(img_ids, dtype=np.int64),
        "file_names": np.asarray(img_names, dtype=object),
        "gt_ptr": np.asarray(gt_ptr, dtype=np.int64),
        "gt_ann_id": np.asarray(gt_ann_l, dtype=np.int64),
        "gt_cls": np.asarray(gt_cls_l, dtype=np.int64),
        "gt_box640": np.asarray(gt_box640_l, dtype=np.float32).reshape(-1, 4),
        "gt_area": np.asarray(gt_area_l, dtype=np.float32),
        "gt_image_id": np.asarray(gt_img_l, dtype=np.int64),
    }
    for lv in LEVELS:
        payload[f"glob_b_l{lv}"] = torch.cat(glob_b[lv], 0).numpy() if glob_b[lv] else np.zeros((0, 1), np.float16)
        payload[f"glob_z_l{lv}"] = torch.cat(glob_z[lv], 0).numpy() if glob_z[lv] else np.zeros((0, 1), np.float16)
        payload[f"roi_b_l{lv}"] = torch.cat(roi_b[lv], 0).numpy() if roi_b[lv] else np.zeros((0, 1), np.float16)
        payload[f"roi_z_l{lv}"] = torch.cat(roi_z[lv], 0).numpy() if roi_z[lv] else np.zeros((0, 1), np.float16)

    np.savez_compressed(out_path, **payload)

    meta = {
        "arm": args.arm, "split": args.split_name, "config": args.config,
        "checkpoint": args.checkpoint, "weights": args.weights,
        "ann_file": args.ann_file, "pool_grid": args.pool_grid, "roi_size": args.roi_size,
        "n_images": len(img_ids), "n_gt": len(gt_ann_l),
        "backbone_shapes": [list(t.shape) for t in B],
        "neck_shapes": [list(t.shape) for t in Z],
        "note": ("features captured from module.backbone(x) and module.encoder(...) at "
                 "640x640; ROI boxes are GT boxes scaled from original pixels to 640-space"),
        "seconds": round(time.time() - t0, 1),
    }
    with open(str(out_path).replace(".npz", ".meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    print(f"[done] {args.arm} {args.split_name}: {len(img_ids)} images, "
          f"{len(gt_ann_l)} GT, roi shapes "
          f"{[payload[f'roi_z_l{lv}'].shape for lv in LEVELS]}, "
          f"{time.time() - t0:.0f}s -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
