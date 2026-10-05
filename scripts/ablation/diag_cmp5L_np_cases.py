"""Rescued-FP / broken-TP case collection (Experiment D).

Runs NN and NP, saving per-query final score / class / GT-IoU / FG-Ring-BG
attention mass (last layer) for case-level analysis. Then locally filters:
  - broken-TP: NN TP (score>=0.5, correct class, IoU>=0.5) -> NP non-TP
  - rescued-FP: NN FP (score>=0.5, wrong/no GT) -> NP suppressed (score<0.5)
Compares their FG/Ring/BG composition to answer WHY the same background
suppression helps FP but hurts some real lesions.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

EC_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(EC_ROOT / "ecdetseg"))
sys.path.insert(0, str(EC_ROOT))

from scripts.ablation.probe_cmp5L_privileged_ema_oracle import privilege_features  # noqa: E402
from scripts.ablation.probe_cmp5L_decoder_internal_tensors import (  # noqa: E402
    _build,
    _normalise_targets,
)
from scripts.ablation.diag_cmp5L_np_layervalue import run_layer_switch  # noqa: E402
from scripts.ablation.diag_cmp5L_np_whowhat import (  # noqa: E402
    AttentionCapture,
    _region_mask,
    _iou_matrix,
)
from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy  # noqa: E402

STRONG_SPEC = {"name": "strong", "background_weight": 0.2,
               "context_weight": None, "context_scale": 1.5}
RING_SCALE = 1.5


def collect(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    cfg, solver, model = _build(args)
    model.eval()

    cap = AttentionCapture(model.decoder.decoder)
    records = []  # per-query records (all queries with score>=0.5 in either branch)

    seen = 0
    started = time.time()

    with cap:
        for samples, targets in cfg.val_dataloader:
            image_ids = [int(t["image_id"].item()) for t in targets]
            samples = samples.to(device)
            matcher_targets = _normalise_targets(targets, samples.shape[-2], samples.shape[-1], device)

            with torch.no_grad():
                base = model.backbone(samples)
                feats_n = model.encoder([f.clone() for f in base])
                feats_p, _ = privilege_features(
                    feats_n, targets, samples.shape[-2:],
                    STRONG_SPEC["background_weight"], None, STRONG_SPEC["context_scale"])

                ec = model.decoder
                mem_n, shapes_n = ec._get_encoder_input(feats_n)
                mem_p, shapes_p = ec._get_encoder_input(feats_p)
                content_n, ref_n, _b, _l = ec._get_decoder_input(mem_n, shapes_n)

                cap.layers = [dict() for _ in ec.decoder.layers]
                out_nn = run_layer_switch(ec, content_n, ref_n, mem_n, mem_p, shapes_n, set())
                nn_layers = [dict(r) for r in cap.layers]
                cap.layers = [dict() for _ in ec.decoder.layers]
                out_np = run_layer_switch(ec, content_n, ref_n, mem_n, mem_p, shapes_n, {0, 1, 2, 3})
                np_layers = [dict(r) for r in cap.layers]

                for bi, target in enumerate(targets):
                    gt_cx = matcher_targets[bi]["boxes"]
                    gt_labels = matcher_targets[bi]["labels"]
                    gt_xyxy = box_cxcywh_to_xyxy(gt_cx)

                    boxes_n = box_cxcywh_to_xyxy(out_nn["pred_boxes"][bi])
                    boxes_p = box_cxcywh_to_xyxy(out_np["pred_boxes"][bi])
                    logits_n = out_nn["pred_logits"][bi]
                    logits_p = out_np["pred_logits"][bi]
                    score_n = logits_n.sigmoid().max(-1).values
                    score_p = logits_p.sigmoid().max(-1).values
                    cls_n = logits_n.argmax(-1)
                    cls_p = logits_p.argmax(-1)

                    # GT match (best IoU)
                    iou_n = torch.zeros(boxes_n.shape[0], device=device)
                    iou_p = torch.zeros_like(iou_n)
                    if gt_xyxy.shape[0] > 0:
                        iou_n = _iou_matrix(boxes_n, gt_xyxy).max(-1).values
                        iou_p = _iou_matrix(boxes_p, gt_xyxy).max(-1).values

                    # FG/Ring/BG attention mass at last layer (L3)
                    def region_mass(layers):
                        rec = layers[3]
                        if "sampling_locations" not in rec:
                            return None
                        loc = rec["sampling_locations"][bi]
                        attn = rec["attention_weights"][bi]
                        region = _region_mask(loc, gt_xyxy)
                        out = {}
                        for r, rname in ((2, "FG"), (1, "Ring"), (0, "BG")):
                            out[rname] = (attn * (region == r).float()).sum(-1).mean(-1).cpu().numpy()
                        return out

                    mass_n = region_mass(nn_layers)
                    mass_p = region_mass(np_layers)

                    for qi in range(boxes_n.shape[0]):
                        if score_n[qi] < 0.5 and score_p[qi] < 0.5:
                            continue
                        rec = {
                            "image_id": image_ids[bi], "query": qi,
                            "score_n": float(score_n[qi]), "score_p": float(score_p[qi]),
                            "cls_n": int(cls_n[qi]), "cls_p": int(cls_p[qi]),
                            "iou_n": float(iou_n[qi]), "iou_p": float(iou_p[qi]),
                            "gt_class": int(gt_labels[0]) if gt_xyxy.shape[0] > 0 else -1,
                        }
                        if mass_n is not None:
                            rec["mass_n_FG"] = float(mass_n["FG"][qi])
                            rec["mass_n_Ring"] = float(mass_n["Ring"][qi])
                            rec["mass_n_BG"] = float(mass_n["BG"][qi])
                        if mass_p is not None:
                            rec["mass_p_FG"] = float(mass_p["FG"][qi])
                            rec["mass_p_Ring"] = float(mass_p["Ring"][qi])
                            rec["mass_p_BG"] = float(mass_p["BG"][qi])
                        records.append(rec)

            seen += len(targets)
            if seen % 400 < len(targets):
                print(f"[{seen}] {time.time()-started:.1f}s  records={len(records)}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"meta": {"n_images": seen}, "records": records},
                                         ensure_ascii=False), encoding="utf-8")
    print(f"[done] {len(records)} records")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ann-file", required=True)
    p.add_argument("--img-folder", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--weights", choices=("ema", "model"), default="ema")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    collect(parse_args())
