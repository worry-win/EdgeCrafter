"""Per-class + scale metrics for NN vs NP (Experiment 13 supplement).

Runs NN and NP (fixed Normal init), outputs per-class AP50/AP/recall and
small/medium/large AP, to test whether privileged value genuinely helps any
hard class or just suppresses FP globally.
"""

from __future__ import annotations

import argparse
import copy
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
from engine.solver.ec_engine import summarize_yolo_pr_curve_metrics  # noqa: E402

STRONG_SPEC = {"name": "strong", "background_weight": 0.2,
               "context_weight": None, "context_scale": 1.5}
CLASS_NAMES = ["实(囊实)性", "囊性", "淋巴结", "导管相关"]


def collect(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    cfg, solver, model = _build(args)
    model.eval()

    evaluators = {c: copy.deepcopy(cfg.evaluator) for c in ("NN", "NP")}
    for e in evaluators.values():
        e.cleanup()

    seen = 0
    started = time.time()

    for samples, targets in cfg.val_dataloader:
        samples = samples.to(device)
        _normalise_targets(targets, samples.shape[-2], samples.shape[-1], device)
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

            conds = {
                "NN": run_layer_switch(ec, content_n, ref_n, mem_n, mem_p, shapes_n, set()),
                "NP": run_layer_switch(ec, content_n, ref_n, mem_n, mem_p, shapes_n, {0, 1, 2, 3}),
            }
            original_sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
            for c in ("NN", "NP"):
                results = solver.postprocessor(conds[c], original_sizes)
                preds = {int(t["image_id"].item()): r for t, r in zip(targets, results)}
                evaluators[c].update(preds)

        seen += len(targets)
        if seen % 400 < len(targets):
            print(f"[{seen}] {time.time()-started:.1f}s", flush=True)

    out = {}
    for c in ("NN", "NP"):
        e = evaluators[c]
        e.synchronize_between_processes()
        e.accumulate()
        e.summarize()
        ce = e.coco_eval["bbox"]
        # per-class AP50 / AP (class dim index 0..C-1)
        precision = ce.eval["precision"]  # [T, R, K, A, M]
        iou50 = int(np.argmin(np.abs(ce.params.iouThrs - 0.50)))
        cat_ids = list(ce.params.catIds)
        per_class = {}
        for ki, cat_id in enumerate(cat_ids):
            p_ap = precision[:, :, ki, 0, -1]  # [T, R] over IoU thresholds
            ap = float(np.mean(p_ap[p_ap > -1]))
            ap50 = float(np.mean(precision[iou50, :, ki, 0, -1][precision[iou50, :, ki, 0, -1] > -1]))
            per_class[str(cat_id)] = {"name": CLASS_NAMES[ki] if ki < len(CLASS_NAMES) else str(cat_id),
                                      "ap50": ap50, "ap": ap}
        out[c] = {
            "stats": ce.stats.tolist(),  # [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ...]
            "per_class": per_class,
        }

    payload = {
        "meta": {"n_images": seen, "seconds": round(time.time() - started, 2),
                 "background_weight": STRONG_SPEC["background_weight"]},
        "NN": out["NN"],
        "NP": out["NP"],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2)[:4000])


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
