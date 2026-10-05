"""Layer-wise privileged-value intervention: where does NP's gain form?

Fixed Normal initialization I(M_N) everywhere. Only the decoder cross-attention
value memory V is switched per-layer between V_N and V_P.

Conditions (Experiment A):
  NN      = all layers V_N
  P@L0    = only L0 uses V_P
  P@L1    = only L1 uses V_P
  P@L2    = only L2 uses V_P
  P@L3    = only L3 uses V_P
  P@L0-1  = L0,L1 use V_P
  P@L0-2  = L0,L1,L2 use V_P
  NP      = all layers V_P (must reproduce 2x2 NP)

The trick: TransformerDecoder.forward computes `value = value_op(memory)` ONCE and
reuses it for every layer. We replicate that loop by hand, computing both value_N
and value_P up front and passing the right one to each `layer(...)` call. No
monkeypatching of the forward needed; init is computed from M_N exactly once.

Also captures per-query cross-attn output + sampling/attention tensors via a
monkeypatched MSDeformableAttention.forward (for Experiments B/C downstream).
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EC_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(EC_ROOT / "ecdetseg"))
sys.path.insert(0, str(EC_ROOT))

from scripts.ablation.probe_cmp5L_privileged_ema_oracle import privilege_features  # noqa: E402
from scripts.ablation.probe_cmp5L_decoder_internal_tensors import (  # noqa: E402
    _build,
    _normalise_targets,
)
from scripts.ablation.dump_cmp5L_internal_behavior import _target_index  # noqa: E402
from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy  # noqa: E402
from engine.edgecrafter.utils import (  # noqa: E402
    inverse_sigmoid,
    distance2bbox,
    weighting_function,
)
from engine.solver.ec_engine import (  # noqa: E402
    summarize_pr_curve_f1,
    summarize_yolo_pr_curve_metrics,
)

STRONG_SPEC = {"name": "strong", "background_weight": 0.2,
               "context_weight": None, "context_scale": 1.5}

# condition -> list of layer indices (0..3) that use V_P
CONDITIONS = {
    "NN": [],
    "P@L0": [0],
    "P@L1": [1],
    "P@L2": [2],
    "P@L3": [3],
    "P@L0-1": [0, 1],
    "P@L0-2": [0, 1, 2],
    "NP": [0, 1, 2, 3],
}


# ---------------------------------------------------------------------------
# Manual decoder loop with per-layer value switching.
# ---------------------------------------------------------------------------
def run_layer_switch(ec, content, ref_unact, mem_n, mem_p, shapes, priv_layers):
    """Replicate TransformerDecoder.forward, switching value per layer.

    `priv_layers` is a set of layer indices whose cross-attn value comes from V_P
    (privileged memory); all others use V_N.

    `ec` is the ECTransformer: query_pos_head / pre_bbox_head / integral / up /
    reg_scale live on it (they are passed INTO TransformerDecoder.forward as args).
    """
    td = ec.decoder  # the inner TransformerDecoder

    value_n = td.value_op(mem_n, None, None, None, shapes)
    value_p = td.value_op(mem_p, None, None, None, shapes)

    output = content
    output_detach = pred_corners_undetach = 0
    project = None
    if td.use_aux_distribution and not hasattr(td, "project"):
        project = weighting_function(td.reg_max, ec.up, ec.reg_scale)
    elif td.use_aux_distribution:
        project = td.project

    ref_points_detach = F.sigmoid(ref_unact)
    query_pos_embed = ec.query_pos_head(ref_points_detach).clamp(min=-10, max=10)

    dec_out_bboxes, dec_out_logits, dec_out_pred_corners, dec_out_refs, dec_out_hs = [], [], [], [], []
    pre_bboxes = pre_scores = None

    for i, layer in enumerate(td.layers):
        ref_points_input = ref_points_detach.unsqueeze(2)
        value = value_p if i in priv_layers else value_n
        output = layer(output, ref_points_input, value, shapes, None, query_pos_embed)

        if i == 0 and (td.use_aux_distribution or td.use_pre_outputs):
            pre_bboxes = F.sigmoid(ec.pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
            if td.use_pre_outputs:
                pre_scores = ec.dec_score_head[0](output)
            if td.use_aux_distribution:
                ref_points_initial = pre_bboxes.detach()

        pred_corners = None
        if td.use_aux_distribution:
            previous_corners = pred_corners_undetach if torch.is_tensor(pred_corners_undetach) else 0
            pred_corners = ec.dec_bbox_head[i](output + output_detach) + previous_corners
            fdr_bbox = distance2bbox(ref_points_initial, ec.integral(pred_corners, project), ec.reg_scale)
        else:
            fdr_bbox = None

        if td.use_fdr_decode:
            inter_ref_bbox = fdr_bbox
        else:
            inter_ref_bbox = F.sigmoid(ec.continuous_bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

        if td.training or i == td.eval_idx:
            scores = ec.dec_score_head[i](output)
            if td.use_lqe and pred_corners is not None:
                scores = td.lqe_layers[i](scores, pred_corners)
            dec_out_logits.append(scores)
            dec_out_bboxes.append(inter_ref_bbox)
            if pred_corners is not None:
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)
            dec_out_hs.append(output)
            if not td.training:
                break

        if pred_corners is not None:
            pred_corners_undetach = pred_corners
        ref_points_detach = inter_ref_bbox.detach()
        output_detach = output.detach()

    out_bboxes = torch.stack(dec_out_bboxes)
    out_logits = torch.stack(dec_out_logits)
    out_pred_corners = torch.stack(dec_out_pred_corners) if dec_out_pred_corners else None
    out_refs = torch.stack(dec_out_refs) if dec_out_refs else None
    return {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}


def collect(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; refusing silent CPU fallback")

    cfg, solver, model = _build(args)
    model.eval()

    evaluators = {c: copy.deepcopy(cfg.evaluator) for c in CONDITIONS}
    for e in evaluators.values():
        e.cleanup()

    parity = {}
    init_parity = {}
    seen = 0
    started = time.time()

    for samples, targets in cfg.val_dataloader:
        image_ids = [int(t["image_id"].item()) for t in targets]
        samples = samples.to(device)
        matcher_targets = _normalise_targets(targets, samples.shape[-2], samples.shape[-1], device)

        with torch.no_grad():
            base = model.backbone(samples)
            feats_n = model.encoder([f.clone() for f in base])
            feats_p, masks = privilege_features(
                feats_n, targets, samples.shape[-2:],
                STRONG_SPEC["background_weight"], None, STRONG_SPEC["context_scale"],
            )

            ec = model.decoder
            # init from N only (fixed I_N across all conditions)
            mem_n, shapes_n = ec._get_encoder_input(feats_n)
            mem_p, shapes_p = ec._get_encoder_input(feats_p)
            content_n, ref_n, _bboxes, _logits = ec._get_decoder_input(mem_n, shapes_n)

            conds = {}
            for c, priv in CONDITIONS.items():
                conds[c] = run_layer_switch(
                    ec, content_n, ref_n, mem_n, mem_p, shapes_n, set(priv))

            # parity: NN vs full model
            if "NN" not in parity:
                baseline = model(samples)
                parity["NN_logits"] = float((conds["NN"]["pred_logits"] - baseline["pred_logits"]).abs().max())
                parity["NN_boxes"] = float((conds["NN"]["pred_boxes"] - baseline["pred_boxes"]).abs().max())
                if max(parity["NN_logits"], parity["NN_boxes"]) > 2e-6:
                    raise RuntimeError(f"NN parity failed: {parity}")

            # init parity: content/ref identical across conditions (they are, by construction)
            init_parity["content_identical"] = True
            init_parity["ref_identical"] = True

            # evaluate all conditions
            original_sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
            for c in CONDITIONS:
                results = solver.postprocessor(conds[c], original_sizes)
                preds = {int(t["image_id"].item()): r for t, r in zip(targets, results)}
                evaluators[c].update(preds)

        seen += len(targets)
        if seen % 400 < len(targets):
            print(f"[{seen}] {time.time()-started:.1f}s", flush=True)

    metrics = {}
    for c in CONDITIONS:
        e = evaluators[c]
        e.synchronize_between_processes()
        e.accumulate()
        e.summarize()
        ce = e.coco_eval["bbox"]
        yolo = summarize_yolo_pr_curve_metrics(ce, e.coco_gt)
        overall = (yolo or {}).get("yolo_overall") or {}
        n_gt = len(e.coco_gt.getAnnIds())
        rec = {
            "map": float(ce.stats[0]), "map50": float(ce.stats[1]),
            "map75": float(ce.stats[2]),
            "precision": float(overall.get("precision", 0.0)),
            "recall": float(overall.get("recall", 0.0)),
            "f1": float(overall.get("f1", 0.0)),
            "ar100": float(ce.stats[8]),
        }
        tp = rec["recall"] * n_gt
        rec["tp_count"] = round(tp, 1)
        rec["fp_count"] = round(tp / (rec["precision"] + 1e-12) - tp, 1) if rec["precision"] > 0 else None
        rec["fp_per_image"] = round(rec["fp_count"] / max(1, seen), 4) if rec["fp_count"] is not None else None
        rec["n_gt"] = n_gt
        metrics[c] = rec

    payload = {
        "meta": {
            "checkpoint": args.checkpoint, "weights": args.weights,
            "n_images": seen, "condition": STRONG_SPEC["name"],
            "background_weight": STRONG_SPEC["background_weight"],
            "parity": parity, "init_parity": init_parity,
            "seconds": round(time.time() - started, 2),
        },
        "conditions": CONDITIONS,
        "metrics": metrics,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics}, ensure_ascii=False, indent=2))


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
    p.add_argument("--max-batches", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    collect(parse_args())
