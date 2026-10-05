"""Query-adaptive evidence suppression Oracle.

Tests whether "harmful background evidence" can be selectively suppressed
(query-specific sampled-value suppression) rather than uniformly suppressed
(fixed bg=0.2). Pure zero-training Oracle mechanism validation.

Suppression happens AFTER bilinear sampling and BEFORE attention aggregation,
inside a re-implemented deformable-attention core. Sampling locations and
attention weights are left UNCHANGED (sanity-checked); only V(p) is scaled by a
per-query-per-point coefficient.

Strategies (all fixed Normal init I_N, only L0-L2 value modified, L3 = V_N):
  NN              : baseline
  NP-fixed0.2     : FG=1, BG=0.2 (uniform, all queries)  [reproduces NP]
  TP-protect      : lambda_TP=1.0, lambda_FP/BG=0.2      [Oracle A1]
  FP-sweep        : lambda_TP=1, lambda_FP/BG in {0,0.1,0.2,0.4,0.6,0.8}  [A2]
  dep-gate        : lambda=0.2 if d_q^BG>tau else 1.0     [B threshold]
  dep-continuous  : lambda=clip(1-alpha*d_q, 0.2, 1)      [B continuous]
  ring-preserve   : FG=1, Ring=1(or 0.5), FarBG=0.2(or 0) [C]
  adaptive-farBG  : low-dep query FG=1,Ring=1,BG=1; high-dep FG=1,Ring=1,BG=0.2 [D]
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
from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy  # noqa: E402
from engine.edgecrafter.utils import (  # noqa: E402
    inverse_sigmoid,
    distance2bbox,
    weighting_function,
)
from engine.solver.ec_engine import summarize_yolo_pr_curve_metrics  # noqa: E402

STRONG_BG = 0.2
RING_SCALE = 1.5


# ---------------------------------------------------------------------------
# Suppression-aware deformable attention core.
# ---------------------------------------------------------------------------
def suppression_core(value, value_spatial_shapes, sampling_locations,
                     attention_weights, num_points_list, suppress_coeff):
    """Re-implement deformable_attention_core_func_v2 with per-point suppression.

    suppress_coeff: [bs, Len_q, n_head, sum_points] per-query-per-point multiplier
    applied to V(p) AFTER bilinear sampling and BEFORE attention aggregation.
    """
    bs, n_head, c, _ = value[0].shape
    _, Len_q, _, _, _ = sampling_locations.shape

    sampling_grids = 2 * sampling_locations - 1
    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value[level].reshape(bs * n_head, c, h, w)
        sampling_grid_l = sampling_locations_list[level]
        sampling_value_l = F.grid_sample(
            value_l, sampling_grid_l, mode='bilinear',
            padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l)

    # concat sampled values: [bs*n_head, c, Len_q, sum_points]
    sampled = torch.concat(sampling_value_list, dim=-1)
    # apply suppression: suppress_coeff [bs, Len_q, n_head, sum_points]
    # -> [bs*n_head, 1, Len_q, sum_points]
    coeff = suppress_coeff.permute(0, 2, 1, 3).reshape(bs * n_head, 1, Len_q, -1)
    sampled = sampled * coeff

    attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(bs * n_head, 1, Len_q, -1)
    weighted = sampled * attn_weights
    output = weighted.sum(-1).reshape(bs, n_head * c, Len_q)
    return output.permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Region mask (FG / Ring / FarBG) from sampling locations + GT boxes.
# ---------------------------------------------------------------------------
def region_mask(loc, gt_xyxy, scale=RING_SCALE):
    """loc: [Q,H,P,2] normalized xy; gt_xyxy: [N,4] normalized xyxy.
    Returns [Q,H,P] int: 2=FG, 1=Ring, 0=FarBG (priority FG>Ring>BG)."""
    if gt_xyxy.shape[0] == 0:
        return torch.zeros(loc.shape[:-1], dtype=torch.long, device=loc.device)
    q, h, p = loc.shape[:-1]
    flat = loc.reshape(-1, 2)
    x, y = flat[:, 0], flat[:, 1]
    inside = torch.zeros(flat.shape[0], dtype=torch.bool, device=loc.device)
    in_ring = torch.zeros_like(inside)
    for g in gt_xyxy:
        x0, y0, x1, y1 = g
        inside |= (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hw, hh = (x1 - x0) * scale / 2, (y1 - y0) * scale / 2
        in_ring |= (x >= cx - hw) & (x <= cx + hw) & (y >= cy - hh) & (y <= cy + hh)
    region = torch.where(inside, torch.full_like(inside, 2, dtype=torch.long),
                         torch.where(in_ring & ~inside,
                                     torch.ones_like(inside, dtype=torch.long),
                                     torch.zeros_like(inside, dtype=torch.long)))
    return region.reshape(q, h, p)


def _iou_matrix(a, b):
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(-1)
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp(min=1e-9)


def query_tp_fp(boxes_xyxy, logits, gt_xyxy, gt_labels, iou_thr=0.5):
    """Return per-query is_tp [Q] bool (TP=correct class + IoU>=thr + score>=0.5)."""
    scores = logits.sigmoid().max(-1).values
    if gt_xyxy.shape[0] == 0:
        return torch.zeros(boxes_xyxy.shape[0], dtype=torch.bool, device=boxes_xyxy.device)
    iou = _iou_matrix(boxes_xyxy, gt_xyxy)
    max_iou, max_gt = iou.max(-1)
    gt_class = gt_labels[max_gt.clamp(min=0)]
    pred_class = logits.argmax(-1)
    return (max_iou >= iou_thr) & (gt_class == pred_class) & (scores >= 0.5)


# ---------------------------------------------------------------------------
# Per-query suppression coefficient builder (Oracle strategies).
# ---------------------------------------------------------------------------
def build_suppression(strategy, qlabel, bg_dep, region_by_layer, arg):
    """Build per-query-per-point coefficient for ONE layer given region [Q,H,P].

    qlabel: [Q] bool is_tp (NN anchor). bg_dep: [Q] BG attention mass (NN).
    region_by_layer: [Q,H,P] int (2=FG,1=Ring,0=FarBG).
    Returns coeff [Q,H,P] float in [0,1].
    """
    q, h, p = region_by_layer.shape
    coeff = torch.ones(q, h, p, device=region_by_layer.device, dtype=torch.float32)

    if strategy == "NN":
        return coeff
    if strategy == "NP-fixed":
        coeff[region_by_layer != 2] = STRONG_BG  # non-FG -> bg weight
        return coeff
    if strategy == "TP-protect":
        lam = torch.where(qlabel, torch.ones(q, device=region_by_layer.device),
                          torch.full((q,), STRONG_BG, device=region_by_layer.device))
        coeff[region_by_layer != 2] = lam[:, None, None].expand(-1, h, p)[region_by_layer != 2]
        return coeff
    if strategy == "FP-sweep":
        lam_fp = arg
        lam = torch.where(qlabel, torch.ones(q, device=region_by_layer.device),
                          torch.full((q,), lam_fp, device=region_by_layer.device))
        coeff[region_by_layer != 2] = lam[:, None, None].expand(-1, h, p)[region_by_layer != 2]
        return coeff
    if strategy == "dep-gate":
        tau = arg
        lam = torch.where(bg_dep > tau,
                          torch.full((q,), STRONG_BG, device=region_by_layer.device),
                          torch.ones(q, device=region_by_layer.device))
        coeff[region_by_layer != 2] = lam[:, None, None].expand(-1, h, p)[region_by_layer != 2]
        return coeff
    if strategy == "dep-continuous":
        alpha = arg
        lam = torch.clamp(1 - alpha * bg_dep, min=STRONG_BG, max=1.0)
        coeff[region_by_layer != 2] = lam[:, None, None].expand(-1, h, p)[region_by_layer != 2]
        return coeff
    if strategy == "ring-preserve":
        ring_w, farbg_w = arg  # e.g. (1.0, 0.2) or (0.5, 0.2) or (1.0, 0.0)
        coeff[region_by_layer == 1] = ring_w
        coeff[region_by_layer == 0] = farbg_w
        return coeff
    if strategy == "adaptive-farBG":
        tau = arg
        # low-dep query: no suppression (all 1.0); high-dep query: farBG=0.2, Ring=1
        high_dep = bg_dep > tau
        for qi in range(q):
            if high_dep[qi]:
                coeff[qi, region_by_layer[qi] == 0] = STRONG_BG
        return coeff
    raise ValueError(f"unknown strategy {strategy}")


# ---------------------------------------------------------------------------
# Run one branch with suppression applied to L0-L2 cross-attn value.
# ---------------------------------------------------------------------------
def run_suppressed(ec, content, ref_unact, mem, shapes, priv_layers, strategy, arg,
                   gt_xyxy, qlabel, bg_dep, cap):
    """Manual decoder loop; for layers in priv_layers, wrap the cross-attn core
    to apply query-specific suppression. `cap` captures per-layer tensors."""
    td = ec.decoder
    value = td.value_op(mem, None, None, None, shapes)

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
    saved_cores = []

    for i, layer in enumerate(td.layers):
        ref_points_input = ref_points_detach.unsqueeze(2)

        if i in priv_layers:
            # wrap the cross-attn core with suppression
            orig_core = layer.cross_attn.ms_deformable_attn_core

            def make_wrapped(core, layer_idx):
                def wrapped(value_, shapes_, loc_, attn_, npts_):
                    # capture locations/attn (for sanity + dependency)
                    if cap is not None:
                        cap["layers"][layer_idx]["sampling_locations"] = loc_.detach().clone()
                        cap["layers"][layer_idx]["attention_weights"] = attn_.detach().clone()
                    # build region + suppression coeff
                    bs, nq, nh, np_, _ = loc_.shape
                    # loc_ is [B,Q,H,P,2]; region per batch
                    coeff = torch.ones(bs, nq, nh, np_, device=loc_.device)
                    for bi in range(bs):
                        # negative image (no GT): no suppression (fairness vs
                        # privileged-memory which fills mask with 1.0 -> V_P=V_N)
                        if gt_xyxy[bi].shape[0] == 0:
                            continue
                        reg = region_mask(loc_[bi], gt_xyxy[bi])  # [Q,H,P]
                        ql = None if qlabel is None else qlabel[bi]
                        bd = None if bg_dep is None else bg_dep[bi]
                        c = build_suppression(strategy, ql, bd, reg, arg)
                        coeff[bi] = c
                    return suppression_core(value_, shapes_, loc_, attn_, npts_, coeff)
                return wrapped

            layer.cross_attn.ms_deformable_attn_core = make_wrapped(orig_core, i)
            saved_cores.append((layer.cross_attn, orig_core))

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

    # restore original cores
    for mod, core in saved_cores:
        mod.ms_deformable_attn_core = core

    out_bboxes = torch.stack(dec_out_bboxes)
    out_logits = torch.stack(dec_out_logits)
    return {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}


# ---------------------------------------------------------------------------
# Main collection: per-image, run NN probe then suppressed branch(es).
# ---------------------------------------------------------------------------
def collect(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    cfg, solver, model = _build(args)
    model.eval()

    strategies = json.loads(args.strategies)  # list of {name, strategy, arg}
    evaluators = {s["name"]: copy.deepcopy(cfg.evaluator) for s in strategies}
    for e in evaluators.values():
        e.cleanup()

    # sanity accumulators
    sanity = {"nn_parity_done": False, "np_parity": None, "lambda1_parity": None,
              "loc_unchanged": [], "attn_unchanged": []}

    seen = 0
    started = time.time()

    for samples, targets in cfg.val_dataloader:
        image_ids = [int(t["image_id"].item()) for t in targets]
        samples = samples.to(device)
        matcher_targets = _normalise_targets(targets, samples.shape[-2], samples.shape[-1], device)

        with torch.no_grad():
            base = model.backbone(samples)
            feats_n = model.encoder([f.clone() for f in base])
            feats_p, _ = privilege_features(
                feats_n, targets, samples.shape[-2:], STRONG_BG, None, RING_SCALE)

            ec = model.decoder
            mem_n, shapes_n = ec._get_encoder_input(feats_n)
            mem_p, shapes_p = ec._get_encoder_input(feats_p)
            content_n, ref_n, _b, _l = ec._get_decoder_input(mem_n, shapes_n)

            # per-image GT (normalized xyxy) for each batch element
            gt_xyxy_list = []
            for t in matcher_targets:
                gt_xyxy_list.append(box_cxcywh_to_xyxy(t["boxes"]))
            gt_labels_list = [t["labels"] for t in matcher_targets]

            # --- NN probe (capture-only, no suppression) to get query type + BG dependency ---
            # We wrap layers {0,1,2} with strategy="NN" (all-ones coeff) so the core is
            # wrapped AND captures sampling_locations/attention_weights, but applies no
            # suppression (coeff stays 1). This is required for bg_dep to be non-zero.
            cap_nn = {"layers": [dict() for _ in ec.decoder.layers]}
            out_nn = run_suppressed(ec, content_n, ref_n, mem_n, shapes_n, {0, 1, 2}, "NN", None,
                                    gt_xyxy_list, None, None, cap_nn)
            nn_layers = [dict(r) for r in cap_nn["layers"]]

            # per-query TP label + BG dependency (from NN)
            boxes_n = box_cxcywh_to_xyxy(out_nn["pred_boxes"])  # [B,Q,4]
            logits_n = out_nn["pred_logits"]
            qlabel_list = []
            bg_dep_list = []
            for bi in range(samples.shape[0]):
                ql = query_tp_fp(boxes_n[bi], logits_n[bi], gt_xyxy_list[bi], gt_labels_list[bi])
                qlabel_list.append(ql)
                # BG dependency: mean over layers L0-L2 of BG attention mass
                deps = []
                for li in range(3):
                    rec = nn_layers[li]
                    if "sampling_locations" in rec:
                        loc = rec["sampling_locations"][bi]  # [Q,H,P,2]
                        attn = rec["attention_weights"][bi]  # [Q,H,P]
                        reg = region_mask(loc, gt_xyxy_list[bi])  # [Q,H,P]
                        bg_mass = (attn * (reg == 0).float()).sum(-1).mean(-1)  # [Q]
                        deps.append(bg_mass)
                if deps:
                    bg_dep_list.append(torch.stack(deps).mean(0))  # [Q]
                else:
                    bg_dep_list.append(torch.zeros(logits_n.shape[1], device=device))

            # --- parity checks (first batch only) ---
            if not sanity["nn_parity_done"]:
                baseline = model(samples)
                d_logits = float((out_nn["pred_logits"] - baseline["pred_logits"]).abs().max())
                d_boxes = float((out_nn["pred_boxes"] - baseline["pred_boxes"]).abs().max())
                sanity["nn_parity"] = {"logits": d_logits, "boxes": d_boxes}
                if max(d_logits, d_boxes) > 2e-6:
                    raise RuntimeError(f"NN parity failed: {sanity['nn_parity']}")
                sanity["nn_parity_done"] = True

            # --- run each suppressed strategy (all use mem_n + per-point suppression) ---
            results_by_strategy = {}
            for s in strategies:
                name = s["name"]
                strat = s["strategy"]
                arg = s.get("arg")
                # NP-fixed also goes through the suppression path (mem_n + all-query
                # non-FG lambda=0.2), which is mathematically equivalent to the
                # privileged-memory NP up to float reorder ~1e-7. This validates the
                # suppression path itself.
                out = run_suppressed(ec, content_n, ref_n, mem_n, shapes_n, {0, 1, 2}, strat, arg,
                                     gt_xyxy_list, qlabel_list, bg_dep_list, None)
                results_by_strategy[name] = out

            # evaluate
            original_sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
            for name, out in results_by_strategy.items():
                results = solver.postprocessor(out, original_sizes)
                preds = {int(t["image_id"].item()): r for t, r in zip(targets, results)}
                evaluators[name].update(preds)

        seen += len(targets)
        if seen % 400 < len(targets):
            print(f"[{seen}] {time.time()-started:.1f}s", flush=True)

    # summarize
    metrics = {}
    for name, e in evaluators.items():
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
        # per-class AP50 (class dim K), for duct-related hard-class analysis
        precision = ce.eval["precision"]  # [T, R, K, A, M]
        iou50 = int(np.argmin(np.abs(ce.params.iouThrs - 0.50)))
        cat_ids = list(ce.params.catIds)
        rec["per_class_ap50"] = {}
        for ki, cat_id in enumerate(cat_ids):
            p50 = precision[iou50, :, ki, 0, -1]
            rec["per_class_ap50"][str(cat_id)] = float(np.mean(p50[p50 > -1])) if (p50 > -1).any() else -1.0
        rec["small_ap"] = float(ce.stats[3])
        rec["medium_ap"] = float(ce.stats[4])
        rec["large_ap"] = float(ce.stats[5])
        tp = rec["recall"] * n_gt
        rec["tp_count"] = round(tp, 1)
        rec["fp_count"] = round(tp / (rec["precision"] + 1e-12) - tp, 1) if rec["precision"] > 0 else None
        metrics[name] = rec

    payload = {"meta": {"n_images": seen, "seconds": round(time.time() - started, 2),
                        "sanity": sanity}, "metrics": metrics}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2)[:5000])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ann-file", required=True)
    p.add_argument("--img-folder", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--strategies", required=True,
                   help='JSON list of {name, strategy, arg}')
    p.add_argument("--device", default="cuda")
    p.add_argument("--weights", choices=("ema", "model"), default="ema")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    collect(parse_args())
