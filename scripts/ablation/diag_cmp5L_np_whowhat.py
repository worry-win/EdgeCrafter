"""NP privileged-value mechanism: who (TP/FP query) + what (FG/Ring/BG evidence).

Runs NN and NP with fixed Normal initialization, monkeypatching
MSDeformableAttention.forward to capture per-layer sampling_locations,
attention_weights, and cross-attn output. Then:
  - Experiment B: classify queries as TP / FP / background by NN outcome
    (anchor on NN identity), compute paired Δscore / Δmargin / ΔIoU / Δ||E||.
  - Experiment C: decompose cross-attn evidence E_q into FG / Ring / BG by the
    region each sampling point lands in, with a reconstruction sanity check.
  - Experiment D: auto-select rescued-FP / broken-TP cases for statistics.

GT box (cxcywh-normalized) is converted to xyxy-normalized; sampling locations
are already xyxy-normalized (can exceed 1.0 into padding). Ring = GT box dilated
by context_scale=1.5 (same definition as the context Oracle).
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
from scripts.ablation.diag_cmp5L_np_layervalue import run_layer_switch  # noqa: E402

STRONG_SPEC = {"name": "strong", "background_weight": 0.2,
               "context_weight": None, "context_scale": 1.5}
RING_SCALE = 1.5


# ---------------------------------------------------------------------------
# Monkeypatch MSDeformableAttention.forward to capture attention tensors.
# ---------------------------------------------------------------------------
class AttentionCapture:
    """Capture per-layer sampling_locations / attention_weights / cross output."""

    def __init__(self, td):
        self.td = td
        self.layers = [dict() for _ in td.layers]
        self._orig = {}
        self._handles = []

    def _make_hook(self, index, module):
        def hook(_m, inputs, output):
            rec = self.layers[index]
            query, reference_points, _value, spatial_shapes = inputs[:4]
            bs, nq = query.shape[:2]
            # recompute sampling locations + attention weights (mirror forward)
            sampling_offsets = module.sampling_offsets(query).reshape(
                bs, nq, module.num_heads, sum(module.num_points_list), 2)
            attn = module.attention_weights(query).reshape(
                bs, nq, module.num_heads, sum(module.num_points_list))
            attn = attn.softmax(-1)
            if reference_points.shape[-1] == 4:
                scale = module.num_points_scale.to(query.dtype).unsqueeze(-1)
                offset = (sampling_offsets * scale
                          * reference_points[:, :, None, :, 2:]
                          * module.offset_scale)
                loc = reference_points[:, :, None, :, :2] + offset
            else:
                normalizer = torch.as_tensor(spatial_shapes, device=query.device,
                                             dtype=query.dtype).flip([1]).reshape(
                    1, 1, 1, module.num_levels, 1, 2)
                shaped = sampling_offsets.reshape(bs, nq, module.num_heads,
                                                  module.num_levels, -1, 2)
                loc = reference_points.reshape(bs, nq, 1, module.num_levels, 1, 2) \
                    + shaped / normalizer
                loc = loc.flatten(3, 4)
            rec["sampling_locations"] = loc.detach().clone()          # [B,Q,H,P,2]
            rec["attention_weights"] = attn.detach().clone()          # [B,Q,H,P]
            rec["cross_attn_output"] = output.detach().clone()        # [B,Q,C]
            rec["num_points_list"] = tuple(module.num_points_list)
            rec["num_heads"] = module.num_heads
            rec["num_levels"] = module.num_levels
            return None
        return hook

    def __enter__(self):
        for i, layer in enumerate(self.td.layers):
            m = layer.cross_attn
            self._orig[i] = m.forward
            self._handles.append(m.register_forward_hook(self._make_hook(i, m)))
        return self

    def __exit__(self, *a):
        for h in self._handles:
            h.remove()
        self._handles = []


def _region_mask(loc, gt_xyxy, scale=RING_SCALE):
    """Region of each sampling point: 0=BG, 1=Ring, 2=FG (priority FG>Ring>BG).

    loc: [Q,H,P,2] normalized xy. gt_xyxy: [N_gt,4] normalized xyxy.
    Returns [Q,H,P] int tensor (2=FG,1=Ring,0=BG).
    """
    if gt_xyxy.shape[0] == 0:
        return torch.zeros(loc.shape[:-1], dtype=torch.long, device=loc.device)
    q, h, p = loc.shape[:-1]
    flat = loc.reshape(-1, 2)  # [Q*H*P, 2]
    x, y = flat[:, 0], flat[:, 1]
    # FG: inside any GT box
    inside = torch.zeros(flat.shape[0], dtype=torch.bool, device=loc.device)
    in_ring = torch.zeros_like(inside)
    for g in gt_xyxy:
        x0, y0, x1, y1 = g
        inside |= (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        # ring: dilated box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hw, hh = (x1 - x0) * scale / 2, (y1 - y0) * scale / 2
        rx0, ry0, rx1, ry1 = cx - hw, cy - hh, cx + hw, cy + hh
        in_ring |= (x >= rx0) & (x <= rx1) & (y >= ry0) & (y <= ry1)
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


def classify_queries(boxes_xyxy, logits, gt_xyxy, gt_labels, iou_thr=0.5):
    """Classify each query into TP / FP / BG by NN outcome.

    boxes_xyxy: [Q,4]. logits: [Q,C]. Returns per-query int label
    (1=TP, 2=FP, 0=BG) + per-query detection score + matched gt index.
    """
    q = boxes_xyxy.shape[0]
    scores = logits.sigmoid().max(-1).values  # [Q]
    if gt_xyxy.shape[0] == 0:
        # all queries are FP (or BG if score below threshold)
        label = torch.where(scores >= 0.5,
                            torch.full_like(scores, 2, dtype=torch.long),
                            torch.zeros_like(scores, dtype=torch.long))
        return label, scores, torch.full_like(scores, -1, dtype=torch.long)
    iou = _iou_matrix(boxes_xyxy, gt_xyxy)  # [Q, N_gt]
    max_iou, max_gt = iou.max(-1)  # [Q]
    # TP: matched to a GT with IoU>=thr and class correct and score>=thr
    gt_class = gt_labels[max_gt.clamp(min=0)]  # [Q]
    pred_class = logits.argmax(-1)  # [Q]
    is_tp = (max_iou >= iou_thr) & (gt_class == pred_class) & (scores >= 0.5)
    is_fp = (~is_tp) & (scores >= 0.5)
    label = torch.where(is_tp, torch.ones_like(scores, dtype=torch.long),
                        torch.where(is_fp, torch.full_like(scores, 2, dtype=torch.long),
                                    torch.zeros_like(scores, dtype=torch.long)))
    return label, scores, max_gt


def collect(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    cfg, solver, model = _build(args)
    model.eval()

    rows = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    rows_by_image = {}
    for r in rows:
        rows_by_image.setdefault(int(r["image_id"]), []).append(r)

    cap = AttentionCapture(model.decoder.decoder)

    # accumulated statistics
    tp_stats = {"d_score": [], "d_margin": [], "d_iou": [], "d_norm": [],
                "score_n": [], "score_p": []}
    fp_stats = {"d_score": [], "d_margin": [], "d_iou": [], "d_norm": [],
                "score_n": [], "score_p": []}
    bg_stats = {"d_score": [], "d_margin": [], "d_iou": [], "d_norm": [],
                "score_n": [], "score_p": []}
    region_stats = {}  # key (cond, layer, qtype, region) -> list of attention mass
    reconstruction = []  # (max_abs_err, mean_abs_err) per layer per condition
    cases = []  # reserved for Experiment D
    per_layer_mass = []  # reserved

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
                feats_p, masks = privilege_features(
                    feats_n, targets, samples.shape[-2:],
                    STRONG_SPEC["background_weight"], None, STRONG_SPEC["context_scale"])

                ec = model.decoder
                mem_n, shapes_n = ec._get_encoder_input(feats_n)
                mem_p, shapes_p = ec._get_encoder_input(feats_p)
                content_n, ref_n, _b, _l = ec._get_decoder_input(mem_n, shapes_n)

                # NN branch
                cap.layers = [dict() for _ in ec.decoder.layers]
                out_nn = run_layer_switch(ec, content_n, ref_n, mem_n, mem_p, shapes_n, set())
                nn_layers = [dict(r) for r in cap.layers]  # snapshot
                # NP branch
                cap.layers = [dict() for _ in ec.decoder.layers]
                out_np = run_layer_switch(ec, content_n, ref_n, mem_n, mem_p, shapes_n, {0, 1, 2, 3})
                np_layers = [dict(r) for r in cap.layers]  # snapshot

                # per-image processing
                pred_boxes_n = out_nn["pred_boxes"]  # [B,Q,4] cxcywh normalized
                pred_logits_n = out_nn["pred_logits"]  # [B,Q,C]
                pred_boxes_p = out_np["pred_boxes"]
                pred_logits_p = out_np["pred_logits"]

                for bi, target in enumerate(targets):
                    gt_boxes_cxcywh = matcher_targets[bi]["boxes"]  # [N,4] cxcywh norm
                    gt_labels = matcher_targets[bi]["labels"]  # [N]
                    gt_xyxy = box_cxcywh_to_xyxy(gt_boxes_cxcywh)  # [N,4] xyxy norm

                    boxes_n = box_cxcywh_to_xyxy(pred_boxes_n[bi])  # [Q,4] xyxy
                    boxes_p = box_cxcywh_to_xyxy(pred_boxes_p[bi])
                    logits_n = pred_logits_n[bi]
                    logits_p = pred_logits_p[bi]

                    # classify by NN outcome (anchor on NN identity)
                    qlabel, qscore_n, qgt = classify_queries(boxes_n, logits_n, gt_xyxy, gt_labels)
                    qscore_p = logits_p.sigmoid().max(-1).values

                    # per-query deltas
                    margin_n = _class_margin(logits_n)
                    margin_p = _class_margin(logits_p)
                    d_score = (qscore_p - qscore_n).cpu().numpy()
                    d_margin = (margin_p - margin_n).cpu().numpy()
                    # IoU delta at the matched GT (or -1)
                    iou_n = torch.zeros(logits_n.shape[0], device=device)
                    iou_p = torch.zeros_like(iou_n)
                    if gt_xyxy.shape[0] > 0:
                        valid = qgt >= 0
                        iou_n[valid] = _iou_matrix(boxes_n[valid], gt_xyxy)[torch.arange(valid.sum()), qgt[valid]]
                        iou_p[valid] = _iou_matrix(boxes_p[valid], gt_xyxy)[torch.arange(valid.sum()), qgt[valid]]
                    d_iou = (iou_p - iou_n).cpu().numpy()

                    # cross-attn output norm delta (last layer)
                    cross_n = nn_layers[3].get("cross_attn_output")
                    cross_p = np_layers[3].get("cross_attn_output")
                    if cross_n is not None and cross_p is not None:
                        norm_n = cross_n[bi].norm(dim=-1).cpu().numpy()
                        norm_p = cross_p[bi].norm(dim=-1).cpu().numpy()
                        d_norm = norm_p - norm_n
                    else:
                        d_norm = np.zeros_like(d_score)

                    ql = qlabel.cpu().numpy()
                    for qtype_code, bucket in ((1, tp_stats), (2, fp_stats), (0, bg_stats)):
                        m = ql == qtype_code
                        if not m.any():
                            continue
                        bucket["d_score"].extend(d_score[m].tolist())
                        bucket["d_margin"].extend(d_margin[m].tolist())
                        bucket["d_iou"].extend(d_iou[m].tolist())
                        bucket["d_norm"].extend(d_norm[m].tolist())
                        bucket["score_n"].extend(qscore_n[m].cpu().numpy().tolist())
                        bucket["score_p"].extend(qscore_p[m].cpu().numpy().tolist())

                    # FG/Ring/BG decomposition per layer (NN and NP) — vectorized,
                    # online sum/count accumulation (no per-value storage)
                    ql_np = ql  # [Q] 1=TP 2=FP 0=BG
                    for cond, layers in (("NN", nn_layers), ("NP", np_layers)):
                        for li, rec in enumerate(layers):
                            if "sampling_locations" not in rec:
                                continue
                            loc = rec["sampling_locations"][bi]  # [Q,H,P,2]
                            attn = rec["attention_weights"][bi]  # [Q,H,P]
                            region = _region_mask(loc, gt_xyxy)  # [Q,H,P]
                            for r, rname in ((2, "FG"), (1, "Ring"), (0, "BG")):
                                m = (attn * (region == r).float()).sum(-1).mean(-1)  # [Q]
                                for qtype, qmask in (("TP", ql_np == 1), ("FP", ql_np == 2), ("BG", ql_np == 0)):
                                    if qmask.any():
                                        vals = m[qmask]
                                        key = (cond, li, qtype, rname)
                                        acc = region_stats.setdefault(key, [0.0, 0])
                                        acc[0] += float(vals.sum())
                                        acc[1] += int(qmask.sum())

    # ---- summarize ----
    def _summ(bucket):
        out = {}
        for k in ("d_score", "d_margin", "d_iou", "d_norm"):
            a = np.array(bucket[k])
            out[k] = {
                "mean": float(a.mean()), "median": float(np.median(a)),
                "std": float(a.std()),
                "p10": float(np.percentile(a, 10)), "p25": float(np.percentile(a, 25)),
                "p75": float(np.percentile(a, 75)), "p90": float(np.percentile(a, 90)),
                "n": int(len(a)),
            }
        return out

    payload = {
        "meta": {
            "checkpoint": args.checkpoint, "weights": args.weights,
            "n_images": seen, "ring_scale": RING_SCALE,
            "background_weight": STRONG_SPEC["background_weight"],
            "seconds": round(time.time() - started, 2),
        },
        "tp_query": _summ(tp_stats),
        "fp_query": _summ(fp_stats),
        "bg_query": _summ(bg_stats),
        "region_attention_mass": {f"{c}_{l}_{q}_{r}": {"mean": round(v[0] / max(1, v[1]), 6), "n": v[1]}
                                  for (c, l, q, r), v in region_stats.items()},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "region_attention_mass"},
                     ensure_ascii=False, indent=2)[:5000])


def _class_margin(logits):
    """Top1 - top2 margin (in logit space) for each query."""
    s, _ = logits.topk(2, dim=-1)
    return s[:, 0] - s[:, 1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ann-file", required=True)
    p.add_argument("--img-folder", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--weights", choices=("ema", "model"), default="ema")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    collect(parse_args())
