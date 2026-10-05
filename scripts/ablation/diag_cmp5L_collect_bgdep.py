"""Collect Normal-branch query features + GT BG-dependency labels (Experiment 1).

Stores compact float32 arrays in a .npz (NOT JSONL) to bound size:
  emb    : [N, 768]  per-layer query embedding (L0|L1|L2 concat)
  stats  : [N, S]    inference-available internal statistics
  d_bg   : [N]       GT far-BG dependency (mean over L0-L2), training target
  g_bg   : [N]       binary 1[d_bg > 0.20]
  image_id / query_idx : provenance

No GT feeds any predictor feature; GT only produces the label.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EC_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(EC_ROOT / "ecdetseg"))
sys.path.insert(0, str(EC_ROOT))

from scripts.ablation.probe_cmp5L_decoder_internal_tensors import (  # noqa: E402
    _build,
    _normalise_targets,
    DecoderInternalCapture,
)
from scripts.ablation.diag_cmp5L_adaptive_suppression import (  # noqa: E402
    region_mask,
)
from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy  # noqa: E402

RING_SCALE = 1.5
BGDEP_TAU = 0.20


def _capture_core_factory(rec):
    """Monkeypatch core to capture sampled-value norm (all-ones coeff)."""
    def core(value, value_spatial_shapes, sampling_locations,
             attention_weights, num_points_list):
        bs, n_head, c, _ = value[0].shape
        _, Len_q, _, _, _ = sampling_locations.shape
        grids = 2 * sampling_locations - 1
        grids = grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
        loc_list = grids.split(num_points_list, dim=-2)
        sampled_list = []
        for level, (h, w) in enumerate(value_spatial_shapes):
            value_l = value[level].reshape(bs * n_head, c, h, w)
            sv = F.grid_sample(value_l, loc_list[level], mode='bilinear',
                               padding_mode='zeros', align_corners=False)
            sampled_list.append(sv)
        sampled = torch.concat(sampled_list, dim=-1)  # [bs*H, c, Q, sumP]
        pv = sampled.reshape(bs, n_head, c, Len_q, -1)  # [bs,H,c,Q,P]
        rec["sampled_value_norm"] = pv.norm(dim=2).permute(0, 2, 1, 3).detach().clone()  # [bs,Q,H,P]
        attn = attention_weights.permute(0, 2, 1, 3).reshape(bs * n_head, 1, Len_q, -1)
        out = (sampled * attn).sum(-1).reshape(bs, n_head * c, Len_q)
        return out.permute(0, 2, 1)
    return core


# statistics layout: per layer (norm, entropy, max, alloc0, alloc1, alloc2, sampled_mean, sampled_std) = 8 * 3 = 24
# plus global (score, margin, box_size, box_aspect) = 4  -> 28
STAT_DIM = 28


def _query_vector(capture, bi, qi, score, margin, box_size, box_aspect):
    """Return (emb [768], stats [28]) for one query."""
    emb_parts = []
    stat_parts = [score, margin, box_size, box_aspect]
    for li in range(3):
        rec = capture.layers[li]
        qe = rec.get("query_in")
        if qe is not None:
            emb_parts.append(qe[bi, qi].float().cpu().numpy())
            stat_parts.append(float(qe[bi, qi].float().norm().item()))
        else:
            emb_parts.append(np.zeros(256, dtype=np.float32))
            stat_parts.append(0.0)
        if "attention_weights" in rec:
            aw = rec["attention_weights"][bi, qi]  # [H,P]
            stat_parts.append(float(-(aw * (aw + 1e-9).log()).sum().item()))
            stat_parts.append(float(aw.max().item()))
            npl = rec.get("num_points_list", (3, 6, 3))
            off = 0
            for n in npl:
                stat_parts.append(float(aw[:, off:off + n].sum().item()))
                off += n
        else:
            stat_parts += [0.0, 0.0, 0.0, 0.0, 0.0]
        if "sampled_value_norm" in rec:
            svn = rec["sampled_value_norm"][bi, qi]  # [H,P]
            stat_parts.append(float(svn.mean().item()))
            stat_parts.append(float(svn.std().item()))
        else:
            stat_parts += [0.0, 0.0]
    emb = np.concatenate(emb_parts).astype(np.float32)  # [768]
    stats = np.asarray(stat_parts, dtype=np.float32)
    return emb, stats


def collect(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    cfg, solver, model = _build(args)
    model.eval()

    emb_list, stats_list, d_list, g_list, iid_list, qidx_list = [], [], [], [], [], []
    seen = 0
    started = time.time()

    for samples, targets in cfg.val_dataloader:
        if args.limit and seen >= args.limit:
            break
        samples = samples.to(device)
        matcher_targets = _normalise_targets(targets, samples.shape[-2], samples.shape[-1], device)

        with torch.no_grad():
            with DecoderInternalCapture(model.decoder) as capture:
                for li in range(3):
                    layer = model.decoder.decoder.layers[li]
                    layer.cross_attn.ms_deformable_attn_core = _capture_core_factory(capture.layers[li])
                out = model(samples)

            logits = out["pred_logits"]
            boxes = out["pred_boxes"]
            scores = logits.sigmoid()

        B, Q, C = logits.shape
        for bi in range(B):
            gt_xyxy = box_cxcywh_to_xyxy(matcher_targets[bi]["boxes"])
            deps = []
            for li in range(3):
                rec = capture.layers[li]
                if "sampling_locations" in rec and "attention_weights" in rec:
                    loc = rec["sampling_locations"][bi]
                    attn = rec["attention_weights"][bi]
                    reg = region_mask(loc, gt_xyxy)
                    bg = (attn * (reg == 0).float()).sum(-1).mean(-1)
                    deps.append(bg)
            d_bar = torch.stack(deps).mean(0) if deps else torch.zeros(Q, device=device)

            q_score = scores[bi].max(-1).values
            q_cls = scores[bi].argmax(-1)
            top2 = scores[bi].topk(2, dim=-1).values
            q_margin = (top2[:, 0] - top2[:, 1])
            box_sizes = boxes[bi, :, 2] * boxes[bi, :, 3]
            box_aspect = boxes[bi, :, 2] / (boxes[bi, :, 3] + 1e-9)

            for qi in range(Q):
                emb, stats = _query_vector(capture, bi, qi,
                                           float(q_score[qi].item()),
                                           float(q_margin[qi].item()),
                                           float(box_sizes[qi].item()),
                                           float(box_aspect[qi].item()))
                emb_list.append(emb)
                stats_list.append(stats)
                d_list.append(float(d_bar[qi].item()))
                g_list.append(int(float(d_bar[qi].item()) > BGDEP_TAU))
                iid_list.append(int(targets[bi]["image_id"].item()))
                qidx_list.append(qi)

        seen += len(targets)
        if seen % 200 < len(targets):
            n = len(d_list)
            print(f"[{seen} images] {n} queries {time.time()-started:.1f}s", flush=True)

    emb = np.stack(emb_list).astype(np.float32)
    stats = np.stack(stats_list).astype(np.float32)
    d_bg = np.asarray(d_list, dtype=np.float32)
    g_bg = np.asarray(g_list, dtype=np.int8)
    iid = np.asarray(iid_list, dtype=np.int32)
    qidx = np.asarray(qidx_list, dtype=np.int32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, emb=emb, stats=stats, d_bg=d_bg, g_bg=g_bg,
        image_id=iid, query_idx=qidx, stat_dim=STAT_DIM)
    print(f"DONE {seen} images, {len(d_list)} queries, emb {emb.shape}, {time.time()-started:.1f}s")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ann-file", required=True)
    p.add_argument("--img-folder", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--weights", choices=("ema", "model"), default="ema")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    collect(parse_args())
