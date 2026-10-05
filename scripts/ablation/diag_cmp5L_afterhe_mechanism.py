"""Locate the internal mechanism of After-HE GT feature modulation.

Runs two branches (Normal vs Privileged-strong After-HE) per image with the
SAME EMA checkpoint, eval mode, transforms and parameters, and captures the
full per-layer decoder trace on both branches. The only difference between the
branches is the GT spatial mask applied to the HybridEncoder outputs.

Reuses, without modification:
  - privilege_features()             from probe_cmp5L_privileged_ema_oracle.py
  - DecoderInternalCapture           from probe_cmp5L_decoder_internal_tensors.py
  - _build / _normalise_targets      from probe_cmp5L_decoder_internal_tensors.py
  - _target_index                    from dump_cmp5L_internal_behavior.py

Sampled values V_l(p) are NOT hooked inside the deformable-attention core;
they are REBUILT offline from the captured `sampling_locations` and the decoder
input `memory` using the exact grid_sample formula of
`deformable_attention_core_func_v2`, then verified against the module's real
cross-attention output in the smoke run (see --smoke).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EC_ROOT / "ecdetseg"))
sys.path.insert(0, str(EC_ROOT))

from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy  # noqa: E402
from scripts.ablation.probe_cmp5L_privileged_ema_oracle import (  # noqa: E402
    privilege_features,
)
from scripts.ablation.probe_cmp5L_decoder_internal_tensors import (  # noqa: E402
    DecoderInternalCapture,
    _build,
    _normalise_targets,
)
from scripts.ablation.dump_cmp5L_internal_behavior import _target_index  # noqa: E402


STRONG_SPEC = {"name": "strong", "background_weight": 0.2,
               "context_weight": None, "context_scale": 1.5}


# ---------------------------------------------------------------------------
# Sampled-value reconstruction (offline, must match the module's core)
# ---------------------------------------------------------------------------

def rebuild_sampled_values(memory, spatial_shapes, sampling_locations,
                           num_points_list, num_heads, method="default"):
    """Replicate the per-point bilinear sample of the deformable-attention core.

    Reproduces TransformerDecoder.value_op for the cmp5L setting
    (value_proj=None, value_scale=None, memory_mask=None):
        value = memory.reshape(B, L, num_head, -1).permute(0,2,3,1)
        value = value.split([h*w for h,w in spatial_shapes], dim=-1)
    then grid_samples each level. Returns per-level sampled tensors
    [B*num_heads, c, Q, n_points_l] and the per-level value tensors
    [B, num_heads, c, h*w].
    """
    bs, L, C = memory.shape
    c_per_head = C // num_heads
    # value_op: [B, L, H, c] -> [B, H, c, L]
    value = memory.view(bs, L, num_heads, c_per_head).permute(0, 2, 3, 1)
    split_shape = [h * w for h, w in spatial_shapes]
    # value list per level: [B, H, c, h*w]
    value_list = list(torch.split(value, split_shape, dim=-1))

    if method == "default":
        grids = 2 * sampling_locations - 1
    else:
        grids = sampling_locations
    # sampling_locations: [B, Q, H, sum_points, 2]
    grids = grids.permute(0, 2, 1, 3, 4).flatten(0, 1)  # [B*H, Q, sum_points, 2]
    grid_list = list(grids.split(list(num_points_list), dim=-2))

    sampled_list = []
    for level, (h, w) in enumerate(spatial_shapes):
        value_l = value_list[level].reshape(bs * num_heads, c_per_head, h, w)
        sampled_l = F.grid_sample(
            value_l, grid_list[level], mode="bilinear",
            padding_mode="zeros", align_corners=False,
        )  # [B*H, c, Q, n_points_l]
        sampled_list.append(sampled_l)
    return sampled_list, value_list


# ---------------------------------------------------------------------------
# Per-lesion, per-branch compact trace
# ---------------------------------------------------------------------------

def _cos(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    denom = (a.norm() * b.norm()).clamp(min=1e-9)
    return float((a * b).sum() / denom)


def _norm(a):
    return float(a.float().norm())


def layer_trace(snapshot, replay, batch_index, query_id):
    """Compact per-layer metrics for ONE query id in one branch."""
    layers = []
    memory = snapshot["decoder_input"]["memory"][batch_index]
    spatial_shapes = snapshot["decoder_input"]["spatial_shapes"]
    for index, rec in enumerate(snapshot["layers"]):
        loc = rec["sampling_locations"][batch_index, query_id]      # [H, P, 2]
        wgt = rec["attention_weights"][batch_index, query_id]        # [H, P]
        q_in = rec["query_in"][batch_index, query_id]
        q_out = rec["query_out"][batch_index, query_id]
        cross = rec["cross_attn_output"][batch_index, query_id]
        ffn_act = rec["ffn_activation"][batch_index, query_id]
        lqe = replay["lqe_logits"][index][batch_index, query_id]
        box = replay["boxes"][index][batch_index, query_id]
        layers.append({
            "layer": index,
            "sampling_loc_mean": float(loc.float().mean()),
            "sampling_loc_std": float(loc.float().std(unbiased=False)),
            "attn_entropy": float(-(wgt.float() * (wgt.float().clamp(min=1e-12).log())).sum()),
            "attn_max": float(wgt.float().max()),
            "query_norm": _norm(q_out),
            "query_move": _norm(q_out - q_in),
            "cross_norm": _norm(cross),
            "ffn_act_norm": _norm(ffn_act),
            "gt_logit": float(lqe[int(replay["gt_class"])]),
            "top1_logit": float(lqe.max()),
            "box_cxcywh": [float(v) for v in box],
        })
    return layers


def branch_outputs(outputs, matcher_targets, batch_index):
    """Decode branch prediction into per-query boxes / logits / scores."""
    boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"][batch_index])  # [Q,4] xyxy norm
    logits = outputs["pred_logits"][batch_index]                     # [Q,4]
    scores = logits.sigmoid()
    return boxes, logits, scores


def _iou_matrix(a, b):
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(-1)
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp(min=1e-9)


def lesion_branch_metrics(boxes, logits, gt_xyxy, gt_class):
    """Leison-level detection endpoints for one branch (matches existing def)."""
    iou = _iou_matrix(boxes, gt_xyxy[None])[:, 0]
    gt_logits = logits[:, gt_class]
    gt_scores = gt_logits.sigmoid()
    final_scores = logits.sigmoid().max(-1).values
    localized = iou >= 0.5
    candidates = torch.nonzero(localized, as_tuple=False).flatten()
    if len(candidates):
        detection = int(candidates[gt_scores[candidates].argmax()])
    else:
        detection = -1
    success = bool(len(candidates) and gt_scores[detection] >= 0.5)
    q_iou = int(iou.argmax())
    q_cls = int(gt_logits.argmax())
    order = torch.argsort(final_scores, descending=True)
    return {
        "success": success,
        "localised": bool(localized.any()),
        "q_detection": detection,
        "q_iou": q_iou,
        "q_cls": q_cls,
        "q_global": int(final_scores.argmax()),
        "max_iou": float(iou.max()),
        "iou_q_detection": float(iou[detection]) if detection >= 0 else -1.0,
        "gt_score_q_detection": float(gt_scores[detection]) if detection >= 0 else 0.0,
        "gt_logit_q_iou": float(gt_logits[q_iou]),
        "class_margin_q_iou": float(gt_logits[q_iou] - logits[q_iou, torch.arange(logits.shape[-1]) != gt_class].max()),
        "iou_q_cls": float(iou[q_cls]),
        "top1_iou": float(iou[order[:1]].max()),
        "top5_iou": float(iou[order[:min(5, len(order))]].max()),
    }


def group_of(normal_ok, priv_ok):
    if not normal_ok and priv_ok:
        return "N-P+"
    if normal_ok and priv_ok:
        return "N+P+"
    if normal_ok and not priv_ok:
        return "N+P-"
    return "N-P-"


def collect(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; refusing silent CPU fallback")

    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    if args.lesion_limit:
        rows = rows[:args.lesion_limit]
    rows_by_image = {}
    for row in rows:
        rows_by_image.setdefault(int(row["image_id"]), []).append(row)

    cfg, solver, model = _build(args)
    model.eval()

    lesion_records = {}
    parity = None
    smoke_rebuild_err = []
    seen = 0
    started = time.time()

    for samples, targets in cfg.val_dataloader:
        image_ids = [int(t["image_id"].item()) for t in targets]
        if not any(i in rows_by_image for i in image_ids):
            continue
        samples = samples.to(device)
        matcher_targets = _normalise_targets(targets, samples.shape[-2], samples.shape[-1], device)

        with torch.no_grad():
            base_features = model.backbone(samples)
            encoded = model.encoder([f.clone() for f in base_features])

            # Branch N: normal
            with DecoderInternalCapture(model.decoder) as cap_n:
                out_n = model.decoder([f.clone() for f in encoded])
            if isinstance(out_n, (tuple, list)):
                out_n = out_n[0]

            # Branch P: privileged strong After-HE
            masked, _ = privilege_features(
                encoded, targets, samples.shape[-2:],
                STRONG_SPEC["background_weight"], None, STRONG_SPEC["context_scale"],
            )
            with DecoderInternalCapture(model.decoder) as cap_p:
                out_p = model.decoder(masked)
            if isinstance(out_p, (tuple, list)):
                out_p = out_p[0]

            # parity: the ONLY difference must be the mask. normal branch via
            # decoder() must equal the full model() forward.
            if parity is None:
                baseline = model(samples)
                parity = {
                    "normal_logits": float((out_n["pred_logits"] - baseline["pred_logits"]).abs().max()),
                    "normal_boxes": float((out_n["pred_boxes"] - baseline["pred_boxes"]).abs().max()),
                }
                if max(parity.values()) > 2e-6:
                    raise RuntimeError(f"normal branch parity failed: {parity}")

            replay_n = cap_n.replay_decoder_heads()
            replay_p = cap_p.replay_decoder_heads()
            snap_n = cap_n.cpu_snapshot()
            snap_p = cap_p.cpu_snapshot()

            out_n_cpu = {k: v.detach().cpu() for k, v in out_n.items() if torch.is_tensor(v)}
            out_p_cpu = {k: v.detach().cpu() for k, v in out_p.items() if torch.is_tensor(v)}

            # sampled-value rebuild verification (once)
            if args.smoke and not smoke_rebuild_err:
                err = _verify_sampled_value_rebuild(snap_n, replay_n, model.decoder)
                smoke_rebuild_err.append(err)

            for bi, target in enumerate(targets):
                image_id = image_ids[bi]
                if image_id not in rows_by_image:
                    continue
                gt_boxes_norm = box_cxcywh_to_xyxy(matcher_targets[bi]["boxes"]).cpu()
                boxes_n, logits_n, _ = branch_outputs(out_n_cpu, matcher_targets, bi)
                boxes_p, logits_p, _ = branch_outputs(out_p_cpu, matcher_targets, bi)
                boxes_n = boxes_n.cpu(); logits_n = logits_n.cpu()
                boxes_p = boxes_p.cpu(); logits_p = logits_p.cpu()

                original_w, original_h = [int(v) for v in target["orig_size"].tolist()]
                target_dev = {k: v.to(device) if torch.is_tensor(v) else v for k, v in target.items()}

                for row in rows_by_image[image_id]:
                    gt_index = _target_index(row, target_dev, original_w, original_h)
                    gt_xyxy = gt_boxes_norm[gt_index]
                    gt_class = int(row["gt_class"])
                    ann_id = int(row["ann_id"])

                    m_n = lesion_branch_metrics(boxes_n, logits_n, gt_xyxy, gt_class)
                    m_p = lesion_branch_metrics(boxes_p, logits_p, gt_xyxy, gt_class)
                    grp = group_of(m_n["success"], m_p["success"])

                    # trace at the detection query and the highest-IoU query,
                    # separately for same-query vs reassignment analysis.
                    q_det_n, q_det_p = m_n["q_detection"], m_p["q_detection"]
                    q_iou_n, q_iou_p = m_n["q_iou"], m_p["q_iou"]

                    lesion_records[ann_id] = {
                        "ann_id": ann_id, "image_id": image_id, "gt_class": gt_class,
                        "class_name": row.get("class_name", str(gt_class)),
                        "group": grp,
                        "normal": m_n, "priv": m_p,
                        "same_detection_query": q_det_n == q_det_p,
                        "same_iou_query": q_iou_n == q_iou_p,
                        "trace_det_n": layer_trace(snap_n, _replay_with_cls(replay_n, gt_class), bi, q_det_n) if q_det_n >= 0 else None,
                        "trace_det_p": layer_trace(snap_p, _replay_with_cls(replay_p, gt_class), bi, q_det_p) if q_det_p >= 0 else None,
                        "trace_iou_n": layer_trace(snap_n, _replay_with_cls(replay_n, gt_class), bi, q_iou_n),
                        "trace_iou_p": layer_trace(snap_p, _replay_with_cls(replay_p, gt_class), bi, q_iou_p),
                        # matched-query traces: same index across both branches,
                        # to isolate same-query representation change from reassignment
                        "trace_matched_iou_n": layer_trace(snap_n, _replay_with_cls(replay_n, gt_class), bi, q_iou_n),
                        "trace_matched_iou_p": layer_trace(snap_p, _replay_with_cls(replay_p, gt_class), bi, q_iou_n),
                        "trace_matched_det_n": layer_trace(snap_n, _replay_with_cls(replay_n, gt_class), bi, q_det_n) if q_det_n >= 0 else None,
                        "trace_matched_det_p": layer_trace(snap_p, _replay_with_cls(replay_p, gt_class), bi, q_det_n) if q_det_n >= 0 else None,
                    }

        seen += len(targets)
        if seen % 200 < len(targets):
            print(f"[{seen}] {time.time()-started:.1f}s", flush=True)

    payload = {
        "meta": {
            "checkpoint": args.checkpoint, "weights": args.weights,
            "ann_file": args.ann_file, "n_images": seen,
            "n_lesions": len(lesion_records), "condition": STRONG_SPEC["name"],
            "background_weight": STRONG_SPEC["background_weight"],
            "normal_parity": parity,
            "smoke_rebuild_max_abs_err": smoke_rebuild_err,
            "seconds": round(time.time() - started, 2),
        },
        "lesions": [lesion_records[int(row["ann_id"])] for row in rows],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["meta"], ensure_ascii=False, indent=2))


def _replay_with_cls(replay, gt_class):
    replay["gt_class"] = gt_class
    return replay


def _verify_sampled_value_rebuild(snapshot, replay, ec_decoder):
    """Verify offline sampled-value rebuild reproduces the module's aggregation."""
    memory = snapshot["decoder_input"]["memory"]
    shapes = snapshot["decoder_input"]["spatial_shapes"]
    rec0 = snapshot["layers"][0]
    loc = rec0["sampling_locations"]
    wgt = rec0["attention_weights"]
    num_points = tuple(rec0["num_points_list"])
    n_head = loc.shape[2]
    sampled, _ = rebuild_sampled_values(memory, shapes, loc, num_points, n_head)
    # reconstruct cross-attn output and compare against captured one
    bs, Q, H, P = wgt.shape
    c = sampled[0].shape[1]
    concat = torch.cat(sampled, dim=-1)  # [B*H, c, Q, P]
    attn = wgt.permute(0, 2, 1, 3).reshape(bs * H, 1, Q, P)
    out = (concat * attn).sum(-1).reshape(bs, H * c, Q).permute(0, 2, 1)  # [B,Q,H*c]
    captured = rec0["cross_attn_output"]
    err = float((out - captured).abs().max())
    return err


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
    p.add_argument("--lesion-limit", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="verify sampled-value rebuild")
    p.add_argument("--max-batches", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    collect(parse_args())
