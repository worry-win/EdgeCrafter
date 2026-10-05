"""2x2 causal decomposition of After-HE: query-init vs decoder-value.

Factorizes After-HE's gain into two mechanisms that the current implementation
conflates:
  - I(M): query initialization (enc_score_head -> top-k -> initial reference /
          query content), driven by which memory feeds `_get_decoder_input`.
  - V(M): decoder cross-attention value/evidence, driven by which memory is
          passed to `TransformerDecoder.forward` as the `memory` argument.

Four conditions (Init x Value):
  NN = I(M_N), V(M_N)   baseline
  PN = I(M_P), V(M_N)   privileged query init, normal evidence
  NP = I(M_N), V(M_P)   normal query init, privileged evidence
  PP = I(M_P), V(M_P)   full After-HE (should reproduce existing strong)

The trick: `ECTransformer.forward` computes `memory = _get_encoder_input(feats)`,
then `init = _get_decoder_input(memory)`, then calls `decoder(init, memory, ...)`.
We simply call `_get_decoder_input` with one memory and pass the OTHER memory into
`decoder`, giving a clean split with no monkeypatching.
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
from scripts.ablation.dump_cmp5L_internal_behavior import _target_index  # noqa: E402
from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy  # noqa: E402
from engine.solver.ec_engine import (  # noqa: E402
    summarize_pr_curve_f1,
    summarize_yolo_pr_curve_metrics,
)

STRONG_SPEC = {"name": "strong", "background_weight": 0.2,
               "context_weight": None, "context_scale": 1.5}


def _get_init(ec, feats):
    memory, spatial_shapes = ec._get_encoder_input(feats)
    content, ref_unact, bboxes_list, logits_list = ec._get_decoder_input(memory, spatial_shapes)
    return content, ref_unact, memory, spatial_shapes


def _pack(decoder_tuple):
    """TransformerDecoder.forward returns an 8-tuple. In eval mode the
    ECTransformer.forward dict is just {'pred_logits': out_logits[-1],
    'pred_boxes': out_bboxes[-1], 'pred_masks': None}. Replicate that."""
    out_bboxes = decoder_tuple[0]   # [num_layers, B, Q, 4]
    out_logits = decoder_tuple[1]   # [num_layers, B, Q, C]
    return {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}


def run_conditions(model, feats_n, feats_p):
    """Return dict condition -> packed output dict for NN/PN/NP/PP."""
    ec = model.decoder

    # init from N and P
    content_n, ref_n, mem_n, shapes_n = _get_init(ec, feats_n)
    content_p, ref_p, mem_p, shapes_p = _get_init(ec, feats_p)

    def call(content, ref, mem, shapes):
        return ec.decoder(None, content, ref, mem, shapes,
                          ec.dec_bbox_head, ec.dec_score_head, ec.query_pos_head,
                          ec.pre_bbox_head, ec.integral, ec.up, ec.reg_scale,
                          continuous_bbox_head=ec.continuous_bbox_head,
                          attn_mask=None, dn_meta=None)

    return {
        # NN: normal init, normal value  (baseline)
        "NN": _pack(call(content_n, ref_n, mem_n, shapes_n)),
        # PN: privileged init, normal value
        "PN": _pack(call(content_p, ref_p, mem_n, shapes_n)),
        # NP: normal init, privileged value
        "NP": _pack(call(content_n, ref_n, mem_p, shapes_p)),
        # PP: privileged init, privileged value  (full After-HE)
        "PP": _pack(call(content_p, ref_p, mem_p, shapes_p)),
    }


def _iou_matrix(a, b):
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(-1)
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp(min=1e-9)


def lesion_success(boxes, logits, gt_xyxy, gt_class):
    iou = _iou_matrix(boxes, gt_xyxy[None])[:, 0]
    gt_logits = logits[:, gt_class]
    gt_scores = gt_logits.sigmoid()
    localized = iou >= 0.5
    cand = torch.nonzero(localized, as_tuple=False).flatten()
    if len(cand):
        det = int(cand[gt_scores[cand].argmax()])
    else:
        det = -1
    success = bool(len(cand) and gt_scores[det] >= 0.5)
    return {
        "success": success, "localised": bool(localized.any()),
        "q_detection": det, "q_iou": int(iou.argmax()),
        "q_cls": int(gt_logits.argmax()),
        "max_iou": float(iou.max()),
        "gt_score_q_detection": float(gt_scores[det]) if det >= 0 else 0.0,
        "gt_logit_q_iou": float(gt_logits[int(iou.argmax())]),
    }


def collect(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; refusing silent CPU fallback")

    rows = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    rows_by_image = {}
    for r in rows:
        rows_by_image.setdefault(int(r["image_id"]), []).append(r)

    cfg, solver, model = _build(args)
    model.eval()

    import copy
    evaluators = {c: copy.deepcopy(cfg.evaluator) for c in ("NN", "PN", "NP", "PP")}
    for e in evaluators.values():
        e.cleanup()

    parity = {}
    topk_stats = []
    lesion_records = {}
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
                feats_n, targets, samples.shape[-2:],
                STRONG_SPEC["background_weight"], None, STRONG_SPEC["context_scale"],
            )

            conds = run_conditions(model, feats_n, feats_p)

            # NN parity vs full model
            if "NN" not in parity:
                baseline = model(samples)
                parity["NN_logits"] = float((conds["NN"]["pred_logits"] - baseline["pred_logits"]).abs().max())
                parity["NN_boxes"] = float((conds["NN"]["pred_boxes"] - baseline["pred_boxes"]).abs().max())
                if max(parity["NN_logits"], parity["NN_boxes"]) > 2e-6:
                    raise RuntimeError(f"NN parity failed: {parity}")

            # PP parity vs standalone after-HE forward (ECTransformer.forward already
            # returns the packed dict)
            if "PP" not in parity:
                pp_standalone = model.decoder(feats_p)
                parity["PP_logits"] = float((conds["PP"]["pred_logits"] - pp_standalone["pred_logits"]).abs().max())
                parity["PP_boxes"] = float((conds["PP"]["pred_boxes"] - pp_standalone["pred_boxes"]).abs().max())
                if max(parity["PP_logits"], parity["PP_boxes"]) > 2e-6:
                    raise RuntimeError(f"PP parity failed: {parity}")

            # evaluate all four
            original_sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
            conds_cpu = {}
            for c in ("NN", "PN", "NP", "PP"):
                results = solver.postprocessor(conds[c], original_sizes)
                preds = {int(t["image_id"].item()): r for t, r in zip(targets, results)}
                evaluators[c].update(preds)
                conds_cpu[c] = {k: v.detach().cpu() for k, v in conds[c].items() if torch.is_tensor(v)}

            # top-k overlap / init comparison between N and P
            mem_n, shapes_n = model.decoder._get_encoder_input(feats_n)
            mem_p, shapes_p = model.decoder._get_encoder_input(feats_p)
            # top-k indices from enc_score_head
            with torch.no_grad():
                score_n = model.decoder.enc_score_head(mem_n).max(-1).values
                score_p = model.decoder.enc_score_head(mem_p).max(-1).values
            k = model.decoder.num_queries
            topk_n = torch.topk(score_n, k, dim=-1).indices
            topk_p = torch.topk(score_p, k, dim=-1).indices
            for bi in range(samples.shape[0]):
                tn = set(topk_n[bi].tolist())
                tp = set(topk_p[bi].tolist())
                overlap = len(tn & tp) / k
                topk_stats.append({
                    "image_id": image_ids[bi],
                    "topk_overlap": overlap,
                    "n_new_p": len(tp - tn),
                    "n_dropped_n": len(tn - tp),
                })

            # lesion success per condition
            for bi, target in enumerate(targets):
                image_id = image_ids[bi]
                if image_id not in rows_by_image:
                    continue
                gt_boxes_norm = box_cxcywh_to_xyxy(matcher_targets[bi]["boxes"]).cpu()
                ow, oh = [int(v) for v in target["orig_size"].tolist()]
                target_dev = {k: v.to(device) if torch.is_tensor(v) else v for k, v in target.items()}
                for row in rows_by_image[image_id]:
                    gi = _target_index(row, target_dev, ow, oh)
                    gt_xyxy = gt_boxes_norm[gi]
                    gc = int(row["gt_class"])
                    ann = int(row["ann_id"])
                    rec = {"ann_id": ann, "image_id": image_id, "gt_class": gc,
                           "class_name": row.get("class_name", str(gc)), "conditions": {}}
                    for c in ("NN", "PN", "NP", "PP"):
                        boxes = box_cxcywh_to_xyxy(conds_cpu[c]["pred_boxes"][bi])
                        logits = conds_cpu[c]["pred_logits"][bi]
                        rec["conditions"][c] = lesion_success(boxes, logits, gt_xyxy, gc)
                    lesion_records[ann] = rec

        seen += len(targets)
        if seen % 400 < len(targets):
            print(f"[{seen}] {time.time()-started:.1f}s", flush=True)

    # summarize
    metrics = {}
    for c in ("NN", "PN", "NP", "PP"):
        e = evaluators[c]
        e.synchronize_between_processes()
        e.accumulate()
        e.summarize()
        ce = e.coco_eval["bbox"]
        macro = summarize_pr_curve_f1(ce)
        yolo = summarize_yolo_pr_curve_metrics(ce, e.coco_gt)
        overall = (yolo or {}).get("yolo_overall") or {}
        # FP/TP count: derive from best-operating-point precision/recall vs GT count.
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
            "ann_file": args.ann_file, "n_images": seen,
            "condition": STRONG_SPEC["name"],
            "background_weight": STRONG_SPEC["background_weight"],
            "parity": parity, "seconds": round(time.time() - started, 2),
        },
        "metrics": metrics,
        "topk_stats": {
            "overlap_mean": float(np.mean([t["topk_overlap"] for t in topk_stats])),
            "overlap_median": float(np.median([t["topk_overlap"] for t in topk_stats])),
            "n": len(topk_stats),
            "per_image": topk_stats,
        },
        "lesions": [lesion_records[int(r["ann_id"])] for r in rows],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "lesions"}, ensure_ascii=False, indent=2)[:3000])


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
    p.add_argument("--max-batches", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    collect(parse_args())
