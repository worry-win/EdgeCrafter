"""Perception-ceiling probe over the low-threshold cmp5L dumps.

The dumps keep every one of the 300 queries per image together with, for each
query, the annotation id and IoU of its best *same-class* ground truth
(``rec_gt_cls_id`` / ``rec_gt_cls_iou``; id = -1 when no same-class GT overlaps
at all). That makes it possible to separate two things that a score-thresholded
dump can never separate:

  (a) the model has NO query that localises the lesion  -> capacity limit
  (b) some query does localise it, but its score is low -> decision limit

Definitions used here, per arm:

  sensed(g)   = exists a query q with rec_gt_cls_id[q] == g and
                rec_gt_cls_iou[q] >= tau            (an ORACLE upper bound:
                ignore the one-to-one greedy competition that AP applies)
  score(g)    = max rec_score over those queries     (how confident the arm
                is about the best localisation it has for g)

  perception ceiling  = #sensed / n_gt                (upper bound on recall)
  decision loss       = #sensed-but-below-sigma / n_gt (recall you could get
                        back by scene-specific re-scoring, not by new capacity)

Usage
-----
    python scripts/ablation/probe_cmp5L_perception_ceiling.py \
        --pred-dir outputs/ablation/cmp5L_predictions --split test \
        --arms lw_xlarge dinov2b dinov2s mae_vitb ecvits_official \
        --tau 0.5 --out outputs/ablation/cmp5L_perception_ceiling_test.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EC_ROOT = Path(__file__).resolve().parents[2]

CLASS_NAMES = {0: "实(囊实)性", 1: "囊性", 2: "淋巴结", 3: "导管相关病变"}

# Score buckets for a lesion the arm *does* localise: how confident is it?
BUCKETS = [("high_ge_0.50", 0.50, np.inf),
           ("mid_0.25_0.50", 0.25, 0.50),
           ("low_0.10_0.25", 0.10, 0.25),
           ("faint_0.05_0.10", 0.05, 0.10),
           ("traces_lt_0.05", 0.0, 0.05)]


def load_arm(pred_dir, name, split):
    p = Path(pred_dir) / f"{name}_{split}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return {
        "path": str(p),
        "rec_ptr": d["rec_ptr"],
        "score": d["rec_score"],
        "cls_id": d["rec_gt_cls_id"],
        "cls_iou": d["rec_gt_cls_iou"],
        "image_ids": d["image_ids"],
        "gt_ptr": d["gt_ptr"],
        "gt_ann": d["gt_ann_id"],
        "gt_cls": d["gt_cls"],
        "gt_area": d["gt_area"],
    }


def probe(arm, tau, sigmas):
    """Return the perception-ceiling block for one arm."""
    rec_ptr, cls_id, cls_iou, score = arm["rec_ptr"], arm["cls_id"], arm["cls_iou"], arm["score"]
    n_img = len(arm["image_ids"])

    # ann_id -> best same-class IoU, and the max score among queries that reach tau
    best_iou = {}
    best_score_at_tau = {}
    also_any_score = {}          # max score among *all* queries that touch this ann (any class)

    for i in range(n_img):
        lo, hi = int(rec_ptr[i]), int(rec_ptr[i + 1])
        if hi <= lo:
            continue
        ids = cls_id[lo:hi]
        ious = cls_iou[lo:hi]
        sc = score[lo:hi]
        touched = np.where(ids >= 0)[0]
        for t in touched:
            a = int(ids[t])
            if ious[t] > best_iou.get(a, -1.0):
                best_iou[a] = float(ious[t])
            if sc[t] > also_any_score.get(a, -1.0):
                also_any_score[a] = float(sc[t])
            if ious[t] >= tau:
                if sc[t] > best_score_at_tau.get(a, -1.0):
                    best_score_at_tau[a] = float(sc[t])

    # the canonical GT population of this arm's split
    gt_ann = arm["gt_ann"]
    gt_cls = arm["gt_cls"]
    gt_area = arm["gt_area"]
    n_gt = int(len(gt_ann))

    sensed = np.zeros(n_gt, dtype=bool)
    for k in range(n_gt):
        sensed[k] = int(gt_ann[k]) in best_score_at_tau

    out = {
        "n_gt": n_gt,
        "sensed_ge_tau": int(sensed.sum()),
        "perception_ceiling": round(float(sensed.mean()), 4),
    }

    # recall at each sigma, computed the same way (query survives sigma AND localises)
    for s in sigmas:
        got = 0
        for k in range(n_gt):
            a = int(gt_ann[k])
            if best_iou.get(a, 0.0) >= tau and also_any_score.get(a, 0.0) >= s:
                got += 1
        out["recall_at_sigma_%.2f" % s] = round(got / max(n_gt, 1), 4)

    # score buckets over the sensed population
    buckets = {b[0]: 0 for b in BUCKETS}
    bucket_cls = {b[0]: defaultdict(int) for b in BUCKETS}
    for k in range(n_gt):
        a = int(gt_ann[k])
        if a not in best_score_at_tau:
            continue
        s = best_score_at_tau[a]
        for bname, lo, hi in BUCKETS:
            if lo <= s < hi:
                buckets[bname] += 1
                bucket_cls[bname][CLASS_NAMES.get(int(gt_cls[k]), str(gt_cls[k]))] += 1
                break
    out["sensed_score_buckets"] = buckets
    out["sensed_score_buckets_by_class"] = {k: dict(v) for k, v in bucket_cls.items()}

    # never-sensed (capacity limit) broken down by class / scale
    never = defaultdict(int)
    never_cls = defaultdict(int)
    for k in range(n_gt):
        if sensed[k]:
            continue
        a = int(gt_area[k])
        scale = "small" if a < 32 ** 2 else "medium" if a < 96 ** 2 else "large"
        never[scale] += 1
        never_cls[CLASS_NAMES.get(int(gt_cls[k]), str(gt_cls[k]))] += 1
    out["never_sensed"] = int((~sensed).sum())
    out["never_sensed_by_class"] = dict(never_cls)
    out["never_sensed_by_scale"] = dict(never)

    # ann-id lists so that several arms can be intersected afterwards
    # (-> the *union* perception ceiling: lesions no arm localises at all)
    out["sensed_ann_ids"] = [int(gt_ann[k]) for k in range(n_gt) if sensed[k]]
    out["never_sensed_ann_ids"] = [int(gt_ann[k]) for k in range(n_gt) if not sensed[k]]

    # per class ceiling
    per_cls = {}
    for c in sorted(set(int(v) for v in gt_cls)):
        idx = np.where(gt_cls == c)[0]
        if len(idx) == 0:
            continue
        ns = int(sum(1 for k in idx if not sensed[k]))
        n = len(idx)
        per_cls[str(c)] = {
            "class_name": CLASS_NAMES.get(c, str(c)),
            "n_gt": n,
            "sensed": n - ns,
            "ceiling": round((n - ns) / n, 4),
            "never_sensed": ns,
            "never_sensed_frac": round(ns / n, 4),
        }
    out["by_class"] = per_cls
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.05, 0.25, 0.5])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    if not pred_dir.is_absolute():
        pred_dir = EC_ROOT / pred_dir

    results = {"split": args.split, "tau": args.tau, "arms": {}}
    for name in args.arms:
        arm = load_arm(pred_dir, name, args.split)
        if arm is None:
            print(f"[skip] {name}: not found", file=sys.stderr)
            continue
        blk = probe(arm, args.tau, args.sigmas)
        results["arms"][name] = blk
        print("[%s] n_gt=%d  ceiling=%.4f (%d/%d)  never_sensed=%d"
              % (name, blk["n_gt"], blk["perception_ceiling"],
                 blk["sensed_ge_tau"], blk["n_gt"], blk["never_sensed"]))
        print("     recall  " + "  ".join("s=%.2f -> %.4f" % (s, blk["recall_at_sigma_%.2f" % s])
                                          for s in args.sigmas))
        print("     buckets " + json.dumps(blk["sensed_score_buckets"]))
        for c, v in blk["by_class"].items():
            print("       cls%d %-14s ceiling=%.4f  never=%d/%d"
                  % (int(c), v["class_name"], v["ceiling"], v["never_sensed"], v["n_gt"]))

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = EC_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[done] %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
