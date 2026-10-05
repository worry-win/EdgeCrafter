"""Analyse the P0-1 / P0-3 stage features: is the neck a *common* space?

Inputs
------
  outputs/ablation/cmp5L_stage_features/{arm}_{split}.npz   (from diag_cmp5L_stage_features.py)
  outputs/ablation/cmp5L_score_ranking_pergt_{split}.csv    (from probe_cmp5L_score_ranking.py,
                                                             used only for the group labels)

Two questions, deliberately kept apart
--------------------------------------
Q1 GLOBAL   samples = images.  CKA between arm i and arm j at the backbone stage vs
            at the neck stage.  If CKA_neck > CKA_backbone, the HybridEncoder is
            *converging* heterogeneous pretrained representations into a shared
            detection space -- the precondition for neck-level KD.

Q2 LESION   samples = ground-truth lesions, using the ROI-pooled features.  This is
            the disagreement-aware version: the same CKA is recomputed on subsets
            (easy / disagree / hard, and on the five decision cells).  The hypothesis
            worth testing is that the student-teacher feature gap is *concentrated*
            on the lesions where their decisions differ, not spread uniformly.

A control is strongly recommended: pass the 4 trained arms plus a random-init arm (if
available) so the CKA floor of "same architecture, no shared training" is visible.

Implementation notes
--------------------
* Linear CKA. Each feature dimension is z-scored first, otherwise a single
  high-variance channel dominates the Frobenius norms and every pair looks similar.
* CKA is scale/shift invariant in principle, but the features here are float16
  activations whose per-channel offsets are large compared with their variance;
  z-scoring makes the statistic stable and comparable across stages.
* Samples must be paired: the same (image) or the same (ann_id) row in both arms.
  The script asserts the GT tables of all arms are element-wise identical.

Usage
-----
    python scripts/ablation/analyze_cmp5L_stage_similarity.py \
        --feat-dir outputs/ablation/cmp5L_stage_features --split valid \
        --arms lw_xlarge dinov2b dinov2s mae_vitb ecvits_official \
        --student dinov2s \
        --per-gt outputs/ablation/cmp5L_score_ranking_pergt_valid.csv \
        --out outputs/ablation/cmp5L_stage_similarity_valid.json
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

LEVELS = (0, 1, 2)
GROUPS = ["correct", "rank_fixable", "calib_only", "feat_fixable", "hard_floor"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--student", default="dinov2s")
    ap.add_argument("--per-gt", default=None,
                    help="cmp5L_score_ranking_pergt_*.csv, for group labels")
    ap.add_argument("--hit-matrix", default=None,
                    help="cmp5L_hitmatrix5_*.csv, for the easy/disagree/hard split")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def load(feat_dir, arm, split):
    p = Path(feat_dir) / f"{arm}_{split}.npz"
    if not p.exists():
        return None
    return np.load(p, allow_pickle=True)


def zscore(X, eps=1e-6):
    X = X.astype(np.float64)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    return (X - mu) / (sd + eps)


def linear_cka(X, Y):
    """Linear CKA between [n,d1] and [n,d2] with the same n rows."""
    n = X.shape[0]
    if n < 3:
        return None
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    num = np.linalg.norm(Xc.T @ Yc, "fro") ** 2
    den = (np.linalg.norm(Xc.T @ Xc, "fro")
           * np.linalg.norm(Yc.T @ Yc, "fro"))
    if den == 0:
        return None
    return float(num / den)


def mean_cosine(A, B):
    """Mean rowwise cosine between paired rows of A and B."""
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    na = np.linalg.norm(A, axis=1)
    nb = np.linalg.norm(B, axis=1)
    ok = (na > 1e-8) & (nb > 1e-8)
    if not ok.any():
        return None
    return float((np.sum(A[ok] * B[ok], axis=1) / (na[ok] * nb[ok])).mean())


def cka_block(arms_data, stage_key, order, row_idx=None, standardize=True):
    """Return (matrix, n_rows_used) of pairwise CKA at one stage, per level.

    stage_key: 'glob_b' / 'glob_z' (samples = images) or 'roi_b' / 'roi_z' (lesions)
    """
    out = {lv: [[None] * len(order) for _ in order] for lv in LEVELS}
    cos = {lv: [[None] * len(order) for _ in order] for lv in LEVELS}
    n_used = {lv: 0 for lv in LEVELS}
    feats = {}
    for lv in LEVELS:
        rows = []
        for a in order:
            X = arms_data[a][f"{stage_key}_l{lv}"]
            if row_idx is not None:
                X = X[row_idx]
            X = zscore(X) if standardize else X.astype(np.float64)
            rows.append(X)
        if any(r.shape[0] < 3 for r in rows):
            continue
        n_used[lv] = rows[0].shape[0]
        for i in range(len(order)):
            for j in range(i, len(order)):
                cka = linear_cka(rows[i], rows[j]) if i != j else 1.0
                out[lv][i][j] = out[lv][j][i] = cka
                c = mean_cosine(rows[i], rows[j]) if i != j else 1.0
                cos[lv][i][j] = cos[lv][j][i] = c
    return out, cos, n_used


def round_matrix(M):
    return [[None if v is None else round(v, 4) for v in row] for row in M]


def mean_offdiag(M):
    vals = [M[i][j] for i in range(len(M)) for j in range(len(M))
            if i < j and M[i][j] is not None]
    return round(float(np.mean(vals)), 4) if vals else None


def main():
    args = parse_args()
    data = {}
    order = []
    for a in args.arms:
        d = load(args.feat_dir, a, args.split)
        if d is None:
            print(f"[warn] missing features for {a}, skipped")
            continue
        data[a] = d
        order.append(a)
    if not order:
        raise SystemExit("no stage features found")
    if args.student not in order:
        raise SystemExit(f"student {args.student} not among loaded arms")

    ref = data[order[0]]
    for a in order[1:]:
        if not np.array_equal(ref["gt_ann_id"], data[a]["gt_ann_id"]):
            raise SystemExit(f"GT tables differ between {order[0]} and {a}; "
                             "the arms did not see the same images in the same order")

    n_img = len(ref["image_ids"])
    n_gt = len(ref["gt_ann_id"])
    print(f"split={args.split}  arms={order}  images={n_img}  gt={n_gt}")

    # ---- group labels ------------------------------------------------------
    groups = {}
    if args.per_gt and Path(args.per_gt).exists():
        cells = defaultdict(list)
        rows = list(csv.DictReader(open(args.per_gt)))
        ann_to_row = {int(r["ann_id"]): r for r in rows}
        for k in range(n_gt):
            ann = int(ref["gt_ann_id"][k])
            r = ann_to_row.get(ann)
            if r:
                cells[r["cell"]].append(k)
        for g, idx in cells.items():
            groups[g] = np.asarray(idx, dtype=np.int64)
        print(f"  groups from {Path(args.per_gt).name}: "
              f"{ {g: len(v) for g, v in groups.items()} }")
    else:
        print("  [warn] no --per-gt given: no decision-cell conditioning (Q2 will be global only)")

    if args.hit_matrix and Path(args.hit_matrix).exists():
        rows = list(csv.DictReader(open(args.hit_matrix)))
        ann_to_hits = {int(r["ann_id"]): int(r["n_hits"]) for r in rows}
        easy, disagree, hard = [], [], []
        for k in range(n_gt):
            h = ann_to_hits.get(int(ref["gt_ann_id"][k]))
            if h is None:
                continue
            (easy if h == 5 else disagree if h >= 1 else hard).append(k)
        groups["n_hits_easy"] = np.asarray(easy, dtype=np.int64)
        groups["n_hits_disagree"] = np.asarray(disagree, dtype=np.int64)
        groups["n_hits_hard"] = np.asarray(hard, dtype=np.int64)
        print(f"  hit-matrix groups: easy={len(easy)} disagree={len(disagree)} hard={len(hard)}")

    res = {"split": args.split, "arms": order, "student": args.student,
           "n_images": n_img, "n_gt": n_gt, "views": {}}

    # ---- Q1 global ---------------------------------------------------------
    gb, gbcos, n_gb = cka_block(data, "glob_b", order)
    gz, gzcos, n_gz = cka_block(data, "glob_z", order)
    q1 = {"stage": "global (samples = images)", "backbone": {}, "neck": {}, "contrast": {}}
    for lv in LEVELS:
        q1["backbone"][f"l{lv}"] = {"cka": round_matrix(gb[lv]),
                                    "mean_offdiag": mean_offdiag(gb[lv])}
        q1["neck"][f"l{lv}"] = {"cka": round_matrix(gz[lv]),
                                "mean_offdiag": mean_offdiag(gz[lv])}
        q1["contrast"][f"l{lv}"] = (None if (q1["neck"][f"l{lv}"]["mean_offdiag"] is None
                                             or q1["backbone"][f"l{lv}"]["mean_offdiag"] is None)
                                    else round(q1["neck"][f"l{lv}"]["mean_offdiag"]
                                               - q1["backbone"][f"l{lv}"]["mean_offdiag"], 4))
    res["views"]["global"] = q1

    # ---- Q2 lesion-conditioned --------------------------------------------
    q2 = {"stage": "lesion ROI (samples = GT boxes)", "subsets": {}}
    for gname in (["all"] + list(groups.keys())):
        idx = None if gname == "all" else groups[gname]
        if idx is not None and len(idx) < 10:
            continue
        rb, rbcos, n_rb = cka_block(data, "roi_b", order, row_idx=idx)
        rz, rzcos, n_rz = cka_block(data, "roi_z", order, row_idx=idx)
        blk = {"n_rows": n_rb[0] or n_rz[0], "backbone": {}, "neck": {},
               "student_cosine": {}}
        for lv in LEVELS:
            blk["backbone"][f"l{lv}"] = {"cka": round_matrix(rb[lv]),
                                         "mean_offdiag": mean_offdiag(rb[lv])}
            blk["neck"][f"l{lv}"] = {"cka": round_matrix(rz[lv]),
                                     "mean_offdiag": mean_offdiag(rz[lv])}
            si = order.index(args.student)
            blk["student_cosine"][f"l{lv}"] = {
                t: (None if rzcos[lv][si][order.index(t)] is None
                    else round(rzcos[lv][si][order.index(t)], 4))
                for t in order if t != args.student}
        q2["subsets"][gname] = blk
    res["views"]["lesion"] = q2

    with open(args.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    # ---- console summary ---------------------------------------------------
    print("\n[Q1] global CKA (mean off-diagonal, samples = images)")
    print("  stage      " + "  ".join(f"l{lv}" for lv in LEVELS))
    print("  backbone   " + "  ".join(
        f"{q1['backbone'][f'l{lv}']['mean_offdiag']}" for lv in LEVELS))
    print("  neck       " + "  ".join(
        f"{q1['neck'][f'l{lv}']['mean_offdiag']}" for lv in LEVELS))
    print("  contrast   " + "  ".join(
        f"{q1['contrast'][f'l{lv}']}" for lv in LEVELS))

    print("\n[Q2] lesion ROI CKA (mean off-diagonal) by subset")
    hdr = "  subset                 n    " + "  ".join(f"bb_l{lv}" for lv in LEVELS) \
          + "  " + "  ".join(f"nk_l{lv}" for lv in LEVELS)
    print(hdr)
    for gname, blk in q2["subsets"].items():
        bb = "  ".join(f"{blk['backbone'][f'l{lv}']['mean_offdiag']}" for lv in LEVELS)
        nk = "  ".join(f"{blk['neck'][f'l{lv}']['mean_offdiag']}" for lv in LEVELS)
        print(f"  {gname:20s} {blk['n_rows']:5d}  {bb}  {nk}")

    print(f"\n  student={args.student} mean ROI cosine to teachers "
          f"(neck, l0..l2):")
    for gname, blk in q2["subsets"].items():
        print(f"    {gname:20s} " + "  ".join(
            f"l{lv}:{blk['student_cosine'][f'l{lv}']}" for lv in LEVELS))
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
