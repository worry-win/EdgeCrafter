"""P0-4: score / ranking disagreement probe over the low-threshold cmp5L dumps.

Reads the EXISTING per-query dumps only -- no GPU, no re-inference. Two questions:

  Q1  Is the student's miss a *perception* miss or a *scoring/ranking* miss?
      Compared against: can any teacher supply the missing decision?

  Q2  Which teacher actually supplies it?  (teacher selection by complementarity,
      not by standalone AP)

Per ground truth g, with an oracle-free criterion that needs no greedy matching:

  loc_a(g)   = arm a has a query with rec_gt_cls_id == g and rec_gt_cls_iou >= tau
               (an ORACLE locator bound: a single query overlaps the right lesion)
  s_a(g)     = max rec_score over exactly those queries   (arm a's confidence about
               the best localisation it actually has for g; 0.0 if it has none)

Then every GT of the split falls into exactly one of five cells:

  correct        loc_S and s_S >= sigma_S                  already right, no KD value
  rank_fixable   loc_S and s_S <  sigma_S and t_best>=sigma_T
                 -> student LOCALISES it but ranks it low, and a teacher is
                    confident about the same lesion.  Pure ranking / logit KD.
  calib_only     loc_S and s_S <  sigma_S and t_best< sigma_T
                 -> student localises it, no teacher is confident either;
                    teacher KD CANNOT help, this is self-calibration territory.
  feat_fixable   not loc_S and t_best >= sigma_T
                 -> student has no query on it at all, a teacher does.
                    Feature / localisation-level KD territory.
  hard_floor     not loc_S and t_best <  sigma_T            nobody can do it.

Also emits a calibration curve  P(localisation correct | score bin)  per arm and
per class: this is what tells "high-precision arm" apart from "well calibrated arm".

Usage
-----
    python scripts/ablation/probe_cmp5L_score_ranking.py \
        --pred-dir outputs/ablation/cmp5L_predictions --split test \
        --student dinov2s --teachers lw_xlarge dinov2b mae_vitb ecvits_official \
        --tau 0.5 --sigma-student 0.05 --sigma-teacher 0.5 \
        --out outputs/ablation/cmp5L_score_ranking_test.json \
        --dump-per-gt outputs/ablation/cmp5L_score_ranking_pergt_test.csv
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CLASS_NAMES = {0: "实(囊实)性", 1: "囊性", 2: "淋巴结", 3: "导管相关病变"}

# COCO area buckets on the GT box area (px^2) -- same cuts as the other scripts.
SMALL_MAX = 32.0 ** 2
MEDIUM_MAX = 96.0 ** 2


def scale_of(area):
    if area < SMALL_MAX:
        return "small"
    if area < MEDIUM_MAX:
        return "medium"
    return "large"


SCORE_BINS = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.25), (0.25, 0.50),
              (0.50, 0.75), (0.75, 1.01)]


def load_arm(pred_dir, name, split):
    p = Path(pred_dir) / f"{name}_{split}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return {
        "path": str(p),
        "rec_ptr": d["rec_ptr"],
        "score": d["rec_score"].astype(np.float64),
        "cls_id": d["rec_gt_cls_id"].astype(np.int64),
        "cls_iou": d["rec_gt_cls_iou"].astype(np.float64),
        "gt_ptr": d["gt_ptr"],
        "gt_ann": d["gt_ann_id"].astype(np.int64),
        "gt_cls": d["gt_cls"].astype(np.int64),
        "gt_area": d["gt_area"].astype(np.float64),
        "image_ids": d["image_ids"],
    }


def canonical_gt(arms):
    """The (ann_id, cls, area) table of the split. Verified identical across arms."""
    ref = arms[0]
    out = []
    for k in range(len(ref["gt_ann"])):
        out.append((int(ref["gt_ann"][k]), int(ref["gt_cls"][k]), float(ref["gt_area"][k])))
    return out


def per_gt_confidence(arm, tau):
    """ann_id -> (localised, best_score_among_localising_queries, top1_iou).

    ``top1_iou`` is the IoU of the arm's HIGHEST-SCORING query that is matched to
    this lesion (same class). It separates two very different situations that the
    plain ``localised`` flag conflates:

        localised and top1_iou >= tau   the arm's own ranking already puts the
                                        correct box first -- only its *score* is low
        localised and top1_iou <  tau   a correct box exists but the arm ranks
                                        another box above it for this lesion
                                        (a genuine ranking/competition failure)

    NOTE: ``rec_gt_cls_id`` holds the COCO ``annotation id`` VALUE directly
    (range -1..2405 on this dataset), NOT an index into the per-image GT slice.
    Verified empirically: on image 0 the GT slice is gt_ann[0:1] == [1] and the
    recorded ids are exactly {1}; the ids also exceed the slice indices.
    """
    rec_ptr, cls_id, cls_iou, score = arm["rec_ptr"], arm["cls_id"], arm["cls_iou"], arm["score"]
    n_img = len(arm["image_ids"])

    loc = {}
    best = {}          # max score among queries that DO localise this lesion
    top1 = {}          # IoU of the highest-scoring query matched to this lesion (any box)
    for i in range(n_img):
        r0, r1 = int(rec_ptr[i]), int(rec_ptr[i + 1])
        if r1 <= r0:
            continue
        ids = cls_id[r0:r1]
        ious = cls_iou[r0:r1]
        sc = score[r0:r1]
        touched = np.where(ids >= 0)[0]
        for t in touched:
            ann = int(ids[t])
            # dimension 2: which box does the arm actually rank first for this lesion
            if sc[t] > top1.get(ann, (-1.0, -1.0))[0]:
                top1[ann] = (float(sc[t]), float(ious[t]))
            # dimension 1: how confident is the arm about a CORRECT localisation
            if ious[t] >= tau:
                loc[ann] = True
                if sc[t] > best.get(ann, -1.0):
                    best[ann] = float(sc[t])
    top1_iou = {k: v[1] for k, v in top1.items()}
    return loc, best, top1_iou


def calibration_curve(arm, ann_to_cls, tau):
    """P(localisation correct | score bin) over all queries, overall and per class."""
    rec_ptr, cls_id, cls_iou, score = arm["rec_ptr"], arm["cls_id"], arm["cls_iou"], arm["score"]
    n_img = len(arm["image_ids"])

    overall = [[0, 0] for _ in SCORE_BINS]           # [n, n_correct_loc]
    per_cls = {c: [[0, 0] for _ in SCORE_BINS] for c in CLASS_NAMES}
    for i in range(n_img):
        r0, r1 = int(rec_ptr[i]), int(rec_ptr[i + 1])
        if r1 <= r0:
            continue
        sc = score[r0:r1]
        ids = cls_id[r0:r1]
        ious = cls_iou[r0:r1]
        ok = (ids >= 0) & (ious >= tau)
        for b, (lo, hi) in enumerate(SCORE_BINS):
            m = (sc >= lo) & (sc < hi)
            n = int(m.sum())
            if n == 0:
                continue
            overall[b][0] += n
            overall[b][1] += int((m & ok).sum())
            # per class: denominator = every query that touches a same-class GT
            touched = np.where(m & (ids >= 0))[0]
            for t in touched:
                c = ann_to_cls.get(int(ids[t]))
                if c is None:
                    continue
                per_cls[c][b][0] += 1
                if ok[t]:
                    per_cls[c][b][1] += 1

    def fmt(rows):
        out = []
        for (lo, hi), (n, k) in zip(SCORE_BINS, rows):
            out.append({
                "bin": f"[{lo:.2f},{hi:.2f})",
                "n_pred": n,
                "n_correct_loc": k,
                "p_correct_loc": round(k / n, 4) if n else None,
            })
        return out

    return {"overall": fmt(overall),
            "by_class": {CLASS_NAMES[c]: fmt(per_cls[c]) for c in CLASS_NAMES}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--student", default="dinov2s")
    ap.add_argument("--teachers", nargs="+",
                    default=["lw_xlarge", "dinov2b", "mae_vitb", "ecvits_official"])
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--sigma-student", type=float, default=0.05)
    ap.add_argument("--sigma-teacher", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-per-gt", default=None)
    args = ap.parse_args()

    names = [args.student] + list(args.teachers)
    arms = {}
    for n in names:
        a = load_arm(args.pred_dir, n, args.split)
        if a is None:
            print(f"[warn] missing dump for {n} on {args.split}, skipped")
            continue
        arms[n] = a
    if args.student not in arms:
        raise SystemExit(f"student dump {args.student}_{args.split}.npz not found")

    order = [n for n in names if n in arms]
    truth = canonical_gt([arms[n] for n in order])
    ann_to_cls = {a: c for a, c, _ in truth}
    ann_to_area = {a: ar for a, _, ar in truth}

    conf = {}
    for n in order:
        loc, best, top1 = per_gt_confidence(arms[n], args.tau)
        conf[n] = {"loc": loc, "best": best, "top1": top1}

    sS = args.sigma_student
    sT = args.sigma_teacher
    stu = conf[args.student]
    teachers = [n for n in order if n != args.student]

    cells = Counter()
    cell_cls = defaultdict(Counter)
    cell_scale = defaultdict(Counter)
    cell_teacher = defaultdict(Counter)      # which teacher supplies the cell
    teacher_supply = defaultdict(Counter)    # teacher -> Counter(cell)
    teacher_unique = Counter()               # teacher -> #GT where it is the ONLY supplier
    teacher_ranked = Counter()               # teacher -> #GT where it attains t_best
    rank_split = Counter()                   # rank_fixable split by student's own top-1

    rows = []
    for ann, cls, area in truth:
        loc_s = bool(stu["loc"].get(ann, False))
        s_s = float(stu["best"].get(ann, 0.0))
        t1_s = float(stu["top1"].get(ann, -1.0))

        t_best = -1.0
        t_best_arm = ""
        suppliers = []
        for tn in teachers:
            tv = float(conf[tn]["best"].get(ann, 0.0))
            if tv >= sT:
                suppliers.append(tn)
                teacher_supply[tn]["available"] += 1
            if tv > t_best:
                t_best = tv
                t_best_arm = tn
        if t_best_arm:
            teacher_ranked[t_best_arm] += 1
        if len(suppliers) == 1:
            teacher_unique[suppliers[0]] += 1

        if loc_s and s_s >= sS:
            cell = "correct"
        elif loc_s and s_s < sS and t_best >= sT:
            cell = "rank_fixable"
        elif loc_s and s_s < sS:
            cell = "calib_only"
        elif t_best >= sT:
            cell = "feat_fixable"
        else:
            cell = "hard_floor"

        if cell == "rank_fixable":
            rank_split["own_top1_correct" if t1_s >= args.tau else "own_top1_wrong"] += 1

        cells[cell] += 1
        cell_cls[cell][CLASS_NAMES[cls]] += 1
        cell_scale[cell][scale_of(area)] += 1
        for tn in suppliers:
            cell_teacher[cell][tn] += 1

        rows.append({
            "ann_id": ann, "gt_class": CLASS_NAMES[cls], "gt_area": round(area, 1),
            "gt_scale": scale_of(area),
            "student_localised": int(loc_s), "student_score": round(s_s, 4),
            "student_top1_iou": round(t1_s, 4) if t1_s >= 0 else None,
            "teacher_best_score": round(t_best, 4), "teacher_best_arm": t_best_arm,
            "n_teacher_available": len(suppliers), "cell": cell,
        })

    n_gt = len(truth)
    res = {
        "split": args.split, "tau": args.tau,
        "sigma_student": sS, "sigma_teacher": sT,
        "student": args.student, "teachers": teachers,
        "n_gt": n_gt,
        "cells": {k: {"n": v, "frac": round(v / n_gt, 4),
                      "by_class": dict(cell_cls[k]),
                      "by_scale": dict(cell_scale[k]),
                      "teacher_available_by": dict(cell_teacher[k])}
                  for k, v in cells.items()},
        "teacher_unique_supply": dict(teacher_unique),
        "teacher_argmax_supply": dict(teacher_ranked),
        "teacher_available_total": {k: dict(v) for k, v in teacher_supply.items()},
        "rank_fixable_split_by_student_top1": dict(rank_split),
        "calibration": {n: calibration_curve(arms[n], ann_to_cls, args.tau) for n in order},
    }

    with open(args.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    if args.dump_per_gt:
        with open(args.dump_per_gt, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"split={args.split}  student={args.student}  n_gt={n_gt}  "
          f"tau={args.tau} sigma_S={sS} sigma_T={sT}")
    for k in ["correct", "rank_fixable", "calib_only", "feat_fixable", "hard_floor"]:
        v = res["cells"].get(k)
        if v:
            print(f"  {k:14s} {v['n']:5d} ({v['frac'] * 100:5.2f}%)  "
                  f"cls={v['by_class']}")
            if v["teacher_available_by"]:
                print(f"                 teacher={v['teacher_available_by']}")
    print(f"  teacher unique-supply (only confident teacher for a lesion): "
          f"{dict(teacher_unique)}")
    print(f"  teacher argmax-supply: {dict(teacher_ranked)}")
    print(f"  rank_fixable split by the student's own top-1 box: {dict(rank_split)}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
