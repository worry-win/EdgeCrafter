"""Zero-cost probe: is the cross-arm teacher signal actually useful for logit KD?

Question (Go/No-Go for path-A logit distillation):
  On the student's rank_fixable lesions, do teachers carry a *more correct* class
  signal than the student — or are they just "higher score, equally wrong class"?

The whole point of logit KD is: teacher's GT-class logit should be higher than the
student's on lesions the student mis-scores. If teachers are merely uniformly
over-confident (higher score on the SAME wrong class), logit KD is doomed.

We measure, on each lesion, for the query that best localises it:
  - student GT-class logit / margin  (from rec_logit)
  - teacher GT-class logit / margin
  - teacher argmax class == GT class?  (does the teacher actually know the class)

Layers of answer:
  Q1  On rank_fixable, is teacher GT-class logit > student GT-class logit?
  Q2  On rank_fixable, what fraction of teachers have argmax == GT class (vs student)?
  Q3  Gate reliability: among lesions where a teacher is "confident" (score>=sigma_T),
      what fraction is the teacher's argmax actually the GT class?  (this bounds how
      good a class-margin gate can be)

Runs on the cluster against cmp5L_predictions/*.npz. No GPU needed.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

STUDENT = "dinov2s"
TEACHERS = ["dinov2b", "lw_xlarge", "mae_vitb", "ecvits_official"]


def load_arm(pred_dir, name, split):
    d = np.load(Path(pred_dir) / f"{name}_{split}.npz", allow_pickle=True)
    # ann_id -> index into gt_* arrays
    ann_id_to_idx = {int(a): i for i, a in enumerate(d["gt_ann_id"])}
    return {
        "rec_ptr": d["rec_ptr"],
        "image_ids": d["image_ids"],
        "rec_score": d["rec_score"],
        "rec_label": d["rec_label"],
        "rec_logit": d["rec_logit"],
        # class-agnostic best-GT matching (use these to pick the "localising" query)
        "rec_gt_any_id": d["rec_gt_any_id"],
        "rec_gt_any_iou": d["rec_gt_any_iou"],
        "rec_gt_cls_id": d["rec_gt_cls_id"],
        "rec_gt_cls_iou": d["rec_gt_cls_iou"],
        "gt_cls": d["gt_cls"],
        "gt_ann_id": d["gt_ann_id"],
        "ann_id_to_idx": ann_id_to_idx,
    }


def per_lesion_best(arm, tau):
    """For each lesion ann_id, find the query with best class-AGNOSTIC IoU (the one
    that actually localises it), then record that query's score, GT-class logit,
    margin, and argmax class.  This does NOT bias argmax_is_gt by construction."""
    rec_ptr = arm["rec_ptr"]
    any_id = arm["rec_gt_any_id"]
    any_iou = arm["rec_gt_any_iou"]
    logit = arm["rec_logit"]
    score = arm["rec_score"]
    label = arm["rec_label"]
    gt_cls = arm["gt_cls"]
    ann_id_to_idx = arm["ann_id_to_idx"]
    n_img = len(arm["image_ids"])

    out = {}  # ann_id -> dict (best class-agnostic IoU query)
    for i in range(n_img):
        r0, r1 = int(rec_ptr[i]), int(rec_ptr[i + 1])
        if r1 <= r0:
            continue
        ids = any_id[r0:r1]
        ious = any_iou[r0:r1]
        for t in np.where(ids >= 0)[0]:
            ann = int(ids[t])
            gi = ann_id_to_idx[ann]
            gt = int(gt_cls[gi])
            lg = logit[r0 + t].astype(np.float64)
            gt_logit = lg[gt]
            other = np.delete(lg, gt)
            margin = gt_logit - (other.max() if other.size else 0.0)
            cand = {
                "score": float(score[r0 + t]),
                "iou": float(ious[t]),
                "gt_logit": float(gt_logit),
                "margin": float(margin),
                "argmax": int(label[r0 + t]),
                "argmax_is_gt": int(label[r0 + t]) == gt,
            }
            # keep best-IoU query for this lesion (the "localising" query)
            if ann not in out or cand["iou"] > out[ann]["iou"]:
                out[ann] = cand
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--sigma-student", type=float, default=0.5)
    ap.add_argument("--sigma-teacher", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    s = load_arm(args.pred_dir, STUDENT, args.split)
    teachers = {t: load_arm(args.pred_dir, t, args.split) for t in TEACHERS}

    s_best = per_lesion_best(s, args.tau)
    t_best = {t: per_lesion_best(arm, args.tau) for t, arm in teachers.items()}

    # student per-lesion localisation flag (best GT-cls IoU >= tau)
    s_loc = {ann: (v["iou"] >= args.tau) for ann, v in s_best.items()}
    s_score = {ann: v["score"] for ann, v in s_best.items()}
    s_ann_to_idx = s["ann_id_to_idx"]
    gt_cls = {ann: int(s["gt_cls"][s_ann_to_idx[ann]]) for ann in s_best}

    # teacher per-lesion best score among localising queries
    t_score = {}
    t_gtlogit = {}
    t_margin = {}
    t_argmax_is_gt = {}
    for t, arm in teachers.items():
        for ann, v in t_best[t].items():
            if v["iou"] >= args.tau:
                if ann not in t_score or v["score"] > t_score[ann]:
                    t_score[ann] = v["score"]
                    t_gtlogit[ann] = v["gt_logit"]
                    t_margin[ann] = v["margin"]
                    t_argmax_is_gt[ann] = v["argmax_is_gt"]

    # classify lesions (reproduce score_ranking cells)
    cells = defaultdict(list)
    for ann in s_best:
        loc_s = s_loc.get(ann, False)
        sc_s = s_score.get(ann, -1.0)
        t_best_score = max((t_score.get(ann, -1.0) for t in TEACHERS), default=-1.0)
        if loc_s and sc_s >= args.sigma_student:
            cell = "correct"
        elif loc_s and sc_s < args.sigma_student and t_best_score >= args.sigma_teacher:
            cell = "rank_fixable"
        elif loc_s and sc_s < args.sigma_student and t_best_score < args.sigma_teacher:
            cell = "calib_only"
        elif not loc_s and t_best_score >= args.sigma_teacher:
            cell = "feat_fixable"
        else:
            cell = "hard_floor"
        cells[cell].append(ann)

    def stat(anns, key_fn):
        vals = [key_fn(ann) for ann in anns if key_fn(ann) is not None]
        return vals

    # Q1/Q2: on rank_fixable, teacher GT-class logit vs student, and argmax==GT rate
    rf = cells["rank_fixable"]
    s_gtlogit_rf = stat(rf, lambda a: s_best[a]["gt_logit"])
    s_margin_rf = stat(rf, lambda a: s_best[a]["margin"])
    t_gtlogit_rf = stat(rf, lambda a: t_gtlogit.get(a))
    t_margin_rf = stat(rf, lambda a: t_margin.get(a))
    t_amgt_rf = stat(rf, lambda a: t_argmax_is_gt.get(a))
    s_amgt_rf = stat(rf, lambda a: s_best[a]["argmax_is_gt"])

    # Q3: gate reliability across ALL lesions where teacher score >= sigma_T
    confident = [a for a in s_best if t_score.get(a, -1.0) >= args.sigma_teacher]
    t_amgt_conf = stat(confident, lambda a: t_argmax_is_gt.get(a))

    result = {
        "split": args.split,
        "student": STUDENT,
        "teachers": TEACHERS,
        "tau": args.tau,
        "sigma_student": args.sigma_student,
        "sigma_teacher": args.sigma_teacher,
        "cell_counts": {k: len(v) for k, v in cells.items()},
        "rank_fixable": {
            "n": len(rf),
            "student_gtlogit_mean": float(np.mean(s_gtlogit_rf)) if s_gtlogit_rf else None,
            "student_margin_mean": float(np.mean(s_margin_rf)) if s_margin_rf else None,
            "teacher_gtlogit_mean": float(np.mean(t_gtlogit_rf)) if t_gtlogit_rf else None,
            "teacher_margin_mean": float(np.mean(t_margin_rf)) if t_margin_rf else None,
            "teacher_gtlogit_minus_student_mean": float(
                np.mean([a - b for a, b in zip(t_gtlogit_rf, s_gtlogit_rf)])
            ) if t_gtlogit_rf and s_gtlogit_rf else None,
            "student_argmax_is_gt_rate": float(np.mean(s_amgt_rf)) if s_amgt_rf else None,
            "teacher_argmax_is_gt_rate": float(np.mean(t_amgt_rf)) if t_amgt_rf else None,
        },
        "gate_reliability": {
            "n_confident_lesions": len(confident),
            "teacher_argmax_is_gt_rate": float(np.mean(t_amgt_conf)) if t_amgt_conf else None,
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
