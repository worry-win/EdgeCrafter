"""Read the internal-routing dumps and decide: routing-side or decision-side?

The single sharpest test needs no teacher at all.  Within ONE arm, compare the
sampling/attention behaviour of its responsible query on the lesions it scores
CONFIDENTLY ('correct') against the lesions it localises but under-scores
('rank_fixable').  Both groups were handled by the same weights, so the only
difference is the lesion:

  same geometry, much lower score   -> decision-side: the query looks at the
                                       lesion just as well and is not convinced.
                                       Fixing this needs attention/logit
                                       calibration, NOT feature-space KD.
  clearly worse geometry            -> routing-side: the query looks elsewhere.
                                       This one a sampling-point / trajectory
                                       distillation loss could actually address.

A teacher comparison is reported alongside as an external confirmation, but the
within-arm contrast is what separates the two mechanisms without confounding.

Usage
-----
    python scripts/ablation/analyze_cmp5L_internal_routing.py \
        --route-dir outputs/ablation/cmp5L_internal_routing --split test \
        --lesions outputs/ablation/cmp5L_internal_lesions_test.csv \
        --arms dinov2s dinov2b lw_xlarge mae_vitb ecvits_official \
        --student dinov2s --out outputs/ablation/cmp5L_internal_routing_analysis_test.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

EC_ROOT = Path(__file__).resolve().parents[2]

CELLS = ["correct", "rank_fixable", "calib_only", "feat_fixable", "hard_floor"]
HIST_BINS = 6
HIST_EXTENT = 1.5          # in units of the lesion-box diagonal, from the centre
EPS = 1e-9


# --------------------------------------------------------------------------- #
def chisq_phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def sign_test(k, n):
    """Two-sided sign test, normal approximation. k = #(teacher > student)."""
    if n <= 0:
        return 1.0
    if n < 12:
        # exact via binomial tail
        p = 0.0
        for i in range(0, n + 1):
            if abs(i - n / 2) >= abs(k - n / 2) - 1e-9:
                p += math.comb(n, i)
        return min(1.0, p / (2.0 ** n))
    z = (k - n / 2.0) / math.sqrt(n / 4.0)
    return max(0.0, min(1.0, 2.0 * (1.0 - chisq_phi(abs(z)))))


def paired(a, b, mask):
    """teacher-side a vs student-side b on the masked subset."""
    a, b = a[mask].astype(np.float64), b[mask].astype(np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = a.size
    if n == 0:
        return dict(n=0, mean_teacher=None, mean_student=None, delta=None,
                    d=None, frac_teacher_gt=None, p_sign=None)
    d = a - b
    sd = b.std(ddof=0)
    # pooled sd for Cohen's d (paired: use sd of the differences)
    ds = d.std(ddof=0)
    return dict(
        n=int(n),
        mean_teacher=float(a.mean()),
        mean_student=float(b.mean()),
        delta=float(d.mean()),
        d=float(d.mean() / ds) if ds > EPS else None,
        frac_teacher_gt=float((d > 0).mean()),
        p_sign=float(sign_test(int((d > 0).sum()), int((d != 0).sum()))),
        student_sd=float(sd),
    )


def chamfer_norm(pts_a, pts_b, diag):
    """Symmetric mean nearest-neighbour distance, in units of the box diagonal.
    Inputs are float16 in the dump, so promote BEFORE squaring or the pair-wise
    differences overflow (pixels^2 exceeds float16's range)."""
    a = pts_a.reshape(-1, 2)
    b = pts_b.reshape(-1, 2)
    a = a[np.isfinite(a).all(1)].astype(np.float64)
    b = b[np.isfinite(b).all(1)].astype(np.float64)
    if a.size == 0 or b.size == 0:
        return np.nan
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return float((np.sqrt(d2.min(1)).mean() + np.sqrt(d2.min(0)).mean()) / 2.0 / diag)


def attn_hist(loc, w, ctr, diag):
    """Box-relative attention-mass histogram: where does the mass sit relative to
    the lesion? Bins over (p - centre)/diag in a [-HIST_EXTENT, HIST_EXTENT]^2
    window, weighted by attention. Head/point/level indices never enter, which is
    the point: head ordering is not comparable across models."""
    r = (loc.reshape(-1, 2) - ctr) / diag
    ww = w.reshape(-1).astype(np.float64)
    r = np.clip(r, -HIST_EXTENT + 1e-6, HIST_EXTENT - 1e-6)
    idx = ((r + HIST_EXTENT) / (2 * HIST_EXTENT) * HIST_BINS).astype(int)
    idx = np.clip(idx, 0, HIST_BINS - 1)
    h = np.zeros((HIST_BINS, HIST_BINS), np.float64)
    np.add.at(h, (idx[:, 0], idx[:, 1]), ww)
    s = h.sum()
    return h / s if s > EPS else h


def kl_pq(p, q):
    p = np.clip(p, 1e-8, None)
    q = np.clip(q, 1e-8, None)
    p = p / p.sum()
    q = q / q.sum()
    return float((p * np.log(p / q)).sum())


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--student", default="dinov2s")
    ap.add_argument("--lesions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--routes", nargs="*", default=None,
                    help="subset of arms used for the teacher comparison "
                         "(default: all non-student arms)")
    args = ap.parse_args()

    rd = Path(args.route_dir)
    if not rd.is_absolute():
        rd = EC_ROOT / rd
    import csv as _csv
    rows = list(_csv.DictReader(open(args.lesions, encoding="utf-8")))
    box = {int(r["ann_id"]): (float(r["x0"]), float(r["y0"]),
                              float(r["x1"]), float(r["y1"])) for r in rows}
    cell = {int(r["ann_id"]): r["cell"] for r in rows}
    cls = {int(r["ann_id"]): r["cls_name"] for r in rows}
    teacher_hint = {int(r["ann_id"]): r["teacher_best_arm"] for r in rows}

    arms, data = [], {}
    ref_ann = None
    for a in args.arms:
        p = rd / f"{a}_{args.split}.npz"
        if not p.exists():
            print(f"[skip] {a}: {p} not found")
            continue
        z = np.load(p, allow_pickle=True)
        if ref_ann is None:
            ref_ann = z["lesion_ann_id"]
        elif not np.array_equal(ref_ann, z["lesion_ann_id"]):
            raise SystemExit(f"{a}: lesion index mismatch -- arms disagree on order")
        arms.append(a)
        data[a] = z
    if args.student not in arms:
        raise SystemExit(f"student {args.student} not among loaded arms: {arms}")
    n = len(ref_ann)
    ann = [int(v) for v in ref_ann]
    cells = np.asarray([cell[v] for v in ann], dtype=object)
    classes = np.asarray([cls[v] for v in ann], dtype=object)
    L = data[args.student]["samp_inbox"].shape[1]
    teachers = args.routes or [a for a in arms if a != args.student]

    # ---- derived per-arm, per-lesion aggregates ---------------------------- #
    def build(a):
        z = data[a]
        found = z["found"].astype(bool)
        geom = np.where(found, z["samp_inbox"].mean(1), np.nan)
        mass = np.where(found, z["samp_mass_inbox"].mean(1) / 8.0, np.nan)
        ent = np.where(found, z["attn_entropy"].mean(1), np.nan)
        dist = np.where(found, z["samp_dist"].mean(1), np.nan)
        iou0 = np.where(found, z["iou_ref"][:, 0], np.nan)
        iouE = np.where(found, z["iou_final"], np.nan)
        lg = z["logit_c"]
        lg0 = np.where(found, lg[:, 0], np.nan)
        lgE = np.where(found, lg[:, L - 1], np.nan)
        lgmin = np.where(found, lg.min(1), np.nan)
        # fields added in the second pass; tolerate their absence
        has_q = "any_inbox_q_iou" in z.files
        return dict(
            found=found, geom=geom, mass=mass, ent=ent, dist=dist,
            iou0=iou0, iouE=iouE, iou_gain=iouE - iou0,
            logit0=lg0, logitE=lgE, logit_min=lgmin, logit_recover=lgE - lg0,
            logit=lg, score=np.where(found, z["score_q"], np.nan),
            any_geom=z["any_inbox_max"].mean(1),
            any_geom_min=z["any_inbox_max"].min(1),
            any_q_score=z["any_inbox_q_score"].mean(1),
            any_q_iou=(z["any_inbox_q_iou"].mean(1) if has_q else None),
            max_iou_all=(z["max_iou_all"].mean(1) if has_q else None),
            max_iou_cls=(z["max_iou_cls"].mean(1) if has_q else None),
            ctl_rand=z["ctl_rand_inbox"].mean(1),
            ctl_top1=z["ctl_top1_inbox"].mean(1),
            level_geom=z["samp_inbox_lvl"].mean(1),
        )

    M = {a: build(a) for a in arms}
    S = M[args.student]
    out = dict(student=args.student, split=args.split, arms=arms,
               n_lesions=n, layers=L,
               cell_counts={c: int((cells == c).sum()) for c in CELLS},
               found_rate={a: float(M[a]["found"].mean()) for a in arms})

    # ---- 0. control validation -------------------------------------------- #
    ctl = {}
    for a in arms:
        m = M[a]["found"]
        if m.sum() == 0:
            continue
        ctl[a] = dict(
            n=int(m.sum()),
            resp_geom=float(np.nanmean(M[a]["geom"][m])),
            rand_geom=float(np.nanmean(M[a]["ctl_rand"][m])),
            top1_geom=float(np.nanmean(M[a]["ctl_top1"][m])),
            resp_minus_rand=float(np.nanmean(M[a]["geom"][m]) -
                                 np.nanmean(M[a]["ctl_rand"][m])),
        )
    out["control_validation"] = ctl

    # ---- 1. headline: within-arm contrast, correct vs rank_fixable -------- #
    headline = {}
    for a in arms:
        d = {}
        for c in ["correct", "rank_fixable", "calib_only"]:
            m = (cells == c) & M[a]["found"]
            if m.sum() == 0:
                d[c] = dict(n=0)
                continue
            d[c] = dict(
                n=int(m.sum()),
                score=float(np.nanmean(M[a]["score"][m])),
                geom=float(np.nanmean(M[a]["geom"][m])),
                mass=float(np.nanmean(M[a]["mass"][m])),
                entropy=float(np.nanmean(M[a]["ent"][m])),
                iou_last=float(np.nanmean(M[a]["iouE"][m])),
                iou_gain=float(np.nanmean(M[a]["iou_gain"][m])),
                logit_last=float(np.nanmean(M[a]["logitE"][m])),
                logit_min=float(np.nanmean(M[a]["logit_min"][m])),
            )
        if "correct" in d and "rank_fixable" in d and d["rank_fixable"]["n"] > 0:
            d["rankfix_minus_correct"] = {
                k: (d["rank_fixable"][k] - d["correct"][k])
                for k in ("score", "geom", "mass", "entropy", "iou_last",
                          "iou_gain", "logit_last", "logit_min")
                if d.get("correct", {}).get(k) is not None
                and d["rank_fixable"].get(k) is not None}
        headline[a] = d
    out["within_arm_contrast"] = headline

    # ---- 2. student vs teacher, paired ------------------------------------- #
    def teacher_best_metric(key, reduce="max"):
        vals = np.stack([M[t][key] for t in teachers], 1)      # [n, n_teachers]
        return np.nanmax(vals, 1) if reduce == "max" else np.nanmean(vals, 1)

    cmp_metrics = {}
    for key in ("geom", "mass", "ent", "iouE", "iou_gain", "logitE", "dist"):
        tbest = teacher_best_metric(key)
        per_cell = {}
        for c in CELLS:
            m = (cells == c)
            per_cell[c] = paired(tbest, S[key], m)
        cmp_metrics[key] = per_cell
    out["student_vs_best_teacher"] = cmp_metrics

    # ---- 3. dichotomy classification on rank_fixable ------------------------ #
    tgeom = teacher_best_metric("geom")
    tmass = teacher_best_metric("mass")
    m = (cells == "rank_fixable") & S["found"] & np.isfinite(tgeom) & np.isfinite(S["geom"])
    dg = tgeom - S["geom"]
    dm = tmass - S["mass"]
    routing = m & (dg > 0.15)
    decision = m & (np.abs(dg) <= 0.15) & (dm > 0.05)
    ambiguous = m & ~routing & ~decision
    out["dichotomy_rank_fixable"] = dict(
        n_considered=int(m.sum()),
        routing_side=int(routing.sum()),
        decision_side=int(decision.sum()),
        ambiguous=int(ambiguous.sum()),
        mean_delta_geom_considered=float(np.nanmean(dg[m])) if m.sum() else None,
        mean_delta_mass_considered=float(np.nanmean(dm[m])) if m.sum() else None,
        # the same classification restricted to the student's own geometry vs its
        # own confident group (no teacher involved)
        student_geom_on_rankfix=float(np.nanmean(S["geom"][m])) if m.sum() else None,
        student_geom_on_correct=float(np.nanmean(S["geom"][cells == "correct"]))
        if (cells == "correct").sum() else None,
    )

    # ---- 4. hard_floor / feat_fixable: does ANY query look at it? ---------- #
    blind = {}
    for c in ("hard_floor", "feat_fixable", "calib_only"):
        cc = cells == c
        if cc.sum() == 0:
            continue
        per_arm = {}
        for a in arms:
            ag = M[a]["any_geom"][cc]           # best-covering query's in-box frac
            e = dict(
                found=int(M[a]["found"][cc].sum()),
                any_geom_mean=float(np.nanmean(ag)),
                frac_any_geom_gt_0_3=float((ag > 0.3).mean()),
                any_q_score_mean=float(np.nanmean(M[a]["any_q_score"][cc])),
            )
            if M[a].get("any_q_iou") is not None:
                e["any_q_iou_mean"] = float(np.nanmean(M[a]["any_q_iou"][cc]))
                e["max_iou_all_mean"] = float(np.nanmean(M[a]["max_iou_all"][cc]))
                e["max_iou_cls_mean"] = float(np.nanmean(M[a]["max_iou_cls"][cc]))
                e["frac_any_q_iou_ge_0_5"] = float((M[a]["any_q_iou"][cc] >= 0.5).mean())
                e["frac_max_iou_all_ge_0_5"] = float((M[a]["max_iou_all"][cc] >= 0.5).mean())
            per_arm[a] = e
        union_any = np.nanmax(np.stack([M[a]["any_geom"][cc] for a in arms], 1), 1)
        blk = dict(
            n=int(cc.sum()), per_arm=per_arm,
            union_any_geom_mean=float(np.nanmean(union_any)),
            frac_union_gt_0_3=float((union_any > 0.3).mean()),
            frac_union_lt_0_05=float((union_any < 0.05).mean()),
        )
        if M[args.student].get("max_iou_all") is not None:
            u_all = np.nanmax(np.stack([M[a]["max_iou_all"][cc] for a in arms], 1), 1)
            u_cls = np.nanmax(np.stack([M[a]["max_iou_cls"][cc] for a in arms], 1), 1)
            blk.update(
                union_max_iou_all_mean=float(np.nanmean(u_all)),
                union_max_iou_cls_mean=float(np.nanmean(u_cls)),
                frac_union_max_iou_all_ge_0_5=float((u_all >= 0.5).mean()),
            )
        blind[c] = blk
    out["sampling_blindness"] = blind

    # ---- 5. cross-arm geometry distance (head-order-free) ------------------ #
    dist_metrics = {}
    for c in CELLS:
        ck = np.where(cells == c)[0]
        cham, kld, cnt = [], [], 0
        for k in ck:
            if not S["found"][k]:
                continue
            bx = box[ann[k]]
            ctr = np.array([(bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2])
            dg_ = float(np.hypot(bx[2] - bx[0], bx[3] - bx[1])) + 1e-6
            sl = data[args.student]["samp_loc"][k]
            aw = data[args.student]["attn_w"][k]
            if not np.isfinite(sl).any():
                continue
            for t in teachers:
                if not M[t]["found"][k]:
                    continue
                tl = data[t]["samp_loc"][k]
                tw = data[t]["attn_w"][k]
                if not np.isfinite(tl).any():
                    continue
                cham.append(chamfer_norm(sl, tl, dg_))
                hs = attn_hist(sl, aw, ctr, dg_)
                ht = attn_hist(tl, tw, ctr, dg_)
                kld.append(0.5 * (kl_pq(hs, ht) + kl_pq(ht, hs)))
                cnt += 1
        dist_metrics[c] = dict(
            n_pairs=cnt,
            chamfer_mean=float(np.mean(cham)) if cham else None,
            chamfer_median=float(np.median(cham)) if cham else None,
            attn_kl_mean=float(np.mean(kld)) if kld else None,
        )
    out["cross_arm_geometry"] = dist_metrics

    # ---- 6. per class ----------------------------------------------------- #
    per_class = {}
    for cn in sorted(set(classes.tolist())):
        cc = classes == cn
        d = {}
        for c in ("correct", "rank_fixable", "calib_only"):
            m2 = cc & (cells == c) & S["found"]
            if m2.sum():
                d[c] = dict(n=int(m2.sum()),
                            score=float(np.nanmean(S["score"][m2])),
                            geom=float(np.nanmean(S["geom"][m2])),
                            mass=float(np.nanmean(S["mass"][m2])))
        hf = cc & (cells == "hard_floor")
        if hf.sum():
            d["hard_floor"] = dict(
                n=int(hf.sum()),
                student_any_geom=float(np.nanmean(S["any_geom"][hf])),
                union_any_geom=float(np.nanmean(
                    np.nanmax(np.stack([M[a]["any_geom"][hf] for a in arms], 1), 1))))
        d["level_geom_correct"] = (
            [float(v) for v in np.nanmean(S["level_geom"][cc & (cells == "correct")], 0)]
            if (cc & (cells == "correct")).sum() else None)
        per_class[cn] = d
    out["per_class"] = per_class

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = EC_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)

    # ---- console summary -------------------------------------------------- #
    print("=" * 96)
    print(f"internal routing -- student={args.student}  split={args.split}  "
          f"lesions={n}  layers={L}")
    print(f"found rate: " + "  ".join(f"{a}={out['found_rate'][a]:.3f}" for a in arms))
    print("=" * 96)
    print("\n[0] control validation -- is routing lesion-specific?")
    print(f"  {'arm':>18} {'n':>5} {'resp':>7} {'rand':>7} {'top1':>7} {'resp-rand':>10}")
    for a, v in ctl.items():
        print(f"  {a:>18} {v['n']:>5} {v['resp_geom']:>7.3f} {v['rand_geom']:>7.3f} "
              f"{v['top1_geom']:>7.3f} {v['resp_minus_rand']:>10.3f}")

    print("\n[1] WITHIN-ARM contrast: correct (scores high) vs rank_fixable "
          "(localises, under-scores)")
    for a in arms:
        d = headline[a]
        if not any(d.get(c, {}).get("n") for c in ("correct", "rank_fixable",
                                                   "calib_only")):
            continue
        print(f"  {a}")
        for c in ("correct", "rank_fixable"):
            v = d.get(c, {})
            if not v.get("n"):
                continue
            print(f"    {c:>13} n={v['n']:>4} score={v['score']:.3f} "
                  f"geom={v['geom']:.3f} mass={v['mass']:.3f} ent={v['entropy']:.3f} "
                  f"iouE={v['iou_last']:.3f} logitE={v['logit_last']:+.3f} "
                  f"logit_min={v['logit_min']:+.3f}")
        dg = d.get("rankfix_minus_correct")
        if dg:
            print(f"    {'Δ(rankfix-correct)':>13} score={dg['score']:+.3f} "
                  f"geom={dg['geom']:+.3f} mass={dg['mass']:+.3f} "
                  f"ent={dg['entropy']:+.3f} iouE={dg['iou_last']:+.3f} "
                  f"logitE={dg['logit_last']:+.3f}")

    print("\n[3] rank_fixable dichotomy (student found the lesion)")
    dd = out["dichotomy_rank_fixable"]
    print(f"  considered={dd['n_considered']}  routing_side={dd['routing_side']}  "
          f"decision_side={dd['decision_side']}  ambiguous={dd['ambiguous']}")
    print(f"  student geom on rank_fixable={dd['student_geom_on_rankfix']} vs "
          f"on correct={dd['student_geom_on_correct']}")

    print("\n[4] sampling blindness (does ANY of the 300 queries cover the lesion?)")
    for c, v in blind.items():
        print(f"  {c}: n={v['n']}  union any_geom={v['union_any_geom_mean']:.3f}  "
              f"frac(any>0.3)={v['frac_union_gt_0_3']:.3f}  "
              f"frac(any<0.05)={v['frac_union_lt_0_05']:.3f}")
        if "union_max_iou_all_mean" in v:
            print(f"      union best box IoU (any class)={v['union_max_iou_all_mean']:.3f}  "
                  f"frac(>=0.5)={v['frac_union_max_iou_all_ge_0_5']:.3f}  "
                  f"union best box IoU (correct class)={v['union_max_iou_cls_mean']:.3f}")
        for a, av in v["per_arm"].items():
            extra = ""
            if "any_q_iou_mean" in av:
                extra = (f" cov_q_iou={av['any_q_iou_mean']:.3f} "
                         f"maxIoU_all={av['max_iou_all_mean']:.3f} "
                         f"maxIoU_cls={av['max_iou_cls_mean']:.3f}")
            print(f"      {a:>18} found={av['found']:>4} any_geom={av['any_geom_mean']:.3f} "
                  f"frac>0.3={av['frac_any_geom_gt_0_3']:.3f} q_score={av['any_q_score_mean']:.3f}"
                  f"{extra}")

    print("\n[5] cross-arm geometry distance (student vs teachers, paired lesions)")
    print(f"  {'cell':>13} {'pairs':>6} {'chamfer':>9} {'chamfer_med':>12} {'attn_KL':>9}")
    for c, v in dist_metrics.items():
        if v["n_pairs"] == 0:
            continue
        print(f"  {c:>13} {v['n_pairs']:>6} {v['chamfer_mean']:>9.4f} "
              f"{v['chamfer_median']:>12.4f} {v['attn_kl_mean']:>9.4f}")

    print("\n[6] per class (student)")
    for cn, v in per_class.items():
        c_ = v.get("correct", {})
        r_ = v.get("rank_fixable", {})
        h_ = v.get("hard_floor", {})
        print(f"  {cn}")
        if c_:
            print(f"    correct      n={c_['n']:>4} score={c_['score']:.3f} "
                  f"geom={c_['geom']:.3f} mass={c_['mass']:.3f}")
        if r_:
            print(f"    rank_fixable n={r_['n']:>4} score={r_['score']:.3f} "
                  f"geom={r_['geom']:.3f} mass={r_['mass']:.3f}")
        if h_:
            print(f"    hard_floor   n={h_['n']:>4} student_any_geom="
                  f"{h_['student_any_geom']:.3f} union_any_geom={h_['union_any_geom']:.3f}")
        if v.get("level_geom_correct"):
            print(f"    geom by level (P3/P4/P5) on correct: "
                  f"{np.round(v['level_geom_correct'], 3).tolist()}")

    print(f"\n-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
