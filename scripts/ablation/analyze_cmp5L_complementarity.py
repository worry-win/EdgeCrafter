"""Lesion-level complementarity analysis for the cmp5L backbone comparison.

Input
-----
Per-(arm, split) prediction dumps written by ``dump_cmp5L_predictions.py``. Every
arm saw the *same* images, so the ground truth is shared and is cross-checked
for consistency across arms before anything is computed.

Method
------
1. Operating point ``(sigma, tau)``: keep predictions with ``score >= sigma``;
   count a match only when ``IoU >= tau`` **and** the class agrees.
2. Per image and per model, match greedily in descending score order, so one GT
   absorbs at most one prediction (COCO-style greedy assignment). Leftover
   predictions are FPs -- including a second box on an already-claimed lesion,
   which keeps "unique FP" meaningful.
3. Project to the **lesion level**: GT g is *hit* by model m iff a prediction of
   m was matched to g. Everything downstream is built on that hit matrix.
4. From it: unique TP, common FN, pairwise FN/TP overlap, the union-versus-best-
   single ceiling, and how much of the reference arm's misses the others recover.

The matching is computed once per operating point and then reused for every
subset (overall / per class / per scale), so stratification is a table lookup
rather than a re-match.

Decisive question for the follow-up: is the union ceiling materially above the
best single model? If ``unique_tp`` is near zero everywhere, per-arm differences
are noise-level and multi-teacher distillation has nothing to transfer.

Usage
-----
    python scripts/ablation/analyze_cmp5L_complementarity.py \
        --pred-dir outputs/ablation/cmp5L_predictions \
        --split test --arms lw_xlarge dinov2b dinov2s mae_vitb \
        --out outputs/ablation/cmp5L_complementarity_test.json
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

EC_ROOT = Path(__file__).resolve().parents[2]

# Breast lesion taxonomy used by this dataset (kept in sync with the npz meta).
CLASS_NAMES = {0: "实(囊实)性", 1: "囊性", 2: "淋巴结", 3: "导管相关病变"}

# COCO area buckets, evaluated on the GT area in px^2.
COCO_SMALL_MAX = 32.0 ** 2    # 1024
COCO_MEDIUM_MAX = 96.0 ** 2   # 9216


def iou_one_to_many(box, boxes):
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.float32)
    lt = np.maximum(box[:2], boxes[:, :2])
    rb = np.minimum(box[2:], boxes[:, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[:, 0] * wh[:, 1]
    ua = (box[2] - box[0]) * (box[3] - box[1])
    ub = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return (inter / np.maximum(ua + ub - inter, 1e-9)).astype(np.float32)


def iou_many_to_many(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    ua = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ub = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = ua[:, None] + ub[None, :] - inter
    return (inter / np.maximum(union, 1e-9)).astype(np.float32)


class ArmPredictions:
    """One arm's predictions on one split, indexed by image."""

    def __init__(self, name, npz_path):
        self.name = name
        self.path = str(npz_path)
        d = np.load(npz_path, allow_pickle=True)
        self.meta = json.loads(Path(str(npz_path)).with_suffix(".meta.json").read_text(encoding="utf-8"))
        # Not every producer records class_names (the decoder-swap cells do not, for
        # instance). Fall back to the breast-lesion class table instead of crashing,
        # so a swap directory can be scored by this same script.
        names = self.meta.get("class_names") or CLASS_NAMES
        self.class_names = {int(k): v for k, v in names.items()}

        self.image_ids = d["image_ids"]
        self.idx_of_image = {int(v): i for i, v in enumerate(self.image_ids)}

        self.rec_ptr = d["rec_ptr"]
        self.box = d["rec_box"]
        self.score = d["rec_score"]
        self.label = d["rec_label"]
        self.logit = d["rec_logit"]
        self.query_id = d["rec_query_id"]

        self.gt_ptr = d["gt_ptr"]
        self.gt_box = d["gt_box"]
        self.gt_cls = d["gt_cls"]
        self.gt_ann = d["gt_ann_id"]
        self.gt_area = d["gt_area"]
        self.gt_image_id = d["gt_image_id"]

        # optional: file names (newer dumps) -- used by the per-GT matrix dump
        self.file_names = d["file_names"] if "file_names" in d.files else None

        self._pred_cache = {}

    def file_name_of(self, image_id):
        if self.file_names is None:
            return ""
        i = self.idx_of_image.get(int(image_id))
        return "" if i is None else str(self.file_names[i])

    def preds(self, image_id, sigma):
        """(boxes, labels, scores) of every query with score >= sigma, UNFILTERED order."""
        key = (image_id, sigma)
        hit = self._pred_cache.get(key)
        if hit is not None:
            return hit
        i = self.idx_of_image.get(int(image_id))
        if i is None:
            out = (np.zeros((0, 4), np.float32), np.zeros(0, np.int8), np.zeros(0, np.float32))
        else:
            lo, hi = int(self.rec_ptr[i]), int(self.rec_ptr[i + 1])
            sc = self.score[lo:hi]
            m = sc >= sigma
            out = (self.box[lo:hi][m], self.label[lo:hi][m], sc[m])
        self._pred_cache[key] = out
        return out

    def gts_local(self, image_id):
        """(box, cls) of the image's GTs, in local (per-image) order."""
        i = self.idx_of_image.get(int(image_id))
        if i is None:
            return (np.zeros((0, 4), np.float32), np.zeros(0, np.int64))
        lo, hi = int(self.gt_ptr[i]), int(self.gt_ptr[i + 1])
        return self.gt_box[lo:hi], self.gt_cls[lo:hi].astype(np.int64)


def match_image(boxes, labels, scores, gt_box, gt_cls, tau):
    """Greedy one-to-one match at a fixed operating point.

    Returns (matched_local_gt set, tp pred idx list, fp pred idx list).
    ``boxes`` are already filtered to score >= sigma.
    """
    if len(boxes) == 0:
        return set(), [], []
    order = np.argsort(-scores, kind="stable")
    iou_all = iou_many_to_many(boxes[order], gt_box) if len(gt_box) else np.zeros((len(order), 0), np.float32)
    gt_taken = np.zeros(len(gt_box), dtype=bool)
    tp, fp = [], []
    for r, p in enumerate(order):
        if len(gt_box):
            cand = np.where((gt_cls == labels[p]) & (iou_all[r] >= tau) & (~gt_taken))[0]
        else:
            cand = np.zeros(0, dtype=int)
        if len(cand):
            g = cand[int(np.argmax(iou_all[r][cand]))]
            gt_taken[g] = True
            tp.append(int(p))
        else:
            fp.append(int(p))
    return set(np.where(gt_taken)[0].tolist()), tp, fp


def classify_fp(box, label, gt_box, gt_cls, tau):
    """background / duplicate / wrong_class, using the image's GTs."""
    same = np.where(gt_cls == label)[0] if len(gt_cls) else np.zeros(0, dtype=int)
    if len(same) and (iou_one_to_many(box, gt_box[same]) >= tau).any():
        return "duplicate"
    other = np.where(gt_cls != label)[0] if len(gt_cls) else np.zeros(0, dtype=int)
    if len(other) and (iou_one_to_many(box, gt_box[other]) >= tau).any():
        return "wrong_class"
    return "background"


def precompute(arms, image_ids, sigma, tau):
    """Match every image for every arm once, plus FP bookkeeping.

    Returns:
      match[arm_name][image_id] = dict(matched:set, tp:list, fp:list, n_pred:int)
      fp_info[arm_name][image_id] = list of dict(pred=int, kind=str, unique=bool)
    """
    match = {a.name: {} for a in arms}
    fp_info = {a.name: {} for a in arms}

    for image_id in image_ids:
        quals = {}
        for a in arms:
            quals[a.name] = a.preds(image_id, sigma)

        for a in arms:
            pb, pl, ps = quals[a.name]
            gb, gc = a.gts_local(image_id)
            matched, tp, fp = match_image(pb, pl, ps, gb, gc, tau)
            match[a.name][image_id] = {
                "matched": matched, "tp": tp, "fp": fp,
                "n_pred": int(len(pb)), "gt_n": int(len(gb)),
            }

            entries = []
            for p in fp:
                kind = classify_fp(pb[p], pl[p], gb, gc, tau)
                unique = True
                for b in arms:
                    if b.name == a.name:
                        continue
                    qb, ql, qs = quals[b.name]
                    if len(qb) == 0:
                        continue
                    same = np.where(ql == pl[p])[0]
                    if len(same) and (iou_one_to_many(pb[p], qb[same]) >= tau).any():
                        unique = False
                        break
                entries.append({"pred": int(p), "kind": kind, "unique": bool(unique)})
            fp_info[a.name][image_id] = entries
    return match, fp_info


def analyze_subset(arms, subset_keys, match, fp_info, sigma, tau, ref_arm=None, fp_label=None):
    """subset_keys: list of (image_id, local_gt_idx) defining the GT population."""
    n_models = len(arms)
    if not subset_keys:
        return {"n_gt": 0}

    keys_by_img = defaultdict(set)
    for image_id, lg in subset_keys:
        keys_by_img[image_id].add(lg)
    pos_of = {k: i for i, k in enumerate(subset_keys)}

    per_model = {a.name: {"tp": 0, "fp": 0, "fn": 0, "n_pred": 0} for a in arms}
    fp_breakdown = {a.name: {"background": 0, "duplicate": 0, "wrong_class": 0} for a in arms}
    unique_fp = Counter()

    hit = np.zeros((len(subset_keys), n_models), dtype=bool)

    for image_id, wanted in keys_by_img.items():
        for i, a in enumerate(arms):
            m = match[a.name][image_id]
            per_model[a.name]["n_pred"] += m["n_pred"]
            hit_local = m["matched"] & wanted
            per_model[a.name]["tp"] += len(hit_local)
            per_model[a.name]["fn"] += len(wanted - m["matched"])
            for lg in hit_local:
                hit[pos_of[(image_id, lg)], i] = True

        # FPs are a per-image property; filter by predicted class when the subset
        # is class-restricted so that per-class precision stays meaningful.
        for a in arms:
            pb, pl, _ = a.preds(image_id, sigma)
            for e in fp_info[a.name][image_id]:
                if fp_label is not None and int(pl[e["pred"]]) != int(fp_label):
                    continue
                per_model[a.name]["fp"] += 1
                fp_breakdown[a.name][e["kind"]] += 1
                if e["unique"]:
                    unique_fp[a.name] += 1

    hits_per_gt = hit.sum(axis=1)
    recall = {a.name: float(hit[:, i].mean()) for i, a in enumerate(arms)}
    best_arm = max(recall, key=recall.get)
    union_recall = float((hits_per_gt > 0).mean())

    pair_fn, pair_tp = {}, {}
    for i, j in combinations(range(n_models), 2):
        na, nb = arms[i].name, arms[j].name
        fa, fb = ~hit[:, i], ~hit[:, j]
        u = int((fa | fb).sum())
        pair_fn[f"{na}|{nb}"] = {
            "fn_overlap": int((fa & fb).sum()),
            "fn_union": u,
            "fn_jaccard": round(int((fa & fb).sum()) / u, 4) if u else None,
        }
        ta, tb = hit[:, i], hit[:, j]
        u2 = int((ta | tb).sum())
        pair_tp[f"{na}|{nb}"] = {
            "tp_overlap": int((ta & tb).sum()),
            "tp_union": u2,
            "tp_jaccard": round(int((ta & tb).sum()) / u2, 4) if u2 else None,
        }

    ref_block = None
    names = [a.name for a in arms]
    if ref_arm in names:
        ri = names.index(ref_arm)
        ref_fn = ~hit[:, ri]
        rec = {a.name: int((ref_fn & hit[:, i]).sum()) for i, a in enumerate(arms) if i != ri}
        rec["n_ref_fn"] = int(ref_fn.sum())
        rec["recovered_by_any"] = int((ref_fn & (hits_per_gt > 0)).sum())
        rec["recovered_by_none"] = int((ref_fn & (hits_per_gt == 0)).sum())
        ref_block = {"arm": ref_arm, **rec}

    return {
        "n_gt": int(len(subset_keys)),
        "sigma": sigma,
        "tau": tau,
        "n_models": n_models,
        "per_model": {
            a.name: {
                **per_model[a.name],
                "recall": round(recall[a.name], 4),
                "precision": round(per_model[a.name]["tp"] / max(per_model[a.name]["tp"] + per_model[a.name]["fp"], 1), 4),
                "f1": round(2 * per_model[a.name]["tp"] /
                            max(2 * per_model[a.name]["tp"] + per_model[a.name]["fp"] + per_model[a.name]["fn"], 1), 4),
                "unique_fp": int(unique_fp[a.name]),
                "fp_breakdown": fp_breakdown[a.name],
            } for a in arms
        },
        "hit_histogram": {str(k): int((hits_per_gt == k).sum()) for k in range(n_models + 1)},
        "all_models_hit": int((hits_per_gt == n_models).sum()),
        "no_model_hit": int((hits_per_gt == 0).sum()),
        "unique_tp": {a.name: int(((hit[:, i]) & (hits_per_gt == 1)).sum()) for i, a in enumerate(arms)},
        "union_recall": round(union_recall, 4),
        "best_single_arm": best_arm,
        "best_single_recall": round(recall[best_arm], 4),
        "union_gain_over_best_single": round(union_recall - recall[best_arm], 4),
        "pairwise_fn": pair_fn,
        "pairwise_tp": pair_tp,
        "reference": ref_block,
    }


def build_hit_matrix(arms, subset_keys, match):
    """(n_gt, n_arms) boolean matrix -- the per-lesion hit matrix itself."""
    pos_of = {k: i for i, k in enumerate(subset_keys)}
    hit = np.zeros((len(subset_keys), len(arms)), dtype=bool)
    keys_by_img = defaultdict(set)
    for image_id, lg in subset_keys:
        keys_by_img[image_id].add(lg)
    for image_id, wanted in keys_by_img.items():
        for i, a in enumerate(arms):
            for lg in (match[a.name][image_id]["matched"] & wanted):
                hit[pos_of[(image_id, lg)], i] = True
    return hit


def dump_hit_matrix_csv(path, arms, subset_keys, match, gt_meta, file_name_of,
                        sigma, tau, split):
    """Write the raw per-GT hit matrix: one row per lesion, one column per arm.

    This is the artefact behind every aggregate number: with it the user can
    re-derive any subset (class / scale / image) without re-running inference.
    """
    hit = build_hit_matrix(arms, subset_keys, match)
    names = [a.name for a in arms]
    if "ecvits_official" not in names:
        names = names + ["ecvits_official"]          # reserved 5th column
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "sigma", "tau", "row", "image_id", "file_name",
                    "ann_id", "gt_class", "gt_class_name", "gt_area", "gt_scale"]
                   + ["hit_" + n for n in names]
                   + ["n_hits", "unique_to"])
        for i, key in enumerate(subset_keys):
            ann_id, cls, area, scale = gt_meta[key]
            hits = hit[i]
            n_hits = int(hits.sum())
            who = [arms[j].name for j in range(len(arms)) if hits[j]]
            uniq = who[0] if n_hits == 1 else ""
            row_hits = [int(v) for v in hits]
            if len(names) > len(arms):
                row_hits = row_hits + [""]          # EC arm not evaluated yet
            w.writerow([split, sigma, tau, i, key[0], file_name_of(key[0]),
                        ann_id, cls, CLASS_NAMES.get(cls, str(cls)), area, scale]
                       + row_hits + [n_hits, uniq])
    print("[matrix] %s  (%d lesions x %d arms)" % (path, len(subset_keys), len(arms)))
    return hit


def collect_dets(arm_list, image_ids, sigma_min):
    """Pool detections from one or several arms into (score, image_id, box, label)."""
    dets = []
    for a in arm_list:
        for image_id in image_ids:
            lo = a.idx_of_image[int(image_id)]
            lo_p, hi_p = int(a.rec_ptr[lo]), int(a.rec_ptr[lo + 1])
            sc = a.score[lo_p:hi_p]
            for p in np.where(sc >= sigma_min)[0]:
                dets.append((float(sc[p]), int(image_id), a.box[lo_p:hi_p][p], int(a.label[lo_p:hi_p][p])))
    return dets


def nms_per_class(boxes, labels, scores, iou_thresh):
    """Greedy per-class NMS. Returns kept indices into the input arrays."""
    keep = []
    for c in np.unique(labels):
        m = np.where(labels == c)[0]
        order = m[np.argsort(-scores[m], kind="stable")]
        b, idx = boxes[order], order
        while len(idx):
            keep.append(int(idx[0]))
            if len(idx) == 1:
                break
            ious = iou_one_to_many(b[0], b[1:])
            sel = ious < iou_thresh
            b, idx = b[1:][sel], idx[1:][sel]
    return keep


def fuse_nms(arm_list, image_ids, sigma_min, iou_thresh=0.6, label_filter=None):
    """Box-level ensemble: pool every arm's detections, then per-class NMS per image.

    Without this step a pooled ranking is dominated by each GT being claimed once
    and then hit by three duplicate boxes, which destroys precision. NMS is the
    cheapest way to get a *usable* ensemble number and is the honest ceiling for
    what several teachers could jointly provide.
    """
    fused = []
    for image_id in image_ids:
        cands = []
        for a in arm_list:
            lo = a.idx_of_image[int(image_id)]
            lo_p, hi_p = int(a.rec_ptr[lo]), int(a.rec_ptr[lo + 1])
            sc = a.score[lo_p:hi_p]
            for p in np.where(sc >= sigma_min)[0]:
                lab = int(a.label[lo_p:hi_p][p])
                if label_filter is not None and lab != label_filter:
                    continue
                cands.append((float(sc[p]), a.box[lo_p:hi_p][p], lab))
        if not cands:
            continue
        scores = np.asarray([c[0] for c in cands], dtype=np.float32)
        boxes = np.asarray([c[1] for c in cands], dtype=np.float32)
        labels = np.asarray([c[2] for c in cands], dtype=np.int8)
        for k in nms_per_class(boxes, labels, scores, iou_thresh):
            fused.append((float(scores[k]), int(image_id), boxes[k], int(labels[k])))
    return fused


def restrict_gt(gt_by_img, cls):
    """Class-conditional GT map, mirroring COCO's per-category evaluation."""
    out = {}
    for i, (gb, gc) in gt_by_img.items():
        m = gc == cls
        out[i] = (gb[m], gc[m])
    return out


def ap_from_dets(dets, gt_by_img, tau):
    """Simplified AP at one IoU threshold via COCO-style global greedy matching.

    Deliberately omits the area-range / maxDets=100 / ignore handling so that
    every row is scored by identical rules. Absolute values therefore sit a bit
    above the official COCO AP50; only the *comparison* between rows is meant to
    be read.
    """
    if not dets:
        return None
    n_gt = int(sum(len(gt_by_img[i][1]) for i in gt_by_img))
    if n_gt == 0:
        return None

    dets = sorted(dets, key=lambda t: -t[0])
    taken = {i: np.zeros(len(gt_by_img[i][1]), dtype=bool) for i in gt_by_img}

    tp = np.zeros(len(dets), dtype=np.int64)
    for k, (score, image_id, box, label) in enumerate(dets):
        gb, gc = gt_by_img[image_id]
        cand = np.where((gc == label) & (~taken[image_id]))[0]
        if len(cand):
            ious = iou_one_to_many(box, gb[cand])
            j = int(np.argmax(ious))
            if ious[j] >= tau:
                taken[image_id][cand[j]] = True
                tp[k] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(1 - tp)
    rec = tp_cum / n_gt
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    ap = 0.0
    for t in np.linspace(0, 1, 101):
        m = rec >= t
        ap += (float(prec[m].max()) if m.any() else 0.0)
    ap /= 101.0

    return {
        "ap": round(ap, 4),
        "n_det": int(len(dets)),
        "n_gt": n_gt,
        "recall_at_end": round(float(rec[-1]), 4),
        "max_precision": round(float(prec.max()), 4),
        "precision_at_recall_0.5": round(float(prec[rec >= 0.5].max()) if (rec >= 0.5).any() else 0.0, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference-arm", default="dinov2s")
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.05, 0.25, 0.5])
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--dump-hit-matrix", default=None,
                    help="write the raw per-GT hit matrix (CSV) at --sigmas[0] here")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    if not pred_dir.is_absolute():
        pred_dir = EC_ROOT / pred_dir

    arms = []
    for name in args.arms:
        p = pred_dir / f"{name}_{args.split}.npz"
        if not p.exists():
            print(f"[skip] {name}: {p} not found", file=sys.stderr)
            continue
        arms.append(ArmPredictions(name, p))
    if not arms:
        print("no prediction files found", file=sys.stderr)
        return 1

    print("[load] %s: %s" % (args.split, ", ".join(f"{a.name}({len(a.image_ids)} img)" for a in arms)))

    ref = arms[0]
    integrity = {"n_arms": len(arms), "arms": [a.name for a in arms], "checks": {}}
    for a in arms[1:]:
        integrity["checks"][f"same_images_vs_{ref.name}"] = bool(
            len(a.image_ids) == len(ref.image_ids) and np.array_equal(a.image_ids, ref.image_ids))
        integrity["checks"][f"same_gt_vs_{ref.name}"] = bool(
            a.gt_box.shape == ref.gt_box.shape and np.allclose(a.gt_box, ref.gt_box)
            and np.array_equal(a.gt_cls, ref.gt_cls))

    image_ids = [int(v) for v in ref.image_ids]

    # canonical GT population as (image_id, local_gt_idx) keys
    gt_records = []
    for i, iid in enumerate(image_ids):
        lo, hi = int(ref.gt_ptr[i]), int(ref.gt_ptr[i + 1])
        for lg in range(hi - lo):
            gi = lo + lg
            gt_records.append((iid, lg, int(ref.gt_cls[gi]), float(ref.gt_area[gi])))
    integrity["n_gt_total"] = len(gt_records)
    integrity["n_gt_per_class"] = {str(k): int(v) for k, v in
                                   Counter(r[2] for r in gt_records).items()}
    integrity["n_gt_per_scale"] = {
        "small": int(sum(1 for r in gt_records if r[3] < COCO_SMALL_MAX)),
        "medium": int(sum(1 for r in gt_records if COCO_SMALL_MAX <= r[3] < COCO_MEDIUM_MAX)),
        "large": int(sum(1 for r in gt_records if r[3] >= COCO_MEDIUM_MAX)),
    }
    print("[gt] %d total, per class %s, per scale %s"
          % (len(gt_records), integrity["n_gt_per_class"], integrity["n_gt_per_scale"]))

    all_keys = [(r[0], r[1]) for r in gt_records]

    # (image_id, local_gt_idx) -> (ann_id, class, area, coco scale)
    gt_meta = {}
    for r in gt_records:
        gi = int(ref.gt_ptr[ref.idx_of_image[int(r[0])]]) + r[1]
        gt_meta[(r[0], r[1])] = (
            int(ref.gt_ann[gi]), int(r[2]), float(r[3]),
            "small" if r[3] < COCO_SMALL_MAX
            else "medium" if r[3] < COCO_MEDIUM_MAX else "large")

    class_keys = defaultdict(list)
    scale_keys = defaultdict(list)
    for r in gt_records:
        class_keys[r[2]].append((r[0], r[1]))
        if r[3] < COCO_SMALL_MAX:
            scale_keys["small"].append((r[0], r[1]))
        elif r[3] < COCO_MEDIUM_MAX:
            scale_keys["medium"].append((r[0], r[1]))
        else:
            scale_keys["large"].append((r[0], r[1]))

    results = {
        "split": args.split,
        "arms": [a.name for a in arms],
        "integrity": integrity,
        "class_names": {int(k): v for k, v in ref.class_names.items()},
        "operating_points": [],
    }

    for sigma in args.sigmas:
        print("[op] sigma=%s tau=%s ..." % (sigma, args.tau))
        match, fp_info = precompute(arms, image_ids, sigma, args.tau)
        op = {
            "sigma": sigma,
            "tau": args.tau,
            "overall": analyze_subset(arms, all_keys, match, fp_info, sigma, args.tau, args.reference_arm),
            "by_class": {str(c): analyze_subset(arms, class_keys[c], match, fp_info, sigma, args.tau,
                                                args.reference_arm, fp_label=c)
                         for c in sorted(class_keys)},
            "by_scale": {s: analyze_subset(arms, scale_keys[s], match, fp_info, sigma, args.tau, args.reference_arm)
                         for s in ("small", "medium", "large")},
        }
        op["by_class_names"] = {str(c): ref.class_names.get(c, str(c)) for c in sorted(class_keys)}
        o = op["overall"]
        print("     union %.4f vs best single %s %.4f (gain %+.4f) | uniqueTP %s | no-model-hit %d/%d"
              % (o["union_recall"], o["best_single_arm"], o["best_single_recall"],
                 o["union_gain_over_best_single"], o["unique_tp"], o["no_model_hit"], o["n_gt"]))
        results["operating_points"].append(op)

        if args.dump_hit_matrix and abs(sigma - args.sigmas[0]) < 1e-12:
            out_csv = Path(args.dump_hit_matrix)
            if not out_csv.is_absolute():
                out_csv = EC_ROOT / out_csv
            dump_hit_matrix_csv(out_csv, arms, all_keys, match, gt_meta,
                                ref.file_name_of, sigma, args.tau, args.split)

    # ---- AP ceiling: what would a box-level ensemble actually buy? ----------
    sigma_min = 0.02
    print("[ap] tau=%s sigma_min=%s ..." % (args.tau, sigma_min))
    gt_by_img = {int(i): ref.gts_local(int(i)) for i in image_ids}

    ap_block = {
        "tau": args.tau,
        "sigma_min": sigma_min,
        "nms_iou": 0.6,
        "note": ("simplified AP at one IoU threshold, COCO-style global greedy matching, "
                 "no area-range / maxDets / ignore handling; identical rules for every row, so "
                 "read the differences, not the absolute values. 'ensemble_nms' pools all arms' "
                 "boxes and applies per-class NMS(0.6) before ranking -- an un-fused pooled "
                 "ranking is meaningless because every GT would be claimed once and then hit by "
                 "duplicate boxes from the other arms."),
        "per_arm": {},
    }
    for a in arms:
        r = ap_from_dets(collect_dets([a], image_ids, sigma_min), gt_by_img, args.tau)
        ap_block["per_arm"][a.name] = r
        print("     AP50(%-11s) = %.4f  (R_end %.4f, n_det %d)" % (a.name, r["ap"], r["recall_at_end"], r["n_det"]))

    pooled_unfused = ap_from_dets(collect_dets(arms, image_ids, sigma_min), gt_by_img, args.tau)
    ap_block["pooled_without_nms"] = pooled_unfused

    nms_iou = 0.6
    fused = fuse_nms(arms, image_ids, sigma_min, iou_thresh=nms_iou)
    ens = ap_from_dets(fused, gt_by_img, args.tau)
    ap_block["ensemble_nms"] = ens
    print("     AP50(pooled, NO nms)  = %.4f  (n_det %d)  <- meaningless, kept as a warning"
          % (pooled_unfused["ap"], pooled_unfused["n_det"]))
    print("     AP50(ensemble NMS %.1f) = %.4f  (R_end %.4f, n_det %d)"
          % (nms_iou, ens["ap"], ens["recall_at_end"], ens["n_det"]))

    best = max(ap_block["per_arm"].items(), key=lambda kv: kv[1]["ap"])
    ap_block["best_single_arm"] = best[0]
    ap_block["best_single_ap"] = best[1]["ap"]
    ap_block["ensemble_gain_over_best_single"] = round(ens["ap"] - best[1]["ap"], 4)
    print("     gain of ensemble over best single (%s %.4f): %+.4f"
          % (best[0], best[1]["ap"], ap_block["ensemble_gain_over_best_single"]))
    # ---- per-class AP: is the ensemble gain concentrated somewhere useful? --
    print("[ap] per class ...")
    per_class_ap = {}
    for c in sorted({int(v) for v in ref.gt_cls}):
        gt_c = restrict_gt(gt_by_img, c)
        n_gt_c = int(sum(len(v[1]) for v in gt_c.values()))
        if n_gt_c == 0:
            continue
        row = {"n_gt": n_gt_c, "class_name": ref.class_names.get(c, str(c)), "per_arm": {}}
        for a in arms:
            dets = [d for d in collect_dets([a], image_ids, sigma_min) if d[3] == c]
            row["per_arm"][a.name] = ap_from_dets(dets, gt_c, args.tau)
        fused_c = fuse_nms(arms, image_ids, sigma_min, iou_thresh=nms_iou, label_filter=c)
        row["ensemble_nms"] = ap_from_dets(fused_c, gt_c, args.tau)
        scored = [(k, v["ap"]) for k, v in row["per_arm"].items() if v]
        if scored:
            row["best_single_arm"], row["best_single_ap"] = max(scored, key=lambda kv: kv[1])
            if row["ensemble_nms"]:
                row["ensemble_gain_over_best_single"] = round(
                    row["ensemble_nms"]["ap"] - row["best_single_ap"], 4)
                print("     class %-14s n_gt=%-5d best(%s)=%.4f  ensemble=%.4f  gain=%+.4f"
                      % (row["class_name"], n_gt_c, row["best_single_arm"], row["best_single_ap"],
                         row["ensemble_nms"]["ap"], row["ensemble_gain_over_best_single"]))
        per_class_ap[str(c)] = row
    ap_block["per_class"] = per_class_ap

    results["detection_ap"] = ap_block

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = EC_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print("[done] %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
