"""Internal-routing diagnosis for the cmp5L five arms (zero training).

Motivation
----------
`D_j(Z_i)` collapses in BOTH off-diagonal directions (Case C): the five necks do
NOT land in a shared decoder-ready space, so aligning feature *values* is the
wrong primitive. But the complementarity is real and concentrated on rank_fixable
/ feat_fixable lesions. The claim to test is therefore behavioural, not
representational:

    two arms may disagree on where to look / how much to trust a look
    even when their neck features live in different spaces.

So we do not compare feature vectors. We compare the *decisions* of the single
query that each arm uses for a given lesion, at the deformable-attention
granularity where those decisions are made:

  1. sampling points      -- where does the query actually look?
  2. attention weights    -- how much is it convinced by each look?
  3. reference trajectory -- how does its box walk across the 4 layers?
  (+ query-embedding displacement and the raw per-layer class logit as context)

All three are captured read-only via hooks (see probe_cmp5L_decoder_hooks.py for
the verified tensor map). Nothing in the model is modified.

Why this can separate the two failure modes
-------------------------------------------
For a lesion the student ALREADY localises but scores too low (rank_fixable),
"perception ceiling 0.93 vs AP50 0.75" says the evidence is present. This script
tells us which of two stories holds for each such lesion:

  routing-side  the student's sampling points do NOT cover the lesion
                (samp_in_box low, box_center_dist high) while the teacher's do
                -> the query is looking in the wrong place; distillation target
                   = where to look (offsets / attention).

  decision-side the student's sampling covers the lesion as well as the
                teacher's, but its attention mass / logit is diluted
                (attn_inbox_mass low, entropy high, logit_c[l] flat)
                -> it looked and was not convinced; distillation target
                   = how much to trust (attention weights / logit calibration).

Those have different fixes, and a single feature-space KD loss conflates them.

Responsible query
-----------------
For GT g (class c, box B, pixels) and one arm: among the arm's 300 queries take
those whose predicted label == c and IoU(box, B) >= 0.5, and keep the
highest-scoring one. That is exactly the criterion the low-threshold dumper and
the complementarity analysis already use, so the cells line up.

Controls, always recorded
-------------------------
  ctrl_random  a seeded random query  -> generic routing baseline
  ctrl_top1    the image-global top-1 query -> "confidently looking somewhere else"
Both are scored with the SAME in-box metric against the SAME lesion box, which
makes "routing is lesion-specific" falsifiable rather than assumed.

Usage
-----
    # 1. build the lesion list once (no GPU)
    python scripts/ablation/diag_cmp5L_internal_routing.py --dump-lesions \
        --score-ranking outputs/ablation/cmp5L_score_ranking_pergt_test.csv \
        --ann-file <test.json> --split-name test \
        --lesions-out outputs/ablation/cmp5L_internal_lesions_test.csv

    # 2. one arm per task (array 0-4)
    python scripts/ablation/diag_cmp5L_internal_routing.py \
        --arm dinov2s --config ... --checkpoint ... \
        --lesions outputs/ablation/cmp5L_internal_lesions_test.csv \
        --ann-file <test.json> --img-folder ... --split-name test \
        --out outputs/ablation/cmp5L_internal_routing/dinov2s_test.npz
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from pycocotools.coco import COCO

EC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EC_ROOT / "ecdetseg"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.core import YAMLConfig          # noqa: E402
from engine.solver import TASKS             # noqa: E402
from engine.misc import dist_utils          # noqa: E402
import engine.edgecrafter.decoder as dec_mod  # noqa: E402

CELLS = ["correct", "rank_fixable", "calib_only", "feat_fixable", "hard_floor"]
CLASS_NAMES = {0: "solid", 1: "cystic", 2: "lymph", 3: "duct"}


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def xyxy_iou_matrix(a, b):
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return (inter / np.clip(aa[:, None] + bb[None, :] - inter, 1e-9, None)).astype(np.float32)


def cxcywh_to_xyxy_px(boxes_norm, ow, oh):
    """normalised cxcywh [..,4] -> pixel xyxy. The eval transform is a plain
    stretch to eval_spatial_size (no letterbox), so per-axis scaling is exact."""
    b = boxes_norm.copy()
    cx = b[..., 0] * ow
    cy = b[..., 1] * oh
    w = b[..., 2] * ow
    h = b[..., 3] * oh
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1).astype(np.float32)


def pts_in_box(pts_xy, box_xyxy):
    """pts_xy [...,2] pixels -> bool [...,] inside box."""
    x0, y0, x1, y1 = box_xyxy
    return (pts_xy[..., 0] >= x0) & (pts_xy[..., 0] <= x1) & \
           (pts_xy[..., 1] >= y0) & (pts_xy[..., 1] <= y1)


# --------------------------------------------------------------------------- #
# MSDeformableAttention patch: stash the tensors that never leave the function
# --------------------------------------------------------------------------- #
CAP = {}


def install_msa_patch():
    if getattr(dec_mod.MSDeformableAttention.forward, "_diag_patched", False):
        return
    orig = dec_mod.MSDeformableAttention.forward

    def patched(self, query, reference_points, value, value_spatial_shapes):
        bs, Len_q = query.shape[:2]
        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, Len_q, self.num_heads, sum(self.num_points_list), 2)
        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = torch.nn.functional.softmax(attention_weights, dim=-1)
        if reference_points.shape[-1] != 4:
            raise SystemExit("unexpected ref last dim %d" % reference_points.shape[-1])
        nps = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
        offset = sampling_offsets * nps * reference_points[:, :, None, :, 2:] * self.offset_scale
        sampling_locations = reference_points[:, :, None, :, :2] + offset

        tag = getattr(self, "_diag_tag", -1)
        CAP.setdefault(tag, []).append(dict(
            sampling_offsets=sampling_offsets.detach().float().cpu(),
            attention_weights=attention_weights.detach().float().cpu(),
            sampling_locations=sampling_locations.detach().float().cpu(),
        ))
        return self.ms_deformable_attn_core(
            value, value_spatial_shapes, sampling_locations, attention_weights,
            self.num_points_list)

    patched._diag_patched = True
    dec_mod.MSDeformableAttention.forward = patched


def build_module(config, checkpoint, weights, device, num_workers, batch_size,
                 ann_file, img_folder):
    cfg = YAMLConfig(config, **{
        "val_dataloader": {
            "dataset": {"ann_file": ann_file, "img_folder": img_folder},
            "num_workers": num_workers,
            "total_batch_size": batch_size,
            "shuffle": False,
            "drop_last": False,
        },
        "num_classes": 4,
        "remap_mscoco_category": False,
    })
    for bn in ("ViTAdapter", "DinoV2Adapter"):
        if bn in cfg.yaml_cfg:
            cfg.yaml_cfg[bn]["skip_load_backbone"] = True
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver._setup()
    solver.load_resume_state(checkpoint)
    module = solver.ema.module if (weights == "ema" and solver.ema is not None) else solver.model
    module = dist_utils.de_parallel(module)
    module.eval()
    return solver, module


# --------------------------------------------------------------------------- #
# lesion list
# --------------------------------------------------------------------------- #
def dump_lesions(args):
    rows = list(csv.DictReader(open(args.score_ranking, encoding="utf-8")))
    coco = COCO(args.ann_file)
    img_of_ann, box_of_ann = {}, {}
    for a in coco.dataset["annotations"]:
        img_of_ann[int(a["id"])] = int(a["image_id"])
        box_of_ann[int(a["id"])] = a["bbox"]

    rng = random.Random(args.seed)
    by_cell = {c: [] for c in CELLS}
    for r in rows:
        c = r["cell"]
        if c in by_cell:
            by_cell[c].append(r)

    picked = []
    for c in CELLS:
        pool = by_cell[c]
        if c == "correct" and args.per_cell and len(pool) > args.per_cell:
            # stratify the control group by class, otherwise it is 74% solid and the
            # rare classes we actually care about get ~5 samples each
            by_cls = {}
            for r in pool:
                by_cls.setdefault(r["gt_class"], []).append(r)
            quota = max(1, args.per_cell // max(1, len(by_cls)))
            chosen, taken = [], set()
            for cn, rs in sorted(by_cls.items()):
                for r in rng.sample(rs, min(quota, len(rs))):
                    chosen.append(r)
                    taken.add(id(r))
            leftovers = [r for r in pool if id(r) not in taken]
            rng.shuffle(leftovers)
            chosen.extend(leftovers[:max(0, args.per_cell - len(chosen))])
            pool = chosen
        elif args.per_cell and len(pool) > args.per_cell:
            pool = rng.sample(pool, args.per_cell)
        for r in pool:
            ann_id = int(r["ann_id"])
            if ann_id not in img_of_ann:
                continue
            x, y, w, h = box_of_ann[ann_id]
            picked.append(dict(
                ann_id=ann_id, image_id=img_of_ann[ann_id],
                gt_class=int(r["gt_class_id"]) if r.get("gt_class_id") not in (None, "")
                else CLASS_NAMES_LOOKUP.get(r["gt_class"], -1),
                cls_name=r["gt_class"], cell=c, gt_scale=r["gt_scale"],
                student_localised=int(r["student_localised"]),
                student_score=float(r["student_score"]),
                teacher_best_score=float(r["teacher_best_score"]),
                teacher_best_arm=r["teacher_best_arm"],
                x0=float(x), y0=float(y), x1=float(x + w), y1=float(y + h),
            ))

    # a stable, deterministic order so every arm writes the same index space
    picked.sort(key=lambda d: (d["cell"], d["ann_id"]))
    out_csv = Path(args.lesions_out)
    if not out_csv.is_absolute():
        out_csv = (EC_ROOT / out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["lesion_idx", "ann_id", "image_id", "gt_class", "cls_name", "cell",
            "gt_scale", "student_localised", "student_score",
            "teacher_best_score", "teacher_best_arm", "x0", "y0", "x1", "y1"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, d in enumerate(picked):
            d = dict(d)
            d["lesion_idx"] = i
            w.writerow({k: d.get(k) for k in cols})

    counts = {}
    for d in picked:
        counts[d["cell"]] = counts.get(d["cell"], 0) + 1
    print(f"[lesions] {len(picked)} lesions -> {out_csv}")
    print(f"[lesions] by cell: {counts}")
    n_img = len({d['image_id'] for d in picked})
    print(f"[lesions] distinct images: {n_img}")
    return 0


CLASS_NAMES_LOOKUP = {"实(囊实)性": 0, "囊性": 1, "淋巴结": 2, "导管相关病变": 3}


def read_lesions(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        for k in ("lesion_idx", "ann_id", "image_id", "gt_class"):
            r[k] = int(r[k])
        for k in ("x0", "y0", "x1", "y1", "student_score", "teacher_best_score"):
            r[k] = float(r[k])
        r["student_localised"] = int(r["student_localised"])
    return rows


# --------------------------------------------------------------------------- #
# main run
# --------------------------------------------------------------------------- #
def run_arm(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("FATAL: CUDA unavailable; refusing to silently run on CPU.", file=sys.stderr)
        return 2

    install_msa_patch()
    solver, module = build_module(args.config, args.checkpoint, args.weights, device,
                                 args.num_workers, args.batch_size,
                                 args.ann_file, args.img_folder)
    ec = module.decoder
    td = ec.decoder
    # EVAL-mode gotcha: TransformerDecoder.forward appends to dec_out_* only when
    # ``i == eval_idx`` and breaks right after -- so in eval the returned tensors
    # are the LAST layer only, and (because the score head is called between
    # layers, not inside them) the per-layer class logits are never evaluated at
    # all. Flipping ONLY the decoder submodule to train mode restores the full
    # per-layer computation with no side effects here: every Dropout in the
    # decoder is p=0, there is no BatchNorm, and eval_idx == num_layers-1 so the
    # selected output is the same last layer as in eval. --decoder-eval-mode makes
    # it easy to confirm the last-layer prediction is bit-identical both ways.
    if not args.decoder_eval_mode:
        td.train()
    L = len(td.layers)
    msa = td.layers[0].cross_attn
    H, P = msa.num_heads, sum(msa.num_points_list)
    npl = list(msa.num_points_list)
    level_of_point = np.concatenate([[li] * n for li, n in enumerate(npl)])
    print(f"[init] arm={args.arm} layers={L} eval_idx={td.eval_idx} heads={H} P={P} "
          f"points={npl} offset_scale={msa.offset_scale}", flush=True)

    # hooks.  Layer index bookkeeping: the score head is NOT called inside the
    # layer -- TransformerDecoder calls it between layer i and layer i+1, and in
    # EVAL it is called for i == eval_idx ONLY (the loop breaks right after). So a
    # positional "append in order" list would hold a single entry, and keying by
    # registration index silently yields zeros for the other layers. Instead each
    # layer's pre-hook publishes its own index, which is still current when the
    # score head fires; that is exact for both shared and per-layer heads.
    caught = {"ref": {}, "qry": {}}
    layer_state = {"cur": -1}
    logits_by_layer = {}
    for i, layer in enumerate(td.layers):
        layer.cross_attn._diag_tag = i

        def make_pre(idx):
            # NOTE: a forward_pre_hook's return value REPLACES the input tuple.
            # Returning anything non-None here silently drops arguments (this bit
            # us with a lambda that returned the tuple from two setitem calls), so
            # this must be a real function that returns None.
            def pre(mod, inp):
                caught["ref"][idx] = inp[1].detach().float().cpu()
                layer_state["cur"] = idx
                return None
            return pre

        def make_post(idx):
            def post(mod, inp, out):
                caught["qry"][idx] = (out if torch.is_tensor(out)
                                      else out[0]).detach().float().cpu()
                return None
            return post

        layer.register_forward_pre_hook(make_pre(i))
        layer.register_forward_hook(make_post(i))
    seen_head_ids = set()
    for i in range(len(ec.dec_score_head)):
        h = ec.dec_score_head[i]
        if isinstance(h, torch.nn.Identity) or id(h) in seen_head_ids:
            continue
        seen_head_ids.add(id(h))
        h.register_forward_hook(
            (lambda m, inp, out: logits_by_layer.__setitem__(
                layer_state["cur"], out.detach().float().cpu())))
    print(f"[init] score heads: {len(ec.dec_score_head)} "
          f"(distinct modules={len(seen_head_ids)}) "
          f"-> {'SHARED' if len(seen_head_ids) < len(ec.dec_score_head) else 'per-layer'}; "
          f"in EVAL only layer {td.eval_idx} calls the head", flush=True)

    lesions = read_lesions(args.lesions)
    needed = {d["image_id"] for d in lesions}
    by_img = {}
    for k, d in enumerate(lesions):
        by_img.setdefault(d["image_id"], []).append(k)
    n_les = len(lesions)
    n_cls = 4

    # output arrays
    Z = lambda *s, dt=np.float32: np.zeros(s, dtype=dt)
    A = dict(
        found=Z(n_les, dt=np.int8),
        q_star=Z(n_les, dt=np.int32),
        score_q=Z(n_les),
        iou_final=Z(n_les),
        n_cand=Z(n_les, dt=np.int16),
        # per-layer
        iou_ref=Z(n_les, L),                # IoU(ref box fed into layer l, lesion)
        qemb_delta=Z(n_les, L),             # ||q_l - q_{l-1}||, layer 0 vs init
        qemb_norm=Z(n_les, L),
        logit_c=Z(n_les, L),                # raw class-c logit at layer l
        attn_entropy=Z(n_les, L),           # mean over heads, / log P
        attn_max=Z(n_les, L),               # mean over heads of max_p w
        samp_inbox=Z(n_les, L),             # mean over (h,p) of [loc in lesion]
        samp_mass_inbox=Z(n_les, L),        # sum_{h,p} w * [loc in lesion]
        samp_valid=Z(n_les, L),             # mean over (h,p) of [loc in image]
        samp_dist=Z(n_les, L),              # mean ||loc_px - centre|| / diag
        samp_inbox_lvl=Z(n_les, L, len(npl)),
        # controls (random query / image top-1), same metrics
        ctl_rand_inbox=Z(n_les, L),
        ctl_rand_mass=Z(n_les, L),
        ctl_top1_inbox=Z(n_les, L),
        ctl_top1_mass=Z(n_les, L),
        # over ALL 300 queries: the best-covering query's geometry and its score.
        # This is the only internal signal available for hard_floor lesions.
        any_inbox_max=Z(n_les, L),
        any_mass_max=Z(n_les, L),
        any_inbox_q_score=Z(n_les, L),
        any_inbox_q_iou=Z(n_les, L),
        any_inbox_q_label=Z(n_les, L, dt=np.int8),
        max_iou_all=Z(n_les, L),
        max_iou_cls=Z(n_les, L),
        # per-level aggregation over layers for the responsible query
        samp_inbox_pts=Z(n_les, L, H, P),
        # payload for cross-arm Chamfer / attention-distribution distance (responsible
        # query only). NaN-filled so "arm never localised this lesion" is never
        # mistaken for "looked at pixel 0,0".
        samp_loc=np.full((n_les, L, H, P, 2), np.nan, dtype=np.float16),
        attn_w=np.full((n_les, L, H, P), np.nan, dtype=np.float16),
    )

    seen_imgs, hit_imgs = 0, 0
    t0 = time.time()
    with torch.no_grad():
        for samples, targets in solver.cfg.val_dataloader:
            ids = [int(t["image_id"].item()) if torch.is_tensor(t["image_id"])
                   else int(t["image_id"]) for t in targets]
            seen_imgs += len(ids)
            if not any(i in needed for i in ids):
                continue
            CAP.clear()
            for li in list(caught["ref"]):
                caught["ref"][li] = None
                caught["qry"][li] = None
            layer_state["cur"] = -1
            logits_by_layer.clear()

            x = samples.to(device)
            B = module.backbone(x)
            Zneck = module.encoder(B)
            out = module.decoder(Zneck, None)
            if isinstance(out, (list, tuple)):
                out = out[0]
            pl = out["pred_logits"].float()
            pb = out["pred_boxes"].float()
            if pl.dim() == 2:
                pl, pb = pl.unsqueeze(0), pb.unsqueeze(0)

            logit_np = pl.cpu().numpy()
            box_np = pb.cpu().numpy()
            prob_np = 1.0 / (1.0 + np.exp(-logit_np))
            score_np = prob_np.max(axis=-1)
            label_np = prob_np.argmax(axis=-1).astype(np.int8)

            # stack captured tensors -> [B, L, ...]
            ref_l = []      # per layer [B,Q,1,4]
            qry_l = []
            for l in range(L):
                ref_l.append(caught["ref"][l].numpy())
                qry_l.append(caught["qry"][l].numpy())
            loc_l, attn_l = [], []
            for l in range(L):
                rec = CAP[l][0]
                loc_l.append(rec["sampling_locations"].numpy())
                attn_l.append(rec["attention_weights"].numpy())
            loc = np.stack(loc_l, 1)      # [B,L,Q,H,P,2]
            attn = np.stack(attn_l, 1)    # [B,L,Q,H,P]
            ref = np.stack(ref_l, 1)      # [B,L,Q,1,4]
            qry = np.stack(qry_l, 1)      # [B,L,Q,C]

            for bi, img_id in enumerate(ids):
                if img_id not in needed:
                    continue
                hit_imgs += 1
                ow, oh = [int(v) for v in (targets[bi]["orig_size"].tolist()
                                           if torch.is_tensor(targets[bi]["orig_size"])
                                           else targets[bi]["orig_size"])]
                # all 300 boxes in pixels
                box_px_all = cxcywh_to_xyxy_px(box_np[bi], ow, oh)      # [Q,4]
                # per-layer reference boxes in pixels
                ref_px = cxcywh_to_xyxy_px(ref[bi].reshape(L, -1, 4), ow, oh) \
                    .reshape(L, -1, 4)                                   # [L,Q,4]
                loc_px = loc[bi].copy()                 # [L,Q,H,P,2]  (bi already applied)
                loc_px[..., 0] *= ow
                loc_px[..., 1] *= oh

                for k in by_img[img_id]:
                    d = lesions[k]
                    c = d["gt_class"]
                    Bx = np.array([d["x0"], d["y0"], d["x1"], d["y1"]], np.float32)
                    diag = float(np.hypot(Bx[2] - Bx[0], Bx[3] - Bx[1])) + 1e-6
                    ctr = np.array([(Bx[0] + Bx[2]) / 2, (Bx[1] + Bx[3]) / 2], np.float32)

                    iou_q = xyxy_iou_matrix(box_px_all, Bx[None, :])[:, 0]
                    cand = np.where((label_np[bi] == c) & (iou_q >= 0.5))[0]
                    A["n_cand"][k] = len(cand)
                    qs = int(cand[np.argmax(score_np[bi][cand])]) if len(cand) else -1
                    A["found"][k] = 1 if len(cand) else 0
                    A["q_star"][k] = qs
                    if qs >= 0:
                        A["score_q"][k] = score_np[bi][qs]
                        A["iou_final"][k] = iou_q[qs]

                    # Controls are recorded for EVERY lesion, including the ones no
                    # query localises (found == 0). That is the point for hard_floor:
                    # if the image's top-1 query samples the lesion heavily yet scores
                    # it ~0 the failure is in the decision, and when an arm has no
                    # responsible query at all the controls are its only internal
                    # evidence.
                    rng = random.Random(args.seed * 1000003 + k)
                    q_r = rng.randrange(score_np.shape[1])
                    q_t1 = int(np.argmax(score_np[bi]))

                    for l in range(L):
                        if qs >= 0:
                            rb = ref_px[l, qs]
                            A["iou_ref"][k, l] = xyxy_iou_matrix(rb[None, :], Bx[None, :])[0, 0]
                            A["qemb_delta"][k, l] = float(
                                np.linalg.norm(qry[bi, l, qs]) if l == 0
                                else np.linalg.norm(qry[bi, l, qs] - qry[bi, l - 1, qs]))
                            A["qemb_norm"][k, l] = float(np.linalg.norm(qry[bi, l, qs]))
                            lg = logits_by_layer.get(l)
                            if lg is not None:
                                A["logit_c"][k, l] = float(lg.numpy()[bi, qs, c])
                            w_star = attn[bi, l, qs]
                            pr = np.clip(w_star, 1e-9, 1)
                            A["attn_entropy"][k, l] = float(
                                (-(pr * np.log(pr)).sum(-1) / np.log(w_star.shape[-1])).mean())
                            A["attn_max"][k, l] = float(w_star.max(-1).mean())

                        # ALL 300 queries: does the arm look at this lesion at all, and
                        # what does the best-covering query score? This is the routing
                        # analogue of the perception ceiling and the only internal
                        # signal that exists for hard_floor lesions.
                        #
                        # The three _iou / _score fields below are what makes the
                        # routing-vs-decision claim falsifiable: if the query that
                        # SAMPLES the lesion best also REGRESSES a good box, then the
                        # only thing missing is the class/score decision; if its box is
                        # bad too, the regression head is also implicated.
                        lp_all = loc_px[l]                                    # [Q,H,P,2]
                        inbox_all = pts_in_box(lp_all, Bx)                    # [Q,H,P]
                        frac_all = inbox_all.mean(-1).mean(-1)                # [Q]
                        q_best = int(frac_all.argmax())
                        A["any_inbox_max"][k, l] = float(frac_all[q_best])
                        A["any_mass_max"][k, l] = float(
                            (attn[bi, l] * inbox_all).sum(-1).sum(-1).max())
                        A["any_inbox_q_score"][k, l] = float(score_np[bi][q_best])
                        A["any_inbox_q_iou"][k, l] = float(iou_q[q_best])
                        A["any_inbox_q_label"][k, l] = int(label_np[bi][q_best])
                        A["max_iou_all"][k, l] = float(iou_q.max())
                        A["max_iou_cls"][k, l] = float(
                            iou_q[label_np[bi] == c].max() if (label_np[bi] == c).any() else 0.0)

                        for qq, bucket in ((q_r, "rand"), (q_t1, "top1")):
                            lp = loc_px[l, qq]
                            inbox = pts_in_box(lp, Bx)
                            ww = attn[bi, l, qq]
                            im = float(inbox.mean())
                            ms = float((ww * inbox).sum())
                            if bucket == "rand":
                                A["ctl_rand_inbox"][k, l] = im
                                A["ctl_rand_mass"][k, l] = ms
                            else:
                                A["ctl_top1_inbox"][k, l] = im
                                A["ctl_top1_mass"][k, l] = ms

                        if qs < 0:
                            continue
                        lp = loc_px[l, qs]
                        inbox = pts_in_box(lp, Bx)
                        valid = (lp[..., 0] >= 0) & (lp[..., 0] <= ow) & \
                                (lp[..., 1] >= 0) & (lp[..., 1] <= oh)
                        A["samp_inbox"][k, l] = float(inbox.mean())
                        A["samp_mass_inbox"][k, l] = float((w_star * inbox).sum())
                        A["samp_valid"][k, l] = float(valid.mean())
                        A["samp_dist"][k, l] = float(
                            (np.linalg.norm(lp - ctr, axis=-1) / diag).mean())
                        for lvl in range(len(npl)):
                            m = level_of_point == lvl
                            A["samp_inbox_lvl"][k, l, lvl] = float(inbox[:, m].mean())
                        A["samp_inbox_pts"][k, l] = inbox.astype(np.float32)
                        A["samp_loc"][k, l] = lp.astype(np.float16)
                        A["attn_w"][k, l] = w_star.astype(np.float16)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (EC_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(A)
    payload["lesion_ann_id"] = np.asarray([d["ann_id"] for d in lesions], np.int64)
    payload["lesion_image_id"] = np.asarray([d["image_id"] for d in lesions], np.int64)
    payload["lesion_cell"] = np.asarray([d["cell"] for d in lesions], dtype=object)
    payload["lesion_cls_name"] = np.asarray([d["cls_name"] for d in lesions], dtype=object)
    payload["lesion_class"] = np.asarray([d["gt_class"] for d in lesions], np.int8)
    np.savez_compressed(out_path, **payload)

    meta = dict(arm=args.arm, config=args.config, checkpoint=args.checkpoint,
                split=args.split_name, weights=args.weights, seed=args.seed,
                n_lesions=n_les, n_found=int(A["found"].sum()),
                n_images_scanned=seen_imgs, n_images_hit=hit_imgs,
                layers=L, num_heads=H, points_per_head=P, num_points_list=npl,
                offset_scale=float(msa.offset_scale), eval_idx=int(td.eval_idx),
                score_floor=0.0, seconds=round(time.time() - t0, 1))
    with open(str(out_path).replace(".npz", ".meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"[done] {args.arm} {args.split_name}: {int(A['found'].sum())}/{n_les} "
          f"lesions localised; {hit_imgs}/{seen_imgs} images scanned; "
          f"{time.time() - t0:.0f}s -> {out_path}", flush=True)
    return 0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="?")
    ap.add_argument("--config")
    ap.add_argument("--checkpoint")
    ap.add_argument("--ann-file", required=True)
    ap.add_argument("--img-folder")
    ap.add_argument("--split-name", default="test")
    ap.add_argument("--weights", default="ema")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    # lesion-list mode
    ap.add_argument("--dump-lesions", action="store_true")
    ap.add_argument("--score-ranking", default=None)
    ap.add_argument("--lesions-out", default=None)
    ap.add_argument("--lesions", default=None)
    ap.add_argument("--per-cell", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decoder-eval-mode", action="store_true",
                    help="keep the decoder in eval (only the last layer is then "
                         "observable; per-layer logits become unavailable)")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.dump_lesions:
        return dump_lesions(args)
    for req in ("config", "checkpoint", "img_folder", "lesions", "out"):
        if getattr(args, req) in (None, ""):
            raise SystemExit(f"--{req.replace('_','-')} is required for a run")
    return run_arm(args)


if __name__ == "__main__":
    sys.exit(main())
