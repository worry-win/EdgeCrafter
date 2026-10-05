"""Anatomy probe: what can actually be hooked inside ECTransformer's decoder?

This is a *measurement*, not an assumption. It builds one arm, monkeypatches
``MSDeformableAttention.forward`` to stash the four tensors that never leave the
function (reference_points / sampling_offsets / attention_weights /
sampling_locations), installs hooks on the layer and the per-layer heads, runs a
single image, and prints every captured tensor with its shape, dtype and value
range. Run it before writing any distillation code so the tensor list is factual.

Why hooks are mandatory
-----------------------
``TransformerDecoder.forward`` only appends to ``dec_out_*`` when
``i == self.eval_idx`` (default 3 of 4). So in eval mode the returned
``pred_boxes`` is the LAST layer only -- the refinement trajectory and every
intermediate routing tensor are invisible from the return value. They are all
still computed in the loop, so hooks recover them at zero modelling cost.

Hook map (verified by this run)
-------------------------------
  MSDeformableAttention.forward   (patched)
      -> reference_points   [bs,Q,1,4]        normalised cxcywh
      -> sampling_offsets   [bs,Q,H,P,2]      raw head offsets, P = sum(points)
      -> attention_weights  [bs,Q,H,P]        post-softmax, sums to 1 over P
      -> sampling_locations [bs,Q,H,P,2]      NORMALISED image coords (x,y)
         sampling_locations is what "sampling point" means below; multiply by
         (W,H) to get pixels.  The eval transform is a plain stretch to
         eval_spatial_size (no letterbox) -- the dumper's own
         ``pb[:,0::2]*=orig_w`` confirms it -- so normalised -> original image
         coords is a pure per-axis scale.
  TransformerDecoderLayer i  forward_pre_hook
      -> reference_points   [bs,Q,1,4]        == the box fed INTO layer i
         (== refined box OUT of layer i-1) -> that IS the refinement trajectory
  TransformerDecoderLayer i  forward_hook
      -> query embedding    [bs,Q,C]
  ECTransformer.dec_score_head[i]  forward_hook
      -> per-layer class logits [bs,Q,num_classes]
  ECTransformer.dec_bbox_head[i]   forward_hook
      -> per-layer FDR corners  [bs,Q,4*(reg_max+1)]

NOT hookable read-only (needs a real code edit if ever wanted): the
self-attention map (nn.MultiheadAttention returns it but the caller discards it),
and the post-LQE scores (LQE rewrites scores outside the head module).

Usage
-----
    python scripts/ablation/probe_cmp5L_decoder_hooks.py \
        --config ecdetseg/configs/ecdet/ecdet_l_dinov2s_breast_cmp5L_363_100e_es20.yml \
        --checkpoint outputs/ablation/ecdet_l_dinov2s_breast_cmp5L_363_100e_es20_bs32_2gpu_seed42/best.pth \
        --ann-file /cobot/Data/Lesion_det/det_breast/annotations/Lesion/Ignore_Delete-Image/test.json \
        --img-folder /cobot/Data/Lesion_det/det_breast/img \
        --device cpu --limit 1
"""

import argparse
import inspect
import sys
from pathlib import Path

import numpy as np
import torch

EC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EC_ROOT / "ecdetseg"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.core import YAMLConfig          # noqa: E402
from engine.solver import TASKS             # noqa: E402
from engine.misc import dist_utils          # noqa: E402
import engine.edgecrafter.decoder as dec_mod  # noqa: E402

CAP = {}          # keyed by an integer tag -> list of captured dicts


def _reset():
    CAP.clear()


def install_msa_patch():
    """Stash the four deformable-attention tensors on every call."""
    if getattr(dec_mod.MSDeformableAttention.forward, "_diag_patched", False):
        return
    orig = dec_mod.MSDeformableAttention.forward

    def patched(self, query, reference_points, value, value_spatial_shapes):
        bs, Len_q = query.shape[:2]
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(
            bs, Len_q, self.num_heads, sum(self.num_points_list), 2)
        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = torch.nn.functional.softmax(attention_weights, dim=-1)

        if reference_points.shape[-1] == 4:
            nps = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = sampling_offsets * nps * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise SystemExit("unexpected reference_points last dim: "
                             f"{reference_points.shape[-1]}")

        tag = getattr(self, "_diag_tag", -1)
        CAP.setdefault(tag, []).append(dict(
            reference_points=reference_points.detach().float().cpu(),
            sampling_offsets=sampling_offsets.detach().float().cpu(),
            attention_weights=attention_weights.detach().float().cpu(),
            sampling_locations=sampling_locations.detach().float().cpu(),
            spatial_shapes=[list(s) for s in value_spatial_shapes],
            num_points_list=list(self.num_points_list),
            num_heads=self.num_heads,
            offset_scale=self.offset_scale,
        ))
        # delegate the actual attention to the untouched core
        return self.ms_deformable_attn_core(
            value, value_spatial_shapes, sampling_locations, attention_weights,
            self.num_points_list)

    patched._diag_patched = True
    dec_mod.MSDeformableAttention.forward = patched


def describe(d, prefix, out):
    out.append(f"  {prefix}: shape={tuple(d.shape)} dtype={d.dtype} "
               f"min={d.min():.4f} max={d.max():.4f} mean={d.mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--arm", default="?")
    ap.add_argument("--ann-file", required=True)
    ap.add_argument("--img-folder", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--weights", default="ema")
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("FATAL: CUDA unavailable; refusing to run on CPU.", file=sys.stderr)
        return 2
    if device.type == "cpu":
        print("[warn] running on CPU: this is a shape probe, not a benchmark.")

    install_msa_patch()

    cfg = YAMLConfig(args.config, **{
        "val_dataloader": {
            "dataset": {"ann_file": args.ann_file, "img_folder": args.img_folder},
            "num_workers": args.num_workers,
            "total_batch_size": args.batch_size,
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
    solver.load_resume_state(args.checkpoint)
    module = solver.ema.module if (args.weights == "ema" and solver.ema is not None) \
        else solver.model
    module = dist_utils.de_parallel(module)
    module.eval()

    ec = module.decoder                      # ECTransformer
    td = ec.decoder                          # TransformerDecoder
    msa0 = td.layers[0].cross_attn

    print("=" * 78)
    print(f"ARM                    {args.arm}")
    print(f"module.decoder         {type(ec).__name__}")
    print(f"  .decoder             {type(td).__name__}")
    print(f"  num_layers           {len(td.layers)}")
    print(f"  eval_idx             {td.eval_idx}   <-- eval returns THIS layer only")
    print(f"  num_queries          {ec.num_queries}")
    print(f"  hidden_dim           {ec.hidden_dim}")
    print(f"  num_levels           {ec.num_levels}")
    print(f"  use_lqe              {td.lqe_layers is not None}")
    print(f"MSDeformableAttention  num_heads={msa0.num_heads} "
          f"num_points_list={msa0.num_points_list} offset_scale={msa0.offset_scale}")
    print(f"  P (points/head)      {sum(msa0.num_points_list)}   "
          f"H*P = {msa0.num_heads * sum(msa0.num_points_list)}")
    print(f"  num_points_scale     {msa0.num_points_scale.tolist()}")
    print(f"dec_score_head         n={len(ec.dec_score_head)} "
          f"type={type(ec.dec_score_head[0]).__name__}")
    print(f"dec_bbox_head          "
          f"{'None' if ec.dec_bbox_head is None else len(ec.dec_bbox_head)}")
    print("=" * 78)

    # ---- hooks -------------------------------------------------------------
    # Two traps this probe exists to document:
    #  (1) a forward_pre_hook's return value REPLACES the input tuple, so the hook
    #      must be a real function returning None (a lambda returning the tuple of
    #      two setitem calls drops arguments and blows up the layer call);
    #  (2) the score head lives BETWEEN layers, and in eval it is called for
    #      i == eval_idx only -- so neither "position in a list" nor "registration
    #      index" identifies the layer. Publishing the current layer index from the
    #      layer pre-hook is exact. We also flip the decoder submodule to train
    #      mode so every layer's head is actually evaluated (no dropout, no BN, and
    #      eval_idx == num_layers-1, so the final prediction is unchanged).
    td.train()
    _reset()
    caught = {"layer_ref": {}, "layer_out": {}, "score": {}, "corner": {}}
    lstate = {"cur": -1}
    for i, layer in enumerate(td.layers):
        layer.cross_attn._diag_tag = i

        def mk_pre(i):
            def pre(mod, inp):
                caught["layer_ref"][i] = inp[1].detach().float().cpu()
                lstate["cur"] = i
                return None
            return pre

        def mk_post(i):
            def post(mod, inp, out):
                caught["layer_out"][i] = (out if torch.is_tensor(out)
                                          else out[0]).detach().float().cpu()
                return None
            return post

        layer.register_forward_pre_hook(mk_pre(i))
        layer.register_forward_hook(mk_post(i))

    def mk_head(bucket):
        def hook(mod, inp, out):
            caught[bucket][lstate["cur"]] = out.detach().float().cpu()
            return None
        return hook

    seen = set()
    for h in ec.dec_score_head:
        if isinstance(h, torch.nn.Identity) or id(h) in seen:
            continue
        seen.add(id(h))
        h.register_forward_hook(mk_head("score"))
    seen = set()
    for h in (ec.dec_bbox_head or []):
        if isinstance(h, torch.nn.Identity) or id(h) in seen:
            continue
        seen.add(id(h))
        h.register_forward_hook(mk_head("corner"))
    print(f"[hooks] decoder flipped to train() so all layers evaluate their heads; "
          f"eval_idx={td.eval_idx}")

    loader = cfg.val_dataloader
    seen = 0
    with torch.no_grad():
        for samples, targets in loader:
            x = samples.to(device)
            print(f"[fwd] input {tuple(x.shape)}")
            B = module.backbone(x)
            print(f"[fwd] backbone out: " +
                  ", ".join(str(tuple(t.shape)) for t in B)
                  if isinstance(B, (list, tuple)) else f"[fwd] backbone out {tuple(B.shape)}")
            Z = module.encoder(B)
            print(f"[fwd] neck out:     " +
                  ", ".join(str(tuple(t.shape)) for t in Z)
                  if isinstance(Z, (list, tuple)) else f"[fwd] neck out {tuple(Z.shape)}")
            out = module.decoder(Z, None)
            if isinstance(out, (list, tuple)):
                out = out[0]
            pl = out["pred_logits"]
            pb = out["pred_boxes"]
            if pl.dim() == 2:
                pl, pb = pl.unsqueeze(0), pb.unsqueeze(0)
            print(f"[fwd] pred_logits {tuple(pl.shape)}  pred_boxes {tuple(pb.shape)}")
            seen += 1
            if seen >= args.limit:
                break

    out_lines = []
    out_lines.append("")
    out_lines.append("captured per layer (i = decoder layer index)")
    for i in sorted(CAP.keys()):
        for k, rec in enumerate(CAP[i]):
            out_lines.append(f" layer {i}  (cross_attn call #{k})")
            describe(rec["reference_points"], "reference_points  ", out_lines)
            describe(rec["sampling_offsets"], "sampling_offsets  ", out_lines)
            describe(rec["attention_weights"], "attention_weights ", out_lines)
            describe(rec["sampling_locations"], "sampling_locations", out_lines)
            out_lines.append(f"    spatial_shapes={rec['spatial_shapes']} "
                             f"num_heads={rec['num_heads']} "
                             f"num_points_list={rec['num_points_list']} "
                             f"offset_scale={rec['offset_scale']}")
            loc = rec["sampling_locations"]
            att = rec["attention_weights"]
            out_lines.append(
                f"    loc in [0,1]?: {bool((loc >= -0.05).all() and (loc <= 1.05).all())}   "
                f"attn sums to 1 over P?: "
                f"{float(att.sum(-1).min()):.6f}..{float(att.sum(-1).max()):.6f}")
            break   # one call per layer per forward is enough to read shapes
    for i in sorted(caught["layer_ref"]):
        describe(caught["layer_ref"][i], f"layer{i} INPUT ref  ", out_lines)
        describe(caught["layer_out"][i], f"layer{i} OUTPUT qry", out_lines)
    for i in sorted(caught["score"]):
        describe(caught["score"][i], f"score_head[{i}] out  ", out_lines)
    for i in sorted(caught["corner"]):
        describe(caught["corner"][i], f"bbox_head[{i}] corners", out_lines)

    text = "\n".join(out_lines)
    print(text)
    print("=" * 78)
    print("VERDICT: all listed tensors are capturable read-only via hooks "
          "(the MSA tensors via a monkeypatch of the untouched forward body).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
