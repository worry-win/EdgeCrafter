"""One-off shape probe: what exactly do backbone / neck emit for each cmp5L arm?

Used only to design the stage-similarity and decoder-swap diagnostics. Prints, for
one image, the shapes of every tensor returned by model.backbone(x), by
model.encoder(...) and by model.decoder(...). No files are written.

    srun -p debug -w cu04 --gres=gpu:1 python scripts/ablation/probe_cmp5L_shapes.py \
        --config ecdetseg/configs/ecdet/ecdet_l_dinov2s_breast_cmp5L_363_100e_es20.yml \
        --checkpoint outputs/ablation/.../best.pth
"""

import argparse
import sys
from pathlib import Path

import torch

EC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EC_ROOT / "ecdetseg"))

from engine.core import YAMLConfig          # noqa: E402
from engine.solver import TASKS             # noqa: E402
from engine.misc import dist_utils          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ann-file", required=True)
    ap.add_argument("--img-folder", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--weights", default="ema")
    args = ap.parse_args()

    # Shapes are device independent, so this probe deliberately tolerates CPU:
    # it is meant to be runnable on the login node while the GPUs are queued.
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA unavailable, falling back to CPU (fine: shapes only)",
              file=sys.stderr)
        args.device = "cpu"

    cfg = YAMLConfig(args.config, **{
        "val_dataloader": {
            "dataset": {"ann_file": args.ann_file, "img_folder": args.img_folder},
            "num_workers": 0, "total_batch_size": 1, "shuffle": False,
            "drop_last": False,
        },
        "num_classes": 4, "remap_mscoco_category": False,
    })
    for bn in ("ViTAdapter", "DinoV2Adapter"):
        if bn in cfg.yaml_cfg:
            cfg.yaml_cfg[bn]["skip_load_backbone"] = True

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver._setup()
    solver.load_resume_state(args.checkpoint)
    module = solver.ema.module if (args.weights == "ema" and solver.ema is not None) else solver.model
    module = dist_utils.de_parallel(module)
    module.eval()

    loader = cfg.val_dataloader
    samples, targets = next(iter(loader))
    x = samples.to(torch.device(args.device))

    def show(tag, obj):
        if isinstance(obj, (list, tuple)):
            for i, t in enumerate(obj):
                if torch.is_tensor(t):
                    print(f"  {tag}[{i}] shape={tuple(t.shape)} dtype={t.dtype}")
                else:
                    print(f"  {tag}[{i}] {type(t).__name__}")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if torch.is_tensor(v):
                    print(f"  {tag}['{k}'] shape={tuple(v.shape)}")
                else:
                    print(f"  {tag}['{k}'] {type(v).__name__}")
        elif torch.is_tensor(obj):
            print(f"  {tag} shape={tuple(obj.shape)}")
        else:
            print(f"  {tag} {type(obj).__name__}")

    with torch.no_grad():
        feats_b = module.backbone(x)
        print("[backbone output]")
        show("backbone", feats_b)

        feats_z = module.encoder(feats_b)
        print("[neck output]")
        show("neck", feats_z)

        out = module.decoder(feats_z, None)
        print("[decoder output]")
        show("decoder", out)

    print("[top-level modules]")
    for n, m in module.named_children():
        print(f"  {n}: {type(m).__name__}")
    print(f"[targets keys] {list(targets.keys())}")
    for k, v in targets.items():
        if torch.is_tensor(v):
            print(f"  targets['{k}'] shape={tuple(v.shape)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
