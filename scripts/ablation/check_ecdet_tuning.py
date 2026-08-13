#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ECDETSEG_ROOT = REPO_ROOT / "ecdetseg"
sys.path.insert(0, str(ECDETSEG_ROOT))

from engine.core import YAMLConfig  # noqa: E402


FEATAUG_PREFIX = "_feataug_crop_head."


def _is_class_head(name: str) -> bool:
    return name == "decoder.denoising_class_embed.weight" or name.startswith(
        ("decoder.enc_score_head.", "decoder.dec_score_head.")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--require-feataug", action="store_true")
    args = parser.parse_args()

    cfg = YAMLConfig(args.config)
    cfg.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True
    target = cfg.model.state_dict()

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    source = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]

    target_feataug = sorted(k for k in target if k.startswith(FEATAUG_PREFIX))
    source_feataug = sorted(k for k in source if k.startswith(FEATAUG_PREFIX))
    if args.require_feataug and not source_feataug:
        print("FeatAug is required but the checkpoint has no FeatAug tensors.", file=sys.stderr)
        return 1
    if args.require_feataug and not target_feataug:
        print(
            "FeatAug is required but the configured model exposes no "
            f"{FEATAUG_PREFIX} tensors.",
            file=sys.stderr,
        )
        return 1

    matched = [
        name
        for name, tensor in target.items()
        if name in source and tensor.shape == source[name].shape
    ]
    shape_mismatches = [
        name
        for name, tensor in target.items()
        if name in source and tensor.shape != source[name].shape
    ]
    missing = [name for name in target if name not in source]
    invalid_mismatches = [name for name in shape_mismatches if not _is_class_head(name)]

    if missing or invalid_mismatches:
        print(f"Missing target tensors: {missing}", file=sys.stderr)
        print(f"Unexpected shape mismatches: {invalid_mismatches}", file=sys.stderr)
        return 1

    print(
        f"matched={len(matched)} "
        f"class_head_reinitialized={len(shape_mismatches)} "
        f"target_feataug={len(target_feataug)} "
        f"checkpoint_feataug={len(source_feataug)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
