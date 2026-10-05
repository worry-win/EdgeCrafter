#!/usr/bin/env python3
"""Build a 7-class Liver initialization from a 17-class joint detector.

The joint label order reserves rows 10..16 for the seven Liver classes.  This
tool copies those rows into a 7-class Liver ECDet state before normal tuning.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ecdetseg.engine.core.yaml_config import YAMLConfig


LIVER_SOURCE_IDS = tuple(range(10, 17))
CLASS_HEAD_PREFIXES = ("decoder.enc_score_head.", "decoder.dec_score_head.")
DN_EMBED_KEY = "decoder.denoising_class_embed.weight"


def _source_state(checkpoint: dict) -> dict[str, torch.Tensor]:
    if "ema" in checkpoint:
        return checkpoint["ema"]["module"]
    return checkpoint["model"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head-mode", choices=("liver_map", "random"), default="liver_map")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    source_checkpoint = torch.load(args.source, map_location="cpu", weights_only=True)
    source = _source_state(source_checkpoint)
    target = YAMLConfig(str(args.target_config)).model.state_dict()

    mapped = {key: value.clone() for key, value in target.items()}
    matched = []
    for key, target_tensor in target.items():
        source_tensor = source.get(key)
        if source_tensor is not None and source_tensor.shape == target_tensor.shape:
            mapped[key] = source_tensor.clone()
            matched.append(key)

    expected_heads = {
        "decoder.enc_score_head.weight", "decoder.enc_score_head.bias",
        "decoder.dec_score_head.0.weight", "decoder.dec_score_head.0.bias",
        "decoder.dec_score_head.1.weight", "decoder.dec_score_head.1.bias",
        "decoder.dec_score_head.2.weight", "decoder.dec_score_head.2.bias",
        "decoder.dec_score_head.3.weight", "decoder.dec_score_head.3.bias",
    }
    source_dn = source.get(DN_EMBED_KEY)
    target_dn = target.get(DN_EMBED_KEY)
    if source_dn is None or target_dn is None or source_dn.shape != (18, target_dn.shape[1]) or target_dn.shape[0] != 8:
        raise ValueError(f"Unexpected denoising embedding shape: {None if source_dn is None else source_dn.shape} -> {None if target_dn is None else target_dn.shape}")
    mapped_heads = []
    if args.head_mode == "liver_map":
        for key, target_tensor in target.items():
            if not key.startswith(CLASS_HEAD_PREFIXES):
                continue
            source_tensor = source.get(key)
            if source_tensor is None or source_tensor.shape[0] != 17 or target_tensor.shape[0] != 7:
                continue
            if source_tensor.shape[1:] != target_tensor.shape[1:]:
                raise ValueError(f"Unexpected class-head shape for {key}: {source_tensor.shape} -> {target_tensor.shape}")
            mapped[key] = source_tensor[list(LIVER_SOURCE_IDS)].clone()
            mapped_heads.append(key)
        mapped_dn = target_dn.clone()
        mapped_dn[:7] = source_dn[list(LIVER_SOURCE_IDS)]
        mapped_dn[7] = source_dn[17]  # joint padding 17 -> Liver padding 7
        mapped[DN_EMBED_KEY] = mapped_dn
        if set(mapped_heads) != expected_heads:
            raise RuntimeError(f"Mapped head set differs from expectation: {mapped_heads}")
        for key in mapped_heads:
            assert torch.equal(mapped[key], source[key][10:17]), key
        assert torch.equal(mapped[DN_EMBED_KEY][:7], source_dn[10:17])
        assert torch.equal(mapped[DN_EMBED_KEY][7], source_dn[17])
    else:
        random_head_keys = {
            key for key in target
            if key.startswith(CLASS_HEAD_PREFIXES) and key in source
            and source[key].shape[0] == 17 and target[key].shape[0] == 7
        }
        if random_head_keys != expected_heads:
            raise RuntimeError(f"Random head set differs from expectation: {sorted(random_head_keys)}")
        # `mapped` retains the seeded target initialization for every class
        # head and for the DN class embedding.

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ema": {"module": mapped},
            "joint_source": str(args.source),
            "liver_source_class_ids": list(LIVER_SOURCE_IDS),
            "head_mode": args.head_mode,
            "mapped_class_heads": mapped_heads,
            "matched_nonhead_tensors": len(matched),
        },
        args.output,
    )
    print(
        f"Saved {args.output}; mode={args.head_mode} nonhead_matched={len(matched)} "
        f"class_heads_mapped={len(mapped_heads)}"
    )


if __name__ == "__main__":
    main()
