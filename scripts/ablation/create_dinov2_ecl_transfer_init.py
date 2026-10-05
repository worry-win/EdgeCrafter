#!/usr/bin/env python3
"""Build a DINOv2 + ECDet-L decoder-body transfer checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import torch


TRANSFER_PREFIXES = ("decoder.",)
CLASS_SPECIFIC_PREFIXES = (
    "decoder.denoising_class_embed.",
    "decoder.enc_score_head.",
    "decoder.dec_score_head.",
)

# The source detector's final box/FDR layers are calibrated to its original
# backbone feature distribution.  Keep the target model's neutral (zero)
# initialization at this interface while still transferring the inner MLPs.
OUTPUT_INITIALIZER_PREFIXES = (
    "decoder.enc_bbox_head.layers.2.",
    "decoder.pre_bbox_head.layers.2.",
)


def _is_output_initializer(key: str) -> bool:
    return key.startswith(OUTPUT_INITIALIZER_PREFIXES) or (
        key.startswith("decoder.dec_bbox_head.") and ".layers.2." in key
    )


def merge_transfer_state(
    target_state: Mapping[str, torch.Tensor],
    source_state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Overlay the class-agnostic EC-L decoder body onto a DINO target."""
    merged = {key: value.detach().cpu() for key, value in target_state.items()}
    transferred: list[str] = []
    shape_mismatch: list[str] = []
    excluded_class_specific: list[str] = []
    preserved_output_initializers: list[str] = []
    source_missing: list[str] = []

    for key, target_value in target_state.items():
        if not key.startswith(TRANSFER_PREFIXES):
            continue
        if key.startswith(CLASS_SPECIFIC_PREFIXES):
            excluded_class_specific.append(key)
            continue
        if _is_output_initializer(key):
            preserved_output_initializers.append(key)
            continue
        source_value = source_state.get(key)
        if source_value is None:
            source_missing.append(key)
            continue
        if source_value.shape != target_value.shape:
            shape_mismatch.append(key)
            continue
        merged[key] = source_value.detach().cpu()
        transferred.append(key)

    audit = {
        "transferred": transferred,
        "transferred_by_family": {
            "encoder": sum(key.startswith("encoder.") for key in transferred),
            "decoder": sum(key.startswith("decoder.") for key in transferred),
        },
        "shape_mismatch": shape_mismatch,
        "excluded_class_specific": excluded_class_specific,
        "preserved_output_initializers": preserved_output_initializers,
        "source_missing": source_missing,
        "preserved_backbone": sum(key.startswith("backbone.") for key in target_state),
    }
    return merged, audit


def _checkpoint_state(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dictionary, got {type(checkpoint).__name__}")
    if "ema" in checkpoint:
        return checkpoint["ema"]["module"]
    if "model" in checkpoint:
        return checkpoint["model"]
    if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint
    raise KeyError("Checkpoint contains neither ema.module nor model state")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository_root))
    from ecdetseg.engine.core import YAMLConfig

    cfg = YAMLConfig(args.config)
    model = cfg.model.cpu().eval()
    target_state = model.state_dict()
    source_checkpoint = torch.load(args.source, map_location="cpu", weights_only=True, mmap=True)
    source_state = _checkpoint_state(source_checkpoint)
    merged, audit = merge_transfer_state(target_state, source_state)

    # A strict reload proves that the generated checkpoint is complete, while
    # these checks prove that the new-backbone neck stays intact and that the
    # transferable EC-L decoder body is actually present.
    model.load_state_dict(merged, strict=True)
    first_neck_key = "encoder.lateral_convs.0.conv.weight"
    if first_neck_key in audit["transferred"] or not torch.equal(merged[first_neck_key], target_state[first_neck_key].cpu()):
        raise RuntimeError(f"New-backbone neck must remain target-initialized: {first_neck_key}")
    if audit["transferred_by_family"]["encoder"] != 0 or audit["transferred_by_family"]["decoder"] == 0:
        raise RuntimeError(f"Incomplete EC-L transfer: {audit['transferred_by_family']}")
    if not audit["excluded_class_specific"]:
        raise RuntimeError("No class-specific target tensors were excluded")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": merged,
            "transfer_audit": audit,
            "source_checkpoint": str(Path(args.source).resolve()),
            "target_config": str(Path(args.config).resolve()),
        },
        output,
    )
    summary = {
        "output": str(output),
        "transferred_by_family": audit["transferred_by_family"],
        "shape_mismatch": audit["shape_mismatch"],
        "excluded_class_specific_count": len(audit["excluded_class_specific"]),
        "preserved_backbone_tensors": audit["preserved_backbone"],
        "preserved_first_neck_shape": list(merged[first_neck_key].shape),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
