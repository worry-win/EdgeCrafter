#!/usr/bin/env python3
"""Create a target-shaped fine-tuning checkpoint from a complete detector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import torch


CLASS_SPECIFIC_PREFIXES = (
    "decoder.denoising_class_embed.",
    "decoder.enc_score_head.",
    "decoder.dec_score_head.",
)

OUTPUT_INITIALIZER_PREFIXES = (
    "decoder.enc_bbox_head.layers.2.",
    "decoder.pre_bbox_head.layers.2.",
)


def is_output_initializer(key: str) -> bool:
    return key.startswith(OUTPUT_INITIALIZER_PREFIXES) or (
        key.startswith("decoder.dec_bbox_head.") and ".layers.2." in key
    )


def merge_full_detector_state(
    target_state: Mapping[str, torch.Tensor],
    source_state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Transfer compatible detector tensors while preserving target output layers."""
    merged = {key: value.detach().cpu() for key, value in target_state.items()}
    transferred: list[str] = []
    preserved_class_specific: list[str] = []
    preserved_output_initializers: list[str] = []
    missing: list[str] = []
    shape_mismatch: list[str] = []

    for key, target_value in target_state.items():
        if key.startswith(CLASS_SPECIFIC_PREFIXES):
            preserved_class_specific.append(key)
            continue
        if is_output_initializer(key):
            preserved_output_initializers.append(key)
            continue
        source_value = source_state.get(key)
        if source_value is None:
            missing.append(key)
            continue
        if source_value.shape != target_value.shape:
            shape_mismatch.append(key)
            continue
        merged[key] = source_value.detach().cpu()
        transferred.append(key)

    audit = {
        "transferred": transferred,
        "transferred_by_family": {
            family: sum(key.startswith(f"{family}.") for key in transferred)
            for family in ("backbone", "encoder", "decoder")
        },
        "preserved_class_specific": preserved_class_specific,
        "preserved_output_initializers": preserved_output_initializers,
        "missing": missing,
        "shape_mismatch": shape_mismatch,
    }
    return merged, audit


def checkpoint_state(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dictionary, got {type(checkpoint).__name__}")
    ema = checkpoint.get("ema")
    if isinstance(ema, dict) and isinstance(ema.get("module"), dict):
        return ema["module"]
    model = checkpoint.get("model")
    if isinstance(model, dict):
        return model
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
    source_state = checkpoint_state(source_checkpoint)
    merged, audit = merge_full_detector_state(target_state, source_state)
    model.load_state_dict(merged, strict=True)

    required_families = ("backbone", "encoder", "decoder")
    if any(audit["transferred_by_family"][family] == 0 for family in required_families):
        raise RuntimeError(f"Incomplete full detector transfer: {audit['transferred_by_family']}")
    if not audit["preserved_class_specific"]:
        raise RuntimeError("No target-specific class tensors were preserved")
    if not audit["preserved_output_initializers"]:
        raise RuntimeError("No target box/distribution output initializers were preserved")
    if audit["shape_mismatch"]:
        raise RuntimeError(f"Unexpected non-class shape mismatches: {audit['shape_mismatch']}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": merged,
        "transfer_audit": audit,
        "source_checkpoint": str(Path(args.source).resolve()),
        "target_config": str(Path(args.config).resolve()),
    }, output)
    print(json.dumps({
        "output": str(output),
        "transferred_by_family": audit["transferred_by_family"],
        "transferred_total": len(audit["transferred"]),
        "preserved_class_specific": len(audit["preserved_class_specific"]),
        "preserved_output_initializers": len(audit["preserved_output_initializers"]),
        "missing": audit["missing"],
        "shape_mismatch": audit["shape_mismatch"],
    }, indent=2))


if __name__ == "__main__":
    main()
