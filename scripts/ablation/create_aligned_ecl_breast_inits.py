#!/usr/bin/env python3
"""Create target-shaped breast initializers with identical EC-L detector-side state."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


ALIGNED_FAMILIES = ("encoder", "decoder")
CLASS_SPECIFIC_PREFIXES = (
    "decoder.denoising_class_embed.",
    "decoder.enc_score_head.",
    "decoder.dec_score_head.",
)
OUTPUT_INITIALIZER_PREFIXES = (
    "decoder.enc_bbox_head.layers.2.",
    "decoder.pre_bbox_head.layers.2.",
)


def _is_output_initializer(key: str) -> bool:
    return key.startswith(OUTPUT_INITIALIZER_PREFIXES) or (
        key.startswith("decoder.dec_bbox_head.") and ".layers.2." in key
    )


def checkpoint_state(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dictionary, got {type(checkpoint).__name__}")
    ema = checkpoint.get("ema")
    if isinstance(ema, dict) and isinstance(ema.get("module"), dict):
        return ema["module"]
    model = checkpoint.get("model")
    if isinstance(model, dict):
        return model
    if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint
    raise KeyError("Checkpoint contains neither ema.module nor model state")


def build_canonical_state(
    target_state: Mapping[str, torch.Tensor],
    source_state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Build one 4-class EC-L state: source decoder body plus target output heads."""
    merged = {key: value.detach().cpu() for key, value in target_state.items()}
    transferred: list[str] = []
    preserved: list[str] = []
    missing: list[str] = []
    shape_mismatch: list[str] = []

    for key, target_value in target_state.items():
        if not key.startswith("decoder."):
            continue
        if key.startswith(CLASS_SPECIFIC_PREFIXES) or _is_output_initializer(key):
            preserved.append(key)
            continue
        source_value = source_state.get(key)
        if source_value is None:
            missing.append(key)
        elif source_value.shape != target_value.shape:
            shape_mismatch.append(key)
        else:
            merged[key] = source_value.detach().cpu()
            transferred.append(key)

    return merged, {
        "transferred_decoder_body": transferred,
        "preserved_target_outputs": preserved,
        "missing": missing,
        "shape_mismatch": shape_mismatch,
    }


def assemble_target_state(
    target_state: Mapping[str, torch.Tensor],
    canonical_state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Keep the target backbone and replace every encoder/decoder tensor canonically."""
    merged = {key: value.detach().cpu() for key, value in target_state.items()}
    aligned = {family: 0 for family in ALIGNED_FAMILIES}
    missing: list[str] = []
    shape_mismatch: list[str] = []

    for key, target_value in target_state.items():
        family = next((name for name in ALIGNED_FAMILIES if key.startswith(f"{name}.")), None)
        if family is None:
            continue
        canonical_value = canonical_state.get(key)
        if canonical_value is None:
            missing.append(key)
        elif canonical_value.shape != target_value.shape:
            shape_mismatch.append(key)
        else:
            merged[key] = canonical_value.detach().cpu()
            aligned[family] += 1

    return merged, {
        "aligned_by_family": aligned,
        "missing": missing,
        "shape_mismatch": shape_mismatch,
    }


def overlay_source_family(
    state: dict[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    source_state: Mapping[str, torch.Tensor],
    family: str,
) -> dict[str, object]:
    prefix = f"{family}."
    copied: list[str] = []
    missing: list[str] = []
    shape_mismatch: list[str] = []
    for key, target_value in target_state.items():
        if not key.startswith(prefix):
            continue
        source_value = source_state.get(key)
        if source_value is None:
            missing.append(key)
        elif source_value.shape != target_value.shape:
            shape_mismatch.append(key)
        else:
            state[key] = source_value.detach().cpu()
            copied.append(key)
    return {"copied": copied, "missing": missing, "shape_mismatch": shape_mismatch}


def family_sha256(state: Mapping[str, torch.Tensor], families: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    prefixes = tuple(f"{family}." for family in families)
    for key in sorted(key for key in state if key.startswith(prefixes)):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _parse_target(value: str) -> tuple[str, Path, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("target must be TAG=CONFIG=OUTPUT")
    return parts[0], Path(parts[1]).resolve(), Path(parts[2]).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--canonical-config", required=True, type=Path)
    parser.add_argument("--target", action="append", required=True, type=_parse_target)
    parser.add_argument("--source-backbone-tag", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository_root))
    from ecdetseg.engine.core import YAMLConfig

    torch.set_num_threads(1)
    source_path = args.source.resolve()
    source_checkpoint = torch.load(source_path, map_location="cpu", weights_only=True, mmap=True)
    source_state = checkpoint_state(source_checkpoint)

    _set_seed(args.seed)
    canonical_cfg = YAMLConfig(str(args.canonical_config.resolve()))
    canonical_model = canonical_cfg.model.cpu().eval()
    canonical_state, canonical_audit = build_canonical_state(canonical_model.state_dict(), source_state)
    if canonical_audit["missing"] or canonical_audit["shape_mismatch"]:
        raise RuntimeError(f"Canonical decoder transfer failed: {canonical_audit}")
    canonical_model.load_state_dict(canonical_state, strict=True)
    detector_sha = family_sha256(canonical_state, ALIGNED_FAMILIES)
    decoder_sha = family_sha256(canonical_state, ("decoder",))
    del canonical_model, canonical_cfg
    gc.collect()

    summaries = []
    source_backbone_tags = set(args.source_backbone_tag)
    for tag, config_path, output_path in args.target:
        _set_seed(args.seed)
        cfg = YAMLConfig(str(config_path))
        model = cfg.model.cpu().eval()
        target_state = model.state_dict()
        merged, alignment_audit = assemble_target_state(target_state, canonical_state)
        if alignment_audit["missing"] or alignment_audit["shape_mismatch"]:
            raise RuntimeError(f"{tag}: detector-side alignment failed: {alignment_audit}")

        source_backbone_audit = None
        if tag in source_backbone_tags:
            source_backbone_audit = overlay_source_family(
                merged, target_state, source_state, "backbone"
            )
            if source_backbone_audit["missing"] or source_backbone_audit["shape_mismatch"]:
                raise RuntimeError(f"{tag}: source backbone transfer failed: {source_backbone_audit}")

        model.load_state_dict(merged, strict=True)
        if family_sha256(merged, ALIGNED_FAMILIES) != detector_sha:
            raise RuntimeError(f"{tag}: detector-side hash differs from canonical state")
        if family_sha256(merged, ("decoder",)) != decoder_sha:
            raise RuntimeError(f"{tag}: decoder hash differs from canonical state")

        audit = {
            "tag": tag,
            "seed": args.seed,
            "source_checkpoint": str(source_path),
            "canonical_config": str(args.canonical_config.resolve()),
            "target_config": str(config_path),
            "canonical_decoder_audit": canonical_audit,
            "alignment_audit": alignment_audit,
            "source_backbone_audit": source_backbone_audit,
            "detector_side_sha256": detector_sha,
            "decoder_sha256": decoder_sha,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": merged, "alignment_audit": audit}, output_path)
        summaries.append({
            "tag": tag,
            "output": str(output_path),
            "aligned_by_family": alignment_audit["aligned_by_family"],
            "source_backbone_tensors": (
                len(source_backbone_audit["copied"]) if source_backbone_audit else 0
            ),
            "detector_side_sha256": detector_sha,
            "decoder_sha256": decoder_sha,
        })
        del model, cfg, target_state, merged
        gc.collect()

    print(json.dumps({
        "source": str(source_path),
        "canonical_decoder_body_tensors": len(canonical_audit["transferred_decoder_body"]),
        "canonical_preserved_output_tensors": len(canonical_audit["preserved_target_outputs"]),
        "targets": summaries,
    }, indent=2))


if __name__ == "__main__":
    main()
