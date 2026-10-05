"""Prepare a class-neutral Objects365 checkpoint for 17-class joint tuning."""

import os
import sys
from pathlib import Path

import torch


CLASS_PREFIXES = (
    "decoder.enc_score_head.",
    "decoder.dec_score_head.",
    "decoder.denoising_class_embed.",
)


def main(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must differ")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    keep = {key: value for key, value in state.items() if not key.startswith(CLASS_PREFIXES)}
    removed = set(state) - set(keep)
    if not removed or not all(any(key.startswith(prefix) for key in keep) for prefix in
                                  ("backbone.", "encoder.", "decoder.")):
        raise ValueError("Checkpoint lacks expected detector weights or class heads")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save({"model": keep}, temporary)
    os.replace(temporary, destination)
    print(f"Prepared {destination}: kept {len(keep)} tensors, excluded {len(removed)} class-head tensors")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(f"Usage: {sys.argv[0]} SOURCE.pth DESTINATION.pth")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
