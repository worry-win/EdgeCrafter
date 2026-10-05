"""Small, explicit contracts for the six-arm EC full-cycle experiment."""

from __future__ import annotations

from pathlib import Path
import math
import statistics

import numpy as np
import torch


def resolve_calibration_batch_path(manifest_path, record_path):
    path = Path(record_path)
    return path if path.is_absolute() else Path(manifest_path).resolve().parent / path.name


def calibration_coefficient_from_rank_ratios(ratios_by_rank):
    values = [float(value) for rank_values in ratios_by_rank for value in rank_values]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError('calibration requires finite positive gradient ratios')
    median = statistics.median(values)
    if median <= 1e-8:
        raise ValueError('calibration median gradient ratio is too small')
    return .30 / median, median


def pack_numpy_rng_state(state):
    return (state[0], state[1].tolist(), state[2], state[3], state[4])


def unpack_numpy_rng_state(state):
    return (state[0], np.asarray(state[1], dtype=np.uint32), state[2], state[3], state[4])


def stage_ramp(epoch: int, batch_index: int, num_batches: int, epochs: int) -> float:
    if num_batches <= 0 or epochs <= 0 or not 0 <= batch_index < num_batches:
        raise ValueError('invalid full-cycle progress')
    progress = (epoch + batch_index / num_batches) / epochs
    return max(0.0, min(1.0, (progress - .10) / .10))


ENCODER_KEYS = frozenset(f'loss_{name}_enc_0' for name in ('mal', 'bbox', 'giou'))
DECODER_KEYS = frozenset(
    [f'loss_{name}' for name in ('mal', 'bbox', 'giou', 'fgl')]
    + [f'loss_{name}_aux_{layer}' for layer in range(3)
       for name in ('mal', 'bbox', 'giou', 'fgl', 'ddf')]
    + [f'loss_{name}_pre' for name in ('mal', 'bbox', 'giou')]
    + [f'loss_{name}_dn_{layer}' for layer in range(4)
       for name in ('mal', 'bbox', 'giou', 'fgl', 'ddf')]
    + [f'loss_{name}_dn_pre' for name in ('mal', 'bbox', 'giou')]
)


def partition_detection_losses(losses: dict):
    unknown = set(losses) - ENCODER_KEYS - DECODER_KEYS
    if unknown:
        raise ValueError(f'unknown EC detection loss keys: {sorted(unknown)}')
    return (
        {key: value for key, value in losses.items() if key in ENCODER_KEYS},
        {key: value for key, value in losses.items() if key in DECODER_KEYS},
    )


def augmentation_mask(targets, height: int, width: int, outside_coefficients):
    """GT-expanded image-space union mask, resampled at each HE feature scale."""
    if len(targets) != len(outside_coefficients) or height < 1 or width < 1:
        raise ValueError('mask batch or feature shape mismatch')
    device = targets[0]['boxes'].device
    dtype = targets[0]['boxes'].dtype
    ys = (torch.arange(height, device=device, dtype=dtype) + .5) / height
    xs = (torch.arange(width, device=device, dtype=dtype) + .5) / width
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    masks = []
    for target, outside in zip(targets, outside_coefficients):
        boxes = target['boxes']
        if boxes.numel() == 0:
            masks.append(torch.ones((height, width), device=device, dtype=dtype))
            continue
        if boxes.ndim != 2 or boxes.shape[1] != 4 or not torch.isfinite(boxes).all():
            raise ValueError('invalid normalized GT boxes')
        inside = torch.zeros((height, width), device=device, dtype=torch.bool)
        for cx, cy, box_w, box_h in boxes:
            half_w = .6 * box_w
            half_h = .6 * box_h
            inside |= ((xx >= (cx - half_w).clamp(0, 1))
                       & (xx <= (cx + half_w).clamp(0, 1))
                       & (yy >= (cy - half_h).clamp(0, 1))
                       & (yy <= (cy + half_h).clamp(0, 1)))
        coefficient = torch.as_tensor(outside, device=device, dtype=dtype)
        if not bool((coefficient >= .3) & (coefficient <= .8)):
            raise ValueError('outside coefficient must be in [0.3,0.8]')
        masks.append(torch.where(inside, torch.ones_like(xx), coefficient))
    return torch.stack(masks, dim=0)[:, None]
