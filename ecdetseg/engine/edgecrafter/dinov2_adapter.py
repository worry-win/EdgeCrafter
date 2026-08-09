"""Standard DINOv2-S adapter for the EdgeCrafter three-level backbone API."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from ..core import register
from .hybrid_encoder import ConvNormLayer_fuse


class RFChannelLayerNorm(nn.Module):
    """RF-DETR channel-wise LayerNorm for NCHW feature maps."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.normalized_shape = (channels,)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class RFConvX(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            padding=kernel_size // 2, bias=False,
        )
        self.norm = RFChannelLayerNorm(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x.contiguous())))


class RFBottleneck(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.cv1 = RFConvX(channels, channels, 3)
        self.cv2 = RFConvX(channels, channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv2(self.cv1(x))


class RFC2f(nn.Module):
    """RF-DETR/LW-DETR C2f projector block."""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 3) -> None:
        super().__init__()
        hidden_channels = out_channels // 2
        self.hidden_channels = hidden_channels
        self.cv1 = RFConvX(in_channels, 2 * hidden_channels, 1)
        self.blocks = nn.ModuleList(RFBottleneck(hidden_channels) for _ in range(num_blocks))
        self.cv2 = RFConvX((2 + num_blocks) * hidden_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = list(self.cv1(x).split((self.hidden_channels, self.hidden_channels), 1))
        outputs.extend(block(outputs[-1]) for block in self.blocks)
        return self.cv2(torch.cat(outputs, 1))


class RFMultiScaleProjector(nn.Module):
    """RF-DETR MultiScaleProjector, with configurable output scales."""

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int,
        scale_factors: tuple[float, ...] = (2.0, 1.0, 0.5),
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        sampling_stages = []
        output_stages = []
        for scale in scale_factors:
            sampling_layers = []
            for channels in in_channels:
                if scale == 2.0:
                    sampling_layers.append(nn.Sequential(nn.ConvTranspose2d(channels, channels // 2, 2, 2)))
                elif scale == 1.0:
                    sampling_layers.append(nn.Identity())
                elif scale == 0.5:
                    sampling_layers.append(nn.Sequential(RFConvX(channels, channels, 3, 2)))
                else:
                    raise NotImplementedError(f"Unsupported RF projector scale factor: {scale}")
            sampling_stages.append(nn.ModuleList(sampling_layers))
            fused_channels = sum(channels // max(1, scale) for channels in in_channels)
            output_stages.append(
                nn.Sequential(RFC2f(int(fused_channels), out_channels, num_blocks), RFChannelLayerNorm(out_channels))
            )
        self.sampling_stages = nn.ModuleList(sampling_stages)
        self.output_stages = nn.ModuleList(output_stages)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs = []
        for samplers, stage in zip(self.sampling_stages, self.output_stages):
            sampled = [sampler(feature) for sampler, feature in zip(samplers, features)]
            outputs.append(stage(torch.cat(sampled, dim=1)))
        return outputs


def _resize_pos_embed(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Resize DINOv2 patch positions, with or without a CLS prefix token."""
    prefix_tokens = None
    for prefix_count in (0, 1):
        source_count = source.shape[1] - prefix_count
        target_count = target.shape[1] - prefix_count
        source_side = int(math.sqrt(source_count))
        target_side = int(math.sqrt(target_count))
        if source_side * source_side == source_count and target_side * target_side == target_count:
            prefix_tokens = prefix_count
            break
    if prefix_tokens is None:
        raise ValueError(
            f"DINOv2 positional grids must be square, got {source.shape[1]} and {target.shape[1]} tokens."
        )
    source_side = int(math.sqrt(source.shape[1] - prefix_tokens))
    target_side = int(math.sqrt(target.shape[1] - prefix_tokens))
    prefix = source[:, :prefix_tokens]
    patch_positions = source[:, prefix_tokens:]
    grid = patch_positions.reshape(source.shape[0], source_side, source_side, source.shape[2]).permute(0, 3, 1, 2)
    grid = F.interpolate(grid.float(), size=(target_side, target_side), mode="bicubic", align_corners=False, antialias=True)
    resized = grid.permute(0, 2, 3, 1).reshape(source.shape[0], target_side * target_side, source.shape[2])
    return torch.cat((prefix, resized), dim=1).reshape_as(target).to(dtype=target.dtype)


def _resize_patch_projection(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.shape == target.shape:
        return source
    if source.ndim != 4 or source.shape[:2] != target.shape[:2]:
        raise ValueError(f"Cannot resize patch projection {tuple(source.shape)} to {tuple(target.shape)}.")
    resized = F.interpolate(
        source.reshape(source.shape[0] * source.shape[1], 1, source.shape[2], source.shape[3]),
        size=target.shape[-2:],
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return resized.reshape_as(target)


@register()
class DinoV2Adapter(nn.Module):
    """Load DINOv2-S and expose P3/P4/P5 features at strides 8/16/32.

    The official checkpoint is patch14. For the 640/patch16 experiment its patch
    projection and learned patch grid are bicubically resized to 16 and 40.
    Detector-side projections and all downstream modules remain random.
    """

    def __init__(
        self,
        weights_path: str,
        backbone_name: str = "vit_small_patch14_dinov2",
        img_size: int = 640,
        patch_size: int = 16,
        proj_dim: int = 256,
        interaction_indexes: list[int] | tuple[int, ...] = (10, 11),
        num_levels: int = 3,
        projector_type: str = "ec",
        rf_num_blocks: int = 3,
        rf_scale_factors: tuple[float, ...] = (2.0, 1.0, 0.5),
        skip_load_backbone: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if projector_type == "ec" and num_levels != 3:
            raise ValueError("The EC projection path exposes exactly three feature levels.")
        if patch_size != 16:
            raise ValueError("This baseline is defined for patch_size=16.")
        self.patch_size = patch_size
        self.interaction_indexes = tuple(int(i) for i in interaction_indexes)
        self.projector_type = projector_type
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            img_size=img_size,
            patch_size=patch_size,
            num_classes=0,
        )
        if not skip_load_backbone:
            self._load_weights(weights_path)
        elif not Path(weights_path).is_file():
            raise FileNotFoundError(f"DINOv2 checkpoint does not exist: {weights_path}")

        if projector_type == "ec":
            self.projector = nn.ModuleList(
                [ConvNormLayer_fuse(self.backbone.embed_dim, proj_dim, kernel_size=1, stride=1) for _ in range(num_levels)]
            )
        elif projector_type == "rf":
            self.projector = RFMultiScaleProjector(
                [self.backbone.embed_dim] * len(self.interaction_indexes), proj_dim,
                scale_factors=tuple(float(v) for v in rf_scale_factors), num_blocks=rf_num_blocks
            )
        else:
            raise ValueError(f"Unknown projector_type={projector_type!r}; expected 'ec' or 'rf'.")

    def _load_weights(self, weights_path: str) -> None:
        path = Path(weights_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"DINOv2 checkpoint does not exist: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state, dict):
            raise TypeError(f"Expected a state dictionary in {path}, got {type(state).__name__}.")
        state = {
            key.removeprefix("module.").removeprefix("backbone."): value
            for key, value in state.items()
            if isinstance(value, torch.Tensor) and not key.startswith("head.")
        }
        # The official no-register checkpoint contains a pretraining-only mask
        # token that is not part of the timm inference backbone.
        state.pop("mask_token", None)
        target = self.backbone.state_dict()
        for key in ("patch_embed.proj.weight", "pos_embed"):
            if key not in state:
                raise KeyError(f"DINOv2 checkpoint is missing required key {key!r}: {path}")
        state["patch_embed.proj.weight"] = _resize_patch_projection(
            state["patch_embed.proj.weight"], target["patch_embed.proj.weight"]
        )
        state["pos_embed"] = _resize_pos_embed(state["pos_embed"], target["pos_embed"])
        incompatible = self.backbone.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"DINOv2 checkpoint mismatch for {path}: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        elements = sum(value.numel() for value in state.values())
        print(f"Loaded DINOv2-S backbone: {path} ({len(state)} tensors, {elements} elements; patch14 -> patch16)", flush=True)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = self.backbone.get_intermediate_layers(
            x,
            n=list(self.interaction_indexes),
            reshape=True,
            return_prefix_tokens=False,
            norm=True,
        )
        if len(features) == 0:
            raise RuntimeError("DINOv2 returned no intermediate features.")
        if self.projector_type == "rf":
            return self.projector(list(features))

        fused = torch.stack(features, dim=0).mean(dim=0)
        height, width = fused.shape[-2:]
        outputs: list[torch.Tensor] = []
        for index, scale in enumerate((2.0, 1.0, 0.5)):
            size = (max(1, round(height * scale)), max(1, round(width * scale)))
            feature = F.interpolate(fused, size=size, mode="bilinear", align_corners=False)
            outputs.append(self.projector[index](feature))
        return outputs


__all__ = ["DinoV2Adapter"]
