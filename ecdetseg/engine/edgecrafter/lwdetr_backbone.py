"""LW-DETR xlarge encoder and projector adapter for EdgeCrafter.

The encoder and MultiScaleProjector definitions preserve the official LW-DETR
xlarge topology and state-dict layout. Only the outer tensor-only interface is
adapted for EdgeCrafter.

LW-DETR Copyright (c) 2024 Baidu. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, Mlp, trunc_normal_

from ..core import register
from .hybrid_encoder import ConvNormLayer_fuse


def _get_abs_pos(abs_pos: torch.Tensor, has_cls_token: bool, hw: tuple[int, int]) -> torch.Tensor:
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    size = int(math.sqrt(abs_pos.shape[1]))
    if size * size != abs_pos.shape[1]:
        raise ValueError(f"LW positional embedding is not square: {abs_pos.shape[1]} tokens")
    height, width = hw
    if size != height or size != width:
        abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        return abs_pos
    return abs_pos.reshape(1, height, width, -1)


class LWPatchEmbed(nn.Module):
    def __init__(self, embed_dim: int = 768) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).permute(0, 2, 3, 1)


class LWAttention(nn.Module):
    def __init__(self, dim: int = 768, num_heads: int = 12) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(batch, tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = (query * self.scale) @ key.transpose(-2, -1)
        if mask is not None:
            attention.masked_fill_(mask.reshape(batch, 1, 1, tokens).expand_as(attention), float("-inf"))
        attention = attention.softmax(dim=-1)
        x = (attention @ value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(x)


class LWBlock(nn.Module):
    def __init__(self, window: bool, drop_path: float) -> None:
        super().__init__()
        dim = 768
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = LWAttention(dim=dim, num_heads=12)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=dim * 4, act_layer=nn.GELU)
        self.window = window
        self.gamma_1 = nn.Parameter(0.1 * torch.ones(dim))
        self.gamma_2 = nn.Parameter(0.1 * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_windows, tokens, channels = x.shape
        shortcut = x
        x = self.norm1(x)
        if not self.window:
            x = x.reshape(batch_windows // 16, 16 * tokens, channels)
        x = self.gamma_1 * self.attn(x)
        if not self.window:
            x = x.reshape(batch_windows, tokens, channels)
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))


class LWViTEncoder(nn.Module):
    """Official LW-DETR xlarge ViT-B encoder topology."""

    def __init__(
        self,
        window_block_indexes: tuple[int, ...],
        out_feature_indexes: tuple[int, ...] = (2, 4, 5, 9),
    ) -> None:
        super().__init__()
        depth = 10
        self.pretrain_use_cls_token = True
        self.patch_embed = LWPatchEmbed(embed_dim=768)
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, 768))
        drop_paths = [value.item() for value in torch.linspace(0, 0.1, depth)]
        self.window_block_indexes = tuple(int(index) for index in window_block_indexes)
        window_indexes = set(self.window_block_indexes)
        invalid_indexes = window_indexes - set(range(depth))
        if invalid_indexes:
            raise ValueError(f"Invalid LW window block indexes: {sorted(invalid_indexes)}")
        self.out_feature_indexes = tuple(int(index) for index in out_feature_indexes)
        if len(set(self.out_feature_indexes)) != len(self.out_feature_indexes):
            raise ValueError(f"Duplicate LW output feature indexes: {self.out_feature_indexes}")
        invalid_output_indexes = set(self.out_feature_indexes) - set(range(depth))
        if invalid_output_indexes:
            raise ValueError(f"Invalid LW output feature indexes: {sorted(invalid_output_indexes)}")
        self.blocks = nn.ModuleList(
            LWBlock(window=index in window_indexes, drop_path=drop_paths[index])
            for index in range(depth)
        )
        output_indexes = set(self.out_feature_indexes)
        self._out_features = [index in output_indexes for index in range(depth)]
        self._out_feature_channels = [768] * len(self.out_feature_indexes)
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.patch_embed(x)
        x = x + _get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, x.shape[1:3])
        batch, height, width, channels = x.shape
        if height % 4 != 0 or width % 4 != 0:
            raise ValueError(
                f"LW encoder patch grid must be divisible by 4, got {height}x{width}. "
                "Use an input size divisible by 64."
            )
        window_height, window_width = height // 4, width // 4
        x = x.reshape(batch, 4, window_height, 4, window_width, channels)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(batch * 16, window_height * window_width, channels)
        outputs = []
        for index, block in enumerate(self.blocks):
            x = block(x)
            if self._out_features[index]:
                feature = x.reshape(batch, 4, 4, window_height, window_width, channels)
                feature = feature.permute(0, 5, 1, 3, 2, 4).reshape(batch, channels, height, width)
                outputs.append(feature)
        return outputs


class LWChannelLayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class LWConvX(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int = 3,
        stride: int = 1,
        act: str = "relu",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=kernel // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act == "silu" else nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class LWBottleneck(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.cv1 = LWConvX(channels, channels, 3, 1, act="silu")
        self.cv2 = LWConvX(channels, channels, 3, 1, act="silu")
        self.add = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.cv2(self.cv1(x))
        return x + output if self.add else output


class LWC2f(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 3) -> None:
        super().__init__()
        self.c = int(out_channels * 0.5)
        self.cv1 = LWConvX(in_channels, 2 * self.c, 1, 1, act="silu")
        self.cv2 = LWConvX((2 + num_blocks) * self.c, out_channels, 1, 1, act="silu")
        self.m = nn.ModuleList(LWBottleneck(self.c) for _ in range(num_blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = list(self.cv1(x).split((self.c, self.c), 1))
        outputs.extend(block(outputs[-1]) for block in self.m)
        return self.cv2(torch.cat(outputs, 1))


class LWMultiScaleProjector(nn.Module):
    """Official LW-DETR xlarge P3/P5 MultiScaleProjector."""

    def __init__(self) -> None:
        super().__init__()
        in_channels = [768] * 4
        out_channels = 384
        stages_sampling = []
        stages = []
        for scale in (2.0, 0.5):
            sampling_layers = []
            for in_dim in in_channels:
                if scale == 2.0:
                    layers = [
                        LWConvX(in_dim, in_dim // 2, kernel=1),
                        nn.ConvTranspose2d(in_dim // 2, in_dim // 4, kernel_size=2, stride=2),
                    ]
                    sampled_channels = in_dim // 4
                else:
                    layers = [LWConvX(in_dim, in_dim, kernel=3, stride=2)]
                    sampled_channels = in_dim
                sampling_layers.append(nn.Sequential(*layers))
            stages_sampling.append(nn.ModuleList(sampling_layers))
            stages.append(
                nn.Sequential(
                    LWC2f(sampled_channels * len(in_channels), out_channels, num_blocks=3),
                    LWChannelLayerNorm(out_channels),
                )
            )
        self.stages_sampling = nn.ModuleList(stages_sampling)
        self.stages = nn.ModuleList(stages)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        results = []
        for stage_index, stage in enumerate(self.stages):
            sampled = [
                sampler(feature)
                for sampler, feature in zip(self.stages_sampling[stage_index], features)
            ]
            results.append(stage(torch.cat(sampled, dim=1)))
        return results


@register()
class LWDetrBackbone(nn.Module):
    """Expose the pretrained official LW-DETR xlarge encoder+neck to EC."""

    def __init__(
        self,
        weights_path: str | None,
        window_block_indexes: tuple[int, ...] | list[int] = (0, 1, 3, 6, 7, 9),
        out_feature_indexes: tuple[int, ...] | list[int] = (2, 4, 5, 9),
        projector_type: str = "official",
        proj_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = LWViTEncoder(tuple(window_block_indexes), tuple(out_feature_indexes))
        self.projector_type = projector_type
        if projector_type == "official":
            if tuple(out_feature_indexes) != (2, 4, 5, 9):
                raise ValueError("The official LW projector requires output blocks (2, 4, 5, 9).")
            self.projector = LWMultiScaleProjector()
        elif projector_type == "ec":
            self.projector = nn.ModuleList(
                ConvNormLayer_fuse(768, proj_dim, kernel_size=1, stride=1)
                for _ in range(3)
            )
        else:
            raise ValueError(
                f"Unknown projector_type={projector_type!r}; expected 'official' or 'ec'."
            )
        self.loaded_parameter_groups: tuple[str, ...] = ()
        if weights_path is not None:
            self._load_weights(weights_path)

    @staticmethod
    def _extract_group(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
        group = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
        if not group:
            raise KeyError(f"LW checkpoint contains no parameters under {prefix!r}")
        return group

    def _load_weights(self, weights_path: str) -> None:
        path = Path(weights_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"LW-DETR checkpoint does not exist: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state, dict):
            raise TypeError(f"Expected a model state dictionary in {path}, got {type(state).__name__}")

        encoder_state = self._extract_group(state, "backbone.0.encoder.")
        self.encoder.load_state_dict(encoder_state, strict=True)
        elements = sum(tensor.numel() for tensor in encoder_state.values())
        if self.projector_type == "official":
            projector_state = self._extract_group(state, "backbone.0.projector.")
            self.projector.load_state_dict(projector_state, strict=True)
            self.loaded_parameter_groups = ("encoder", "projector")
            elements += sum(tensor.numel() for tensor in projector_state.values())
            tensor_count = len(encoder_state) + len(projector_state)
            description = "encoder+projector"
        else:
            self.loaded_parameter_groups = ("encoder",)
            tensor_count = len(encoder_state)
            description = "encoder only"
        print(
            f"Loaded LW-DETR xlarge {description}: {path} "
            f"({tensor_count} tensors, {elements} elements); "
            "detector-side modules remain randomly initialized",
            flush=True,
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = self.encoder(x)
        if self.projector_type == "official":
            return self.projector(features)

        fused = torch.stack(features, dim=0).mean(dim=0)
        height, width = fused.shape[-2:]
        outputs = []
        for index, scale in enumerate((2.0, 1.0, 0.5)):
            size = (max(1, round(height * scale)), max(1, round(width * scale)))
            feature = F.interpolate(fused, size=size, mode="bilinear", align_corners=False)
            outputs.append(self.projector[index](feature))
        return outputs


__all__ = ["LWDetrBackbone"]
