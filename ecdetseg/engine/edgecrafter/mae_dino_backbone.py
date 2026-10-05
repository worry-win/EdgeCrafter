"""MAE-DINO ViT-B/16 encoder adapter for EdgeCrafter."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, PatchEmbed

from ..core import register
from .hybrid_encoder import ConvNormLayer_fuse


def _fixed_2d_sincos_position_embedding(
    height: int,
    width: int,
    embed_dim: int,
    device: torch.device,
) -> torch.Tensor:
    if embed_dim % 4 != 0:
        raise ValueError(f"2D sin-cos embedding dimension must be divisible by 4, got {embed_dim}")

    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )

    def encode(positions: torch.Tensor) -> torch.Tensor:
        half_axis_dim = embed_dim // 4
        omega = torch.arange(half_axis_dim, device=device, dtype=torch.float32)
        omega = 1.0 / (10000.0 ** (omega / half_axis_dim))
        angles = positions.reshape(-1, 1) * omega.reshape(1, -1)
        return torch.cat((angles.sin(), angles.cos()), dim=1)

    return torch.cat((encode(grid_x), encode(grid_y)), dim=1).unsqueeze(0)


class MAEDinoViTEncoder(nn.Module):
    """Checkpoint-compatible 12-block ViT-B/16 encoder."""

    def __init__(self, out_feature_indexes: tuple[int, ...]) -> None:
        super().__init__()
        embed_dim = 768
        depth = 12
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_size = 16
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=None,
            patch_size=self.patch_size,
            in_chans=3,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=12,
                mlp_ratio=4.0,
                qkv_bias=True,
                init_values=1e-6,
                norm_layer=norm_layer,
            )
            for _ in range(depth)
        )
        self.norm = norm_layer(embed_dim)

        self.out_feature_indexes = tuple(int(index) for index in out_feature_indexes)
        if tuple(sorted(set(self.out_feature_indexes))) != self.out_feature_indexes:
            raise ValueError(
                "MAE-DINO output feature indexes must be unique and increasing, "
                f"got {self.out_feature_indexes}"
            )
        invalid_indexes = set(self.out_feature_indexes) - set(range(depth))
        if invalid_indexes:
            raise ValueError(
                f"Invalid MAE-DINO output feature indexes: {sorted(invalid_indexes)}"
            )
        self._out_feature_indexes = set(self.out_feature_indexes)

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        batch, _, image_height, image_width = images.shape
        if image_height % self.patch_size != 0 or image_width % self.patch_size != 0:
            raise ValueError(
                "MAE-DINO ViT-B input dimensions must be divisible by 16, "
                f"got {image_height}x{image_width}"
            )

        grid_height = image_height // self.patch_size
        grid_width = image_width // self.patch_size
        tokens = self.patch_embed(images)
        position = _fixed_2d_sincos_position_embedding(
            grid_height,
            grid_width,
            self.embed_dim,
            tokens.device,
        ).to(dtype=tokens.dtype)
        tokens = tokens + position
        cls_token = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1)

        outputs = []
        for index, block in enumerate(self.blocks):
            tokens = block(tokens)
            if index in self._out_feature_indexes:
                feature = self.norm(tokens)[:, 1:]
                feature = feature.reshape(
                    batch,
                    grid_height,
                    grid_width,
                    self.embed_dim,
                ).permute(0, 3, 1, 2).contiguous()
                outputs.append(feature)
        return outputs


@register()
class MAEDinoViTBackbone(nn.Module):
    """Last-two-block MAE-DINO ViT-B features with the EC three-level projector."""

    def __init__(
        self,
        weights_path: str | None,
        out_feature_indexes: tuple[int, ...] | list[int] = (10, 11),
        proj_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = MAEDinoViTEncoder(tuple(out_feature_indexes))
        self.projector = nn.ModuleList(
            ConvNormLayer_fuse(768, proj_dim, kernel_size=1, stride=1)
            for _ in range(3)
        )
        self.loaded_parameter_groups: tuple[str, ...] = ()
        self.loaded_tensor_count = 0
        self.loaded_element_count = 0
        if weights_path is not None:
            self._load_weights(weights_path)

    def _load_weights(self, weights_path: str) -> None:
        path = Path(weights_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MAE-DINO checkpoint does not exist: {path}")
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        required_groups = ("cls_token", "patch_embed", "blocks", "norm")
        if not isinstance(checkpoint, dict) or not all(
            group in checkpoint for group in required_groups
        ):
            raise KeyError(
                "MAE-DINO checkpoint must contain cls_token, patch_embed, blocks, and norm"
            )

        self.encoder.patch_embed.load_state_dict(checkpoint["patch_embed"], strict=True)
        self.encoder.blocks.load_state_dict(checkpoint["blocks"], strict=True)
        self.encoder.norm.load_state_dict(checkpoint["norm"], strict=True)
        cls_token = checkpoint["cls_token"]
        if cls_token.shape != self.encoder.cls_token.shape:
            raise ValueError(
                f"Unexpected MAE-DINO cls_token shape: {tuple(cls_token.shape)}"
            )
        with torch.no_grad():
            self.encoder.cls_token.copy_(cls_token)

        tensors = [cls_token]
        for group in ("patch_embed", "blocks", "norm"):
            tensors.extend(checkpoint[group].values())
        self.loaded_parameter_groups = ("encoder",)
        self.loaded_tensor_count = len(tensors)
        self.loaded_element_count = sum(tensor.numel() for tensor in tensors)
        print(
            f"Loaded MAE-DINO ViT-B/16 encoder: {path} "
            f"({self.loaded_tensor_count} tensors, "
            f"{self.loaded_element_count} elements, blocks=12, outputs=[10,11]); "
            "EC projector/neck/decoder remain randomly initialized",
            flush=True,
        )

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        features = self.encoder(images)
        fused = torch.stack(features, dim=0).mean(dim=0)
        height, width = fused.shape[-2:]
        outputs = []
        for index, scale in enumerate((2.0, 1.0, 0.5)):
            size = (max(1, round(height * scale)), max(1, round(width * scale)))
            feature = F.interpolate(
                fused,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
            outputs.append(self.projector[index](feature))
        return outputs


__all__ = ["MAEDinoViTBackbone"]
