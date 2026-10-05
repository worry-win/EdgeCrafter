"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
"""

import torch.nn as nn

from ..core import register

__all__ = ['ECDet', 'ECSeg', 'IdentityEncoder']


@register()
class IdentityEncoder(nn.Module):
    """Pass RF projector features directly to the EC decoder."""

    def forward(self, features):
        return features


class _ECBase(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder']

    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

    def forward_features(self, x):
        x = self.backbone(x)
        x = self.encoder(x)
        return x

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, '_freeze_backbone', False):
            self.backbone.eval()
        return self

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self


@register()
class ECDet(_ECBase):

    def forward(self, x, targets=None):
        x = self.forward_features(x)
        return self.decoder(x, targets)


@register()
class ECSeg(_ECBase):

    def forward(self, x, targets=None):
        x = self.forward_features(x)
        spatial_feat = x[0]
        return self.decoder(x, targets, spatial_feat)
