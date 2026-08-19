"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""


from .criterion import ECCriterion
from .decoder import ECTransformer
from .dinov2_adapter import DinoV2Adapter
from .ecvit import ViTAdapter
from .hybrid_encoder import HybridEncoder
from .lwdetr_backbone import LWDetrBackbone
from .matcher import HungarianMatcher
from .modeling import ECDet, ECSeg, IdentityEncoder
from .postprocessor import PostProcessor
