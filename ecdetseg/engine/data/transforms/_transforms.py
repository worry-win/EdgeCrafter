"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import math
from typing import Any, Dict, List, Optional

import PIL
import PIL.Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

from ...core import register
from .._misc import (BoundingBoxes, Image, Mask, SanitizeBoundingBoxes as TVSanitizeBoundingBoxes, Video,
                     _boxes_keys, convert_to_tv_tensor)

torchvision.disable_beta_transforms_warning()


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


@register(name='SanitizeBoundingBoxes')
class SanitizeBoundingBoxes(TVSanitizeBoundingBoxes):
    def _ignore_keep(self, target):
        if not isinstance(target, dict) or 'ignore_boxes' not in target:
            return None
        boxes = target['ignore_boxes']
        if len(boxes) == 0:
            return torch.zeros((0,), dtype=torch.bool, device=boxes.device)
        boxes_tensor = boxes
        box_format = getattr(boxes, _boxes_keys[0], None)
        fmt = box_format.value.lower() if box_format is not None else 'xyxy'
        if fmt != 'xyxy':
            boxes_tensor = torchvision.ops.box_convert(boxes_tensor, in_fmt=fmt, out_fmt='xyxy')
        min_size = getattr(self, 'min_size', 1.0)
        keep = (boxes_tensor[:, 2] - boxes_tensor[:, 0] >= min_size) & \
               (boxes_tensor[:, 3] - boxes_tensor[:, 1] >= min_size)
        return keep & ~boxes_tensor.isnan().any(dim=1)

    def forward(self, *inputs):
        flat_inputs = inputs[0] if len(inputs) == 1 and isinstance(inputs[0], tuple) else inputs
        target_idx = next((i for i, value in enumerate(flat_inputs) if isinstance(value, dict)), None)
        target = flat_inputs[target_idx] if target_idx is not None else None
        ignore_keep = self._ignore_keep(target)
        ignore_boxes = None
        sanitized_inputs = inputs
        if target is not None and 'ignore_boxes' in target:
            ignore_boxes = target['ignore_boxes']
            target_without_ignore = {
                key: value for key, value in target.items() if key != 'ignore_boxes'
            }
            flat_inputs = list(flat_inputs)
            flat_inputs[target_idx] = target_without_ignore
            sanitized_inputs = (
                (tuple(flat_inputs),)
                if len(inputs) == 1 and isinstance(inputs[0], tuple)
                else tuple(flat_inputs)
            )

        outputs = super().forward(*sanitized_inputs)
        flat_outputs = outputs if isinstance(outputs, tuple) else (outputs,)
        out_target = next((value for value in flat_outputs if isinstance(value, dict)), None)
        if ignore_boxes is not None and out_target is not None:
            if ignore_keep is None:
                ignore_keep = torch.ones(
                    (len(ignore_boxes),), dtype=torch.bool, device=ignore_boxes.device
                )
            out_target['ignore_boxes'] = ignore_boxes[ignore_keep]
        return outputs


@register()
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode='constant') -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p

    def _resolve_params(self, image: Any, boxes: BoundingBoxes) -> Dict[str, Any]:
        if hasattr(self, "make_params"):
            return self.make_params([image, boxes])
        return self._get_params([image, boxes])

    @staticmethod
    def _compute_within_crop_area(boxes: BoundingBoxes, params: Dict[str, Any]) -> torch.Tensor:
        if len(params) < 1:
            return torch.ones((len(boxes),), dtype=torch.bool, device=boxes.device)
        if len(boxes) == 0:
            return torch.zeros((0,), dtype=torch.bool, device=boxes.device)

        xyxy_boxes = F.convert_bounding_box_format(
            boxes.as_subclass(torch.Tensor),
            boxes.format,
            torchvision.tv_tensors.BoundingBoxFormat.XYXY,
        )
        cx = 0.5 * (xyxy_boxes[..., 0] + xyxy_boxes[..., 2])
        cy = 0.5 * (xyxy_boxes[..., 1] + xyxy_boxes[..., 3])
        left = params["left"]
        right = left + params["width"]
        top = params["top"]
        bottom = top + params["height"]
        return (left < cx) & (cx < right) & (top < cy) & (cy < bottom)

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        flat_inputs = inputs[0] if len(inputs) == 1 and isinstance(inputs[0], tuple) else inputs
        if len(flat_inputs) != 2 or not isinstance(flat_inputs[1], dict) or "boxes" not in flat_inputs[1]:
            return super().forward(*inputs)

        image, target = flat_inputs
        params = self._resolve_params(image, target["boxes"])
        if len(params) < 1:
            return inputs if len(inputs) > 1 else inputs[0]

        out_target = dict(target)
        out_image = F.crop(
            image,
            top=params["top"],
            left=params["left"],
            height=params["height"],
            width=params["width"],
        )
        out_boxes = F.crop(
            target["boxes"],
            top=params["top"],
            left=params["left"],
            height=params["height"],
            width=params["width"],
        )
        out_boxes[~params["is_within_crop_area"]] = 0
        out_target["boxes"] = out_boxes

        if "ignore_boxes" in target:
            ignore_keep = self._compute_within_crop_area(target["ignore_boxes"], params)
            ignore_boxes = F.crop(
                target["ignore_boxes"],
                top=params["top"],
                left=params["left"],
                height=params["height"],
                width=params["width"],
            )
            if len(ignore_keep) > 0:
                ignore_boxes[~ignore_keep] = 0
            out_target["ignore_boxes"] = ignore_boxes

        if "masks" in target:
            out_target["masks"] = F.crop(
                target["masks"],
                top=params["top"],
                left=params["left"],
                height=params["height"],
                width=params["width"],
            )

        outputs = (out_image, out_target)
        return outputs if len(inputs) > 1 else outputs


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (
        BoundingBoxes,
    )
    def __init__(self, fmt='', normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(inpt, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (
        PIL.Image.Image,
    )
    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == 'float32':
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.

        inpt = Image(inpt)

        return inpt


@register()
class GTExcludedBackgroundCorruption(T.Transform):
    """Corrupt one sampled rectangle that is disjoint from all annotated boxes.

    This transform expects the post-geometry image tensor in ``[0, 1]`` and
    absolute bounding boxes. Both scored GT boxes and neutral ``ignore_boxes``
    are protected, including an optional safety margin.
    """

    def __init__(
        self,
        mode: str,
        p: float = 0.5,
        area_range=(0.05, 0.15),
        aspect_ratio_range=(0.5, 2.0),
        box_margin: float = 4.0,
        max_trials: int = 50,
        noise_std: float = 0.15,
    ) -> None:
        super().__init__()
        if mode not in ("noise", "mask"):
            raise ValueError(f"mode must be 'noise' or 'mask', got {mode!r}")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        if not (0.0 < area_range[0] <= area_range[1] <= 1.0):
            raise ValueError(f"invalid area_range={area_range!r}")
        if not (0.0 < aspect_ratio_range[0] <= aspect_ratio_range[1]):
            raise ValueError(f"invalid aspect_ratio_range={aspect_ratio_range!r}")
        if box_margin < 0 or max_trials < 1 or noise_std < 0:
            raise ValueError("box_margin/noise_std must be non-negative and max_trials positive")
        self.mode = mode
        self.p = float(p)
        self.area_range = tuple(float(value) for value in area_range)
        self.aspect_ratio_range = tuple(float(value) for value in aspect_ratio_range)
        self.box_margin = float(box_margin)
        self.max_trials = int(max_trials)
        self.noise_std = float(noise_std)

    @staticmethod
    def _xyxy(boxes) -> torch.Tensor:
        if boxes is None or len(boxes) == 0:
            return torch.empty((0, 4), dtype=torch.float32)
        tensor = boxes.as_subclass(torch.Tensor) if hasattr(boxes, "as_subclass") else torch.as_tensor(boxes)
        box_format = getattr(boxes, "format", torchvision.tv_tensors.BoundingBoxFormat.XYXY)
        return F.convert_bounding_box_format(
            tensor,
            box_format,
            torchvision.tv_tensors.BoundingBoxFormat.XYXY,
        ).detach().cpu().float()

    def _protected_boxes(self, target: Dict[str, Any], height: int, width: int) -> torch.Tensor:
        groups = [self._xyxy(target.get("boxes"))]
        if "ignore_boxes" in target:
            groups.append(self._xyxy(target.get("ignore_boxes")))
        boxes = torch.cat(groups, dim=0)
        if len(boxes) == 0:
            return boxes
        boxes[:, 0::2] = (boxes[:, 0::2] + torch.tensor([-self.box_margin, self.box_margin])).clamp(0, width)
        boxes[:, 1::2] = (boxes[:, 1::2] + torch.tensor([-self.box_margin, self.box_margin])).clamp(0, height)
        return boxes

    def _sample_rectangle(self, height: int, width: int, protected: torch.Tensor):
        image_area = height * width
        log_ratio_min = math.log(self.aspect_ratio_range[0])
        log_ratio_max = math.log(self.aspect_ratio_range[1])
        for _ in range(self.max_trials):
            area_fraction = torch.empty(1).uniform_(*self.area_range).item()
            aspect_ratio = math.exp(torch.empty(1).uniform_(log_ratio_min, log_ratio_max).item())
            rectangle_area = image_area * area_fraction
            rectangle_width = max(1, round(math.sqrt(rectangle_area * aspect_ratio)))
            rectangle_height = max(1, round(math.sqrt(rectangle_area / aspect_ratio)))
            if rectangle_width > width or rectangle_height > height:
                continue
            left = int(torch.randint(0, width - rectangle_width + 1, (1,)).item())
            top = int(torch.randint(0, height - rectangle_height + 1, (1,)).item())
            right, bottom = left + rectangle_width, top + rectangle_height
            if len(protected) > 0:
                intersects = (
                    (protected[:, 0] < right)
                    & (protected[:, 2] > left)
                    & (protected[:, 1] < bottom)
                    & (protected[:, 3] > top)
                )
                if intersects.any():
                    continue
            return top, left, rectangle_height, rectangle_width
        return None

    def __call__(self, *inputs: Any) -> Any:
        flat_inputs = inputs[0] if len(inputs) == 1 and isinstance(inputs[0], tuple) else inputs
        if len(flat_inputs) != 2 or not isinstance(flat_inputs[1], dict):
            raise TypeError("GTExcludedBackgroundCorruption expects an (image, target) pair")
        image, target = flat_inputs
        if torch.rand(1).item() >= self.p:
            return image, target
        if not isinstance(image, torch.Tensor) or image.ndim != 3:
            raise TypeError("GTExcludedBackgroundCorruption expects a CHW tensor image")

        _, height, width = image.shape
        protected = self._protected_boxes(target, height, width)
        rectangle = self._sample_rectangle(height, width, protected)
        if rectangle is None:
            return image, target

        top, left, rectangle_height, rectangle_width = rectangle
        output = image.as_subclass(torch.Tensor).clone()
        patch = output[:, top:top + rectangle_height, left:left + rectangle_width]
        if self.mode == "noise":
            patch.copy_((patch + torch.randn_like(patch) * self.noise_std).clamp_(0.0, 1.0))
        else:
            channel_mean = output.mean(dim=(1, 2), keepdim=True)
            patch.copy_(channel_mean)
        return Image(output), target
