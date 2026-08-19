import copy
import unittest

import torch
from torchvision import tv_tensors

from ecdetseg.engine.data.transforms import GTExcludedBackgroundCorruption


class GTExcludedBackgroundCorruptionTest(unittest.TestCase):
    def _sample(self):
        image = tv_tensors.Image(
            torch.linspace(0.0, 1.0, 3 * 64 * 64, dtype=torch.float32).reshape(3, 64, 64)
        )
        target = {
            "boxes": tv_tensors.BoundingBoxes(
                [[20.0, 20.0, 36.0, 36.0]],
                format="XYXY",
                canvas_size=(64, 64),
            ),
            "ignore_boxes": tv_tensors.BoundingBoxes(
                [[42.0, 42.0, 58.0, 58.0]],
                format="XYXY",
                canvas_size=(64, 64),
            ),
            "labels": torch.tensor([2]),
        }
        return image, target

    def test_noise_and_mask_change_only_unannotated_background(self):
        for mode in ("noise", "mask"):
            with self.subTest(mode=mode):
                torch.manual_seed(7)
                image, target = self._sample()
                original_image = image.clone()
                original_target = copy.deepcopy(target)
                transform = GTExcludedBackgroundCorruption(
                    mode=mode,
                    p=1.0,
                    area_range=(0.08, 0.08),
                    aspect_ratio_range=(1.0, 1.0),
                    box_margin=2,
                    max_trials=200,
                    noise_std=0.2,
                )

                output, output_target = transform((image, target))

                changed = (output.as_subclass(torch.Tensor) != original_image).any(dim=0)
                self.assertTrue(changed.any(), f"{mode} did not alter any background pixel")
                self.assertFalse(changed[18:38, 18:38].any(), f"{mode} touched a GT margin")
                self.assertFalse(changed[40:60, 40:60].any(), f"{mode} touched an ignore-box margin")
                self.assertTrue(torch.equal(output_target["boxes"], original_target["boxes"]))
                self.assertTrue(torch.equal(output_target["ignore_boxes"], original_target["ignore_boxes"]))
                self.assertTrue(torch.equal(output_target["labels"], original_target["labels"]))
                self.assertGreaterEqual(float(output.min()), 0.0)
                self.assertLessEqual(float(output.max()), 1.0)

    def test_rejects_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "mode"):
            GTExcludedBackgroundCorruption(mode="erase")


if __name__ == "__main__":
    unittest.main()
