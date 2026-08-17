import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import torch
from pycocotools.coco import COCO

from ecdetseg.engine.data.dataset.coco_dataset import CocoDetection
from ecdetseg.engine.data.dataset.coco_eval import CocoEvaluator
from ecdetseg.engine.data.dataloader import BatchImageCollateFunction
from ecdetseg.engine.data._misc import convert_to_tv_tensor
from ecdetseg.engine.data.transforms._transforms import RandomIoUCrop, SanitizeBoundingBoxes
from ecdetseg.engine.data.transforms.mosaic import Mosaic
from ecdetseg.engine.edgecrafter.criterion import ECCriterion


class _UnusedMatcher:
    mask_point_sample_ratio = 1


class _FixedIoUCrop(RandomIoUCrop):
    def _resolve_params(self, image, boxes):
        return {
            "top": 0,
            "left": 0,
            "height": 16,
            "width": 16,
            "is_within_crop_area": torch.tensor([True]),
        }


class LabelIgnoreTest(unittest.TestCase):
    def test_classification_reduction_is_unchanged_without_ignore_queries(self):
        loss = torch.arange(42, dtype=torch.float32).reshape(1, 2, 21) / 10
        ignored = torch.zeros((1, 2), dtype=torch.bool)

        original = loss.mean(1).sum() * loss.shape[1] / 3
        reduced = ECCriterion._reduce_classification_loss(loss, ignored, num_boxes=3)

        self.assertTrue(torch.allclose(reduced, original))

    def test_named_ignore_category_is_removed_from_detector_targets(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            Image.new("RGB", (32, 32)).save(root / "sample.png")
            annotation = {
                "images": [{"id": 1, "file_name": "sample.png", "width": 32, "height": 32}],
                "categories": [{"id": 0, "name": "lesion"}, {"id": 7, "name": "ignore"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 0,
                        "bbox": [2, 2, 8, 8],
                        "area": 64,
                        "iscrowd": 0,
                    },
                    {
                        "id": 2,
                        "image_id": 1,
                        "category_id": 7,
                        "bbox": [12, 12, 10, 10],
                        "area": 100,
                        "iscrowd": 0,
                    },
                ],
            }
            ann_file = root / "annotations.json"
            ann_file.write_text(json.dumps(annotation), encoding="utf-8")

            dataset = CocoDetection(str(root), str(ann_file), transforms=None)
            _, target = dataset[0]

            self.assertEqual(target["labels"].tolist(), [0])
            self.assertEqual(target["ignore_boxes"].shape, (1, 4))

    def test_mosaic_moves_ignore_boxes_with_their_image_quadrant(self):
        samples = []
        for _ in range(4):
            samples.append({
                "img": Image.new("RGB", (16, 16)),
                "labels": {
                    "boxes": torch.tensor([[1.0, 1.0, 5.0, 5.0]]),
                    "labels": torch.tensor([0]),
                    "ignore_boxes": torch.tensor([[2.0, 2.0, 6.0, 6.0]]),
                },
            })

        _, target = Mosaic(output_size=16).create_mosaic_from_cache(samples, 16, 16)

        self.assertEqual(
            target["ignore_boxes"].tolist(),
            [
                [2.0, 2.0, 6.0, 6.0],
                [18.0, 2.0, 22.0, 6.0],
                [2.0, 18.0, 6.0, 22.0],
                [18.0, 18.0, 22.0, 22.0],
            ],
        )

    def test_mixup_combines_ignore_regions_from_both_images(self):
        collate = BatchImageCollateFunction(mixup_prob=1.0, mixup_epoch=1)
        collate.set_epoch(0)
        images = torch.zeros((2, 3, 8, 8))
        targets = [
            {
                "boxes": torch.tensor([[0.2, 0.2, 0.1, 0.1]]),
                "labels": torch.tensor([0]),
                "area": torch.tensor([0.01]),
                "ignore_boxes": torch.tensor([[0.3, 0.3, 0.1, 0.1]]),
            },
            {
                "boxes": torch.tensor([[0.7, 0.7, 0.1, 0.1]]),
                "labels": torch.tensor([1]),
                "area": torch.tensor([0.01]),
                "ignore_boxes": torch.tensor([[0.6, 0.6, 0.1, 0.1]]),
            },
        ]

        with patch("ecdetseg.engine.data.dataloader.random.random", return_value=0.0), \
                patch("ecdetseg.engine.data.dataloader.random.uniform", return_value=0.5):
            _, mixed_targets = collate.apply_mixup(images, targets)

        self.assertTrue(torch.allclose(
            mixed_targets[0]["ignore_boxes"],
            torch.tensor([[0.3, 0.3, 0.1, 0.1], [0.6, 0.6, 0.1, 0.1]]),
        ))

    def test_sanitize_filters_ignore_boxes_independently(self):
        image = torch.zeros((3, 32, 32))
        target = {
            "boxes": convert_to_tv_tensor(
                torch.tensor([[1.0, 1.0, 10.0, 10.0]]),
                key="boxes", box_format="xyxy", spatial_size=(32, 32),
            ),
            "labels": torch.tensor([0]),
            "area": torch.tensor([81.0]),
            "iscrowd": torch.tensor([0]),
            "ignore_boxes": convert_to_tv_tensor(
                torch.tensor([[2.0, 2.0, 8.0, 8.0], [4.0, 4.0, 4.5, 4.5]]),
                key="boxes", box_format="xyxy", spatial_size=(32, 32),
            ),
        }

        _, sanitized = SanitizeBoundingBoxes(min_size=2)(image, target)

        self.assertEqual(sanitized["labels"].tolist(), [0])
        self.assertEqual(sanitized["ignore_boxes"].shape, (1, 4))

    def test_iou_crop_uses_normal_targets_and_transforms_ignore_boxes_separately(self):
        target = {
            "boxes": convert_to_tv_tensor(
                torch.tensor([[1.0, 1.0, 10.0, 10.0]]),
                key="boxes", box_format="xyxy", spatial_size=(32, 32),
            ),
            "labels": torch.tensor([0]),
            "ignore_boxes": convert_to_tv_tensor(
                torch.tensor([[2.0, 2.0, 8.0, 8.0], [20.0, 20.0, 28.0, 28.0]]),
                key="boxes", box_format="xyxy", spatial_size=(32, 32),
            ),
        }

        _, cropped = _FixedIoUCrop(p=1.0)(Image.new("RGB", (32, 32)), target)

        self.assertTrue(torch.equal(
            cropped["ignore_boxes"].as_subclass(torch.Tensor),
            torch.tensor([[2.0, 2.0, 8.0, 8.0], [0.0, 0.0, 0.0, 0.0]]),
        ))

    def test_mal_has_no_gradient_for_unmatched_query_inside_ignore_region(self):
        criterion = ECCriterion(
            matcher=_UnusedMatcher(),
            weight_dict={"loss_mal": 1.0},
            losses=["mal"],
            num_classes=7,
            reg_max=4,
            ignore_iou_threshold=0.5,
            ignore_iof_threshold=0.7,
        )
        logits = torch.zeros((1, 2, 7), requires_grad=True)
        outputs = {
            "pred_logits": logits,
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.9, 0.9, 0.1, 0.1]]]),
        }
        targets = [{
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty((0,), dtype=torch.int64),
            "ignore_boxes": torch.tensor([[0.5, 0.5, 0.8, 0.8]]),
        }]
        empty = torch.empty((0,), dtype=torch.int64)

        loss = criterion.loss_labels_mal(outputs, targets, [(empty, empty)], num_boxes=1)["loss_mal"]
        loss.backward()

        self.assertEqual(torch.count_nonzero(logits.grad[0, 0]), 0)
        self.assertGreater(torch.count_nonzero(logits.grad[0, 1]), 0)

    def test_ddf_has_no_gradient_for_unmatched_query_inside_ignore_region(self):
        criterion = ECCriterion(
            matcher=_UnusedMatcher(),
            weight_dict={"loss_ddf": 1.0},
            losses=["local"],
            num_classes=7,
            reg_max=4,
            ignore_iou_threshold=0.5,
            ignore_iof_threshold=0.7,
        )
        pred_corners = torch.zeros((1, 2, 4, 5), requires_grad=True)
        outputs = {
            "pred_logits": torch.zeros((1, 2, 7)),
            "teacher_logits": torch.zeros((1, 2, 7)),
            "pred_boxes": torch.tensor([[[0.25, 0.25, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]]]),
            "pred_corners": pred_corners,
            "teacher_corners": torch.tensor(
                [[[[2.0, -2.0, 0.0, 0.0, 0.0]] * 4,
                  [[-2.0, 2.0, 0.0, 0.0, 0.0]] * 4]]
            ),
            "ref_points": torch.zeros((1, 2, 4)),
            "reg_scale": torch.tensor([1.0]),
            "up": torch.tensor([1.0]),
        }
        targets = [{
            "boxes": torch.tensor([[0.8, 0.8, 0.1, 0.1]]),
            "labels": torch.tensor([2]),
            "ignore_boxes": torch.tensor([[0.25, 0.25, 0.1, 0.1]]),
        }]
        indices = [(torch.tensor([1]), torch.tensor([0]))]

        loss = criterion.loss_local(outputs, targets, indices, num_boxes=1)["loss_ddf"]
        loss.backward()

        self.assertEqual(torch.count_nonzero(pred_corners.grad[0, 0]), 0)
        self.assertGreater(torch.count_nonzero(pred_corners.grad[0, 1]), 0)

    def test_named_ignore_annotations_become_class_agnostic_crowd_regions(self):
        coco = COCO()
        coco.dataset = {
            "images": [{"id": 1, "width": 100, "height": 100}],
            "categories": [{"id": 0, "name": "lesion"}, {"id": 7, "name": "ignore"}],
            "annotations": [{
                "id": 1,
                "image_id": 1,
                "category_id": 7,
                "bbox": [10, 10, 20, 20],
                "area": 400,
                "iscrowd": 0,
            }],
        }
        coco.createIndex()

        evaluator = CocoEvaluator(coco, ["bbox"], verbose=False)

        annotations = evaluator.coco_gt.dataset["annotations"]
        self.assertEqual([annotation["category_id"] for annotation in annotations], [0])
        self.assertTrue(all(
            annotation["iscrowd"] == 1 and annotation["ignore"] == 1
            for annotation in annotations
        ))


if __name__ == "__main__":
    unittest.main()
