import unittest

import torch
from pycocotools.coco import COCO

from ecdetseg.engine.data.dataset.coco_eval import CocoEvaluator
from ecdetseg.engine.edgecrafter.criterion import ECCriterion


class _UnusedMatcher:
    mask_point_sample_ratio = 1


def _criterion():
    return ECCriterion(
        matcher=_UnusedMatcher(),
        weight_dict={"loss_mal": 1.0},
        losses=["mal"],
        num_classes=7,
        ignore_iou_threshold=0.5,
        ignore_iof_threshold=0.7,
    )


class IgnoreRegionTest(unittest.TestCase):
    def test_mal_has_zero_gradient_for_unmatched_query_inside_ignore_region(self):
        criterion = _criterion()
        targets = [{
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty((0,), dtype=torch.int64),
            "ignore_boxes": torch.tensor([[0.25, 0.25, 0.1, 0.1]]),
        }]
        logits = torch.zeros((1, 2, 7), requires_grad=True)
        outputs = {
            "pred_logits": logits,
            "pred_boxes": torch.tensor([[[0.25, 0.25, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]]]),
        }
        empty = torch.empty(0, dtype=torch.int64)

        loss = criterion.loss_labels_mal(outputs, targets, [(empty, empty)], num_boxes=1)["loss_mal"]
        loss.backward()

        self.assertEqual(torch.count_nonzero(logits.grad[0, 0]), 0)
        self.assertGreater(torch.count_nonzero(logits.grad[0, 1]), 0)

    def test_ddf_has_zero_gradient_for_unmatched_query_inside_ignore_region(self):
        criterion = _criterion()
        criterion.use_ddf = True
        criterion.reg_max = 4
        targets = [{
            "boxes": torch.tensor([[0.8, 0.8, 0.1, 0.1]]),
            "labels": torch.tensor([2]),
            "ignore_boxes": torch.tensor([[0.25, 0.25, 0.1, 0.1]]),
        }]
        pred_corners = torch.zeros((1, 2, 4, 5), requires_grad=True)
        teacher_corners = torch.tensor(
            [[[[2.0, -2.0, 0.0, 0.0, 0.0]] * 4,
              [[-2.0, 2.0, 0.0, 0.0, 0.0]] * 4]]
        )
        outputs = {
            "pred_logits": torch.zeros((1, 2, 7)),
            "teacher_logits": torch.zeros((1, 2, 7)),
            "pred_boxes": torch.tensor([[[0.25, 0.25, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]]]),
            "pred_corners": pred_corners,
            "teacher_corners": teacher_corners,
            "ref_points": torch.zeros((1, 2, 4)),
            "reg_scale": torch.tensor([1.0]),
            "up": torch.tensor([1.0]),
        }
        indices = [(torch.tensor([1]), torch.tensor([0]))]

        loss = criterion.loss_local(outputs, targets, indices, num_boxes=1)["loss_ddf"]
        loss.backward()

        self.assertEqual(torch.count_nonzero(pred_corners.grad[0, 0]), 0)
        self.assertGreater(torch.count_nonzero(pred_corners.grad[0, 1]), 0)

    def test_ignore_annotations_become_category_agnostic_crowd_regions(self):
        coco = COCO()
        coco.dataset = {
            "images": [{"id": 1, "width": 100, "height": 100}],
            "categories": [
                {"id": 0, "name": "a"},
                {"id": 1, "name": "b"},
                {"id": 7, "name": "ignore"},
            ],
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

        evaluator = CocoEvaluator(
            coco,
            ["bbox"],
            verbose=False,
            ignore_category_ids=[7],
            ignore_as_crowd=True,
        )

        annotations = evaluator.coco_gt.dataset["annotations"]
        self.assertEqual({ann["category_id"] for ann in annotations}, {0, 1})
        self.assertTrue(all(ann["iscrowd"] == 1 and ann["ignore"] == 1 for ann in annotations))


if __name__ == "__main__":
    unittest.main()
