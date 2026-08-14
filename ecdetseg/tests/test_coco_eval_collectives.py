from pathlib import Path
import sys
import unittest
from unittest import mock

import torch
from pycocotools.coco import COCO


ECDETSEG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ECDETSEG_ROOT))

from engine.data.dataset import coco_eval  # noqa: E402


class CocoEvalCollectiveTest(unittest.TestCase):
    def test_ignore_category_is_excluded_from_gt_eval_and_predictions(self):
        coco_gt = COCO()
        coco_gt.dataset = {
            "images": [{"id": 1, "width": 100, "height": 100}],
            "categories": [
                {"id": cat_id, "name": f"class-{cat_id}"}
                for cat_id in range(8)
            ],
            "annotations": [
                {
                    "id": cat_id + 1,
                    "image_id": 1,
                    "category_id": cat_id,
                    "bbox": [10, 10, 20, 20],
                    "area": 400,
                    "iscrowd": 0,
                }
                for cat_id in range(8)
            ],
        }
        coco_gt.createIndex()

        evaluator = coco_eval.CocoEvaluator(
            coco_gt,
            ["bbox"],
            ignore_category_ids=[7],
        )
        prepared = evaluator.prepare_for_coco_detection({
            1: {
                "boxes": torch.tensor([
                    [10.0, 10.0, 30.0, 30.0],
                    [20.0, 20.0, 40.0, 40.0],
                ]),
                "scores": torch.tensor([0.9, 0.8]),
                "labels": torch.tensor([6, 7]),
            }
        })

        self.assertEqual(evaluator.coco_gt.getCatIds(), list(range(7)))
        self.assertEqual(evaluator.coco_eval["bbox"].params.catIds, list(range(7)))
        self.assertEqual([result["category_id"] for result in prepared], [6])

    def test_picklable_data_uses_the_shared_object_collective(self):
        payload = {"img_ids": [1, 2, 3]}
        gathered = [payload, {"img_ids": [4]}]

        with mock.patch.object(
            coco_eval.dist_utils,
            "all_gather",
            return_value=gathered,
        ) as shared_all_gather:
            result = coco_eval.all_gather(payload)

        self.assertIs(result, gathered)
        shared_all_gather.assert_called_once_with(payload)


if __name__ == "__main__":
    unittest.main()
