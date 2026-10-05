"""Paired-resampling contract for postprocessed cmp5L detections."""

import unittest

from scripts.ablation.bootstrap_cmp5L_depgate_audit import coco_ap50, remap_sample


class TestPairedResample(unittest.TestCase):
    def test_duplicate_draws_get_distinct_ids_in_all_arms(self):
        ground_truth = {
            "images": [{"id": 7, "file_name": "a.png"}, {"id": 9, "file_name": "b.png"}],
            "annotations": [{"id": 20, "image_id": 7, "category_id": 0, "bbox": [1, 2, 3, 4]}],
            "categories": [{"id": 0, "name": "lesion"}],
        }
        predictions = {
            "one": {7: [{"image_id": 7, "category_id": 0, "bbox": [1, 2, 3, 4], "score": 0.8}], 9: []},
            "two": {7: [], 9: []},
        }
        gt, arms = remap_sample(ground_truth, predictions, [7, 7, 9])
        self.assertEqual([i["id"] for i in gt["images"]], [1, 2, 3])
        self.assertEqual([a["image_id"] for a in gt["annotations"]], [1, 2])
        self.assertEqual([p["image_id"] for p in arms["one"]], [1, 2])
        self.assertEqual(arms["two"], [])

    def test_perfect_coco_detections_have_unit_ap50(self):
        ground_truth = {
            "info": {}, "licenses": [],
            "images": [{"id": 1, "file_name": "a.png", "width": 20, "height": 20},
                       {"id": 2, "file_name": "b.png", "width": 20, "height": 20}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 0, "bbox": [1, 1, 4, 4], "area": 16, "iscrowd": 0},
                {"id": 2, "image_id": 2, "category_id": 3, "bbox": [2, 2, 5, 5], "area": 25, "iscrowd": 0},
            ],
            "categories": [{"id": 0, "name": "other"}, {"id": 3, "name": "duct"}],
        }
        detections = [
            {"image_id": 1, "category_id": 0, "bbox": [1, 1, 4, 4], "score": 0.9},
            {"image_id": 2, "category_id": 3, "bbox": [2, 2, 5, 5], "score": 0.9},
        ]
        result = coco_ap50(ground_truth, detections)
        self.assertAlmostEqual(result["ap50"], 1.0)
        self.assertAlmostEqual(result["duct_ap50"], 1.0)


if __name__ == "__main__":
    unittest.main()
