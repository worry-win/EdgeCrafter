"""Behavior tests for A–F offline comparisons."""

import unittest

from PIL import Image

from scripts.ablation.analyze_cmp5L_af_validation import (
    classify_gt_transitions,
    fixed_threshold_match,
    remap_bootstrap_sample,
)
from scripts.ablation.visualize_cmp5L_af_validation import _draw_panel


class TestMatching(unittest.TestCase):
    def test_matching_is_class_constrained_and_one_to_one(self):
        ground_truth = [
            {"id": 11, "category_id": 0, "bbox": [0, 0, 10, 10]},
            {"id": 12, "category_id": 1, "bbox": [20, 20, 10, 10]},
        ]
        predictions = [
            {"query_id": 3, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.9},
            {"query_id": 4, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.8},
            {"query_id": 5, "category_id": 0, "bbox": [20, 20, 10, 10], "score": 0.95},
        ]
        result = fixed_threshold_match(ground_truth, predictions, 0.5, 0.5)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 2)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["gt_matches"], {11: 3})

    def test_transition_counts_use_real_gt_matches(self):
        left = {1: 7, 2: None, 3: 4, 4: None}
        right = {1: 8, 2: 6, 3: None, 4: None}
        self.assertEqual(
            classify_gt_transitions(left, right),
            {"stable_detected": 1, "rescued": 1, "destroyed": 1, "stable_missed": 1},
        )


class TestBootstrapRemap(unittest.TestCase):
    def test_duplicate_draws_receive_unique_image_and_annotation_ids(self):
        ground_truth = {
            "images": [{"id": 1}, {"id": 2}],
            "annotations": [{"id": 5, "image_id": 1, "category_id": 0, "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 0}],
            "categories": [{"id": 0, "name": "x"}],
        }
        predictions = {"A": {1: [{"image_id": 1, "category_id": 0, "bbox": [0, 0, 1, 1], "score": 1.0}], 2: []}}
        gt, arms = remap_bootstrap_sample(ground_truth, predictions, [1, 1, 2])
        self.assertEqual([image["id"] for image in gt["images"]], [1, 2, 3])
        self.assertEqual([ann["id"] for ann in gt["annotations"]], [1, 2])
        self.assertEqual([ann["image_id"] for ann in gt["annotations"]], [1, 2])
        self.assertEqual([pred["image_id"] for pred in arms["A"]], [1, 2])


class TestVisualization(unittest.TestCase):
    def test_draw_panel_preserves_image_size(self):
        image = Image.new("L", (32, 24), 0)
        panel = _draw_panel(
            image,
            "A",
            [{"category_id": 3, "bbox": [2, 2, 8, 8]}],
            [{"category_id": 3, "bbox": [3, 3, 7, 7], "score": 0.9, "query_id": 4}],
            0.5,
            True,
        )
        self.assertEqual(panel.size, (32, 24))
        self.assertEqual(panel.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
