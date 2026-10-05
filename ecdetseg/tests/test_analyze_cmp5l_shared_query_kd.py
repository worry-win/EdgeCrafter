import json
import tempfile
import unittest
from pathlib import Path

from scripts.ablation.analyze_cmp5L_shared_query_kd import (
    classify_gt_transitions,
    fixed_threshold_match,
    load_predictions,
)


class SharedQueryAnalysisTests(unittest.TestCase):
    def test_fixed_threshold_match_is_class_constrained_and_one_to_one(self):
        ground_truth = [
            {"id": 11, "category_id": 2, "bbox": [0, 0, 10, 10]},
            {"id": 12, "category_id": 3, "bbox": [20, 20, 10, 10]},
        ]
        predictions = [
            {"category_id": 2, "bbox": [0, 0, 10, 10], "score": 0.9},
            {"category_id": 2, "bbox": [0, 0, 10, 10], "score": 0.8},
            {"category_id": 2, "bbox": [20, 20, 10, 10], "score": 0.99},
            {"category_id": 3, "bbox": [20, 20, 10, 10], "score": 0.49},
        ]

        result = fixed_threshold_match(ground_truth, predictions, 0.5, 0.5)

        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 2)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["gt_matches"], {11: True})

    def test_gt_transitions_are_counted_by_annotation_identity(self):
        left = {1: True, 2: None, 3: True, 4: None}
        right = {1: True, 2: True, 3: None, 4: None}

        self.assertEqual(classify_gt_transitions(left, right), {
            "stable_detected": 1,
            "rescued": 1,
            "destroyed": 1,
            "stable_missed": 1,
        })

    def test_load_predictions_accepts_evaluator_schema_without_query_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(
                json.dumps({
                    "image_id": 7,
                    "boxes_xyxy": [[1, 2, 6, 8]],
                    "scores": [0.75],
                    "labels": [3],
                }) + "\n",
                encoding="utf-8",
            )

            loaded = load_predictions(path, [7])

        self.assertEqual(loaded[7], [{
            "image_id": 7,
            "category_id": 3,
            "bbox": [1.0, 2.0, 5.0, 6.0],
            "score": 0.75,
        }])


if __name__ == "__main__":
    unittest.main()
