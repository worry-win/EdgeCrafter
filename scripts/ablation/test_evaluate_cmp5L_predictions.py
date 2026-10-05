"""Observable prediction-export contract for the held-out evaluator."""

import unittest

import torch

from scripts.ablation.evaluate_cmp5L_qbeh_checkpoint import prediction_record


class TestPredictionRecord(unittest.TestCase):
    def test_preserves_postprocessed_boxes_scores_and_labels(self):
        result = {
            "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            "scores": torch.tensor([0.75]),
            "labels": torch.tensor([3]),
        }
        self.assertEqual(prediction_record(17, result), {
            "image_id": 17,
            "boxes_xyxy": [[1.0, 2.0, 3.0, 4.0]],
            "scores": [0.75],
            "labels": [3],
        })


if __name__ == "__main__":
    unittest.main()
