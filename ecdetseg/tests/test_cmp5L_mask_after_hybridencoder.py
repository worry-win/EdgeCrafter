import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ablation.evaluate_cmp5L_mask_after_hybridencoder import (
    compact_binding_metrics,
    mask_encoder_outputs,
)


class MaskAfterHybridEncoderTest(unittest.TestCase):
    def test_mask_encoder_outputs_uses_each_actual_level_shape_and_preserves_negatives(self):
        encoded = [torch.ones(2, 2, 4, 4), torch.ones(2, 2, 2, 2)]
        targets = [
            {"boxes": torch.tensor([[2.0, 2.0, 6.0, 6.0]])},
            {"boxes": torch.empty(0, 4)},
        ]

        masked, masks = mask_encoder_outputs(
            encoded, targets, image_hw=(8, 8), background_weight=0.2
        )

        expected_l0 = torch.full((4, 4), 0.2)
        expected_l0[1:3, 1:3] = 1.0
        expected_l1 = torch.full((2, 2), 0.2)
        expected_l1[:, :] = 1.0
        torch.testing.assert_close(masks[0][0, 0], expected_l0)
        torch.testing.assert_close(masks[1][0, 0], expected_l1)
        torch.testing.assert_close(masked[0][0, 0], expected_l0)
        torch.testing.assert_close(masked[0][1], encoded[0][1])
        self.assertTrue(all(torch.equal(level, torch.ones_like(level)) for level in encoded))

    def test_compact_binding_metrics_reports_only_requested_mechanism_endpoints(self):
        boxes = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.5, 0.5],
            [0.6, 0.6, 0.9, 0.9],
        ])
        logits = torch.tensor([
            [1.0, 2.0],
            [3.0, -1.0],
            [-2.0, -2.0],
        ])

        result = compact_binding_metrics(
            boxes, logits, torch.tensor([0.0, 0.0, 1.0, 1.0]), gt_class=0
        )

        self.assertEqual(set(result), {
            "q_iou_equals_q_cls", "gt_logit_q_iou", "iou_q_cls",
            "rank_q_iou_final_score", "top1_top5_iou_gap",
        })
        self.assertFalse(result["q_iou_equals_q_cls"])
        self.assertAlmostEqual(result["gt_logit_q_iou"], 1.0)
        self.assertAlmostEqual(result["iou_q_cls"], 0.25)
        self.assertEqual(result["rank_q_iou_final_score"], 2)
        self.assertAlmostEqual(result["top1_top5_iou_gap"], 0.75)


if __name__ == "__main__":
    unittest.main()
