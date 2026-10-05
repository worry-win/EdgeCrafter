import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ablation.probe_cmp5L_privileged_ema_oracle import (
    image_score_metrics,
    lesion_query_metrics,
    privilege_features,
)


class PrivilegedFeatureMaskTest(unittest.TestCase):
    def test_soft_mask_preserves_foreground_and_negative_images(self):
        features = [torch.ones(2, 1, 4, 4), torch.ones(2, 1, 2, 2)]
        targets = [
            {"boxes": torch.tensor([[2.0, 2.0, 6.0, 6.0]])},
            {"boxes": torch.empty(0, 4)},
        ]
        privileged, masks = privilege_features(
            features, targets, image_hw=(8, 8), background_weight=0.2
        )

        expected = torch.full((4, 4), 0.2)
        expected[1:3, 1:3] = 1.0
        torch.testing.assert_close(masks[0][0, 0], expected)
        torch.testing.assert_close(privileged[0][0, 0], expected)
        torch.testing.assert_close(privileged[0][1], features[0][1])
        torch.testing.assert_close(masks[1][1], torch.ones_like(masks[1][1]))
        self.assertTrue(all(torch.equal(feature, torch.ones_like(feature)) for feature in features))

    def test_query_metrics_keep_binding_identities_and_score_rank_separate(self):
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
        result = lesion_query_metrics(
            boxes, logits, torch.tensor([0.0, 0.0, 1.0, 1.0]),
            gt_class=0, hungarian_query=0,
        )

        self.assertEqual(result["q_iou"], 0)
        self.assertEqual(result["q_cls"], 1)
        self.assertEqual(result["q_hungarian"], 0)
        self.assertFalse(result["q_iou_equals_q_cls"])
        self.assertTrue(result["q_iou_equals_q_hungarian"])
        self.assertFalse(result["q_cls_equals_q_hungarian"])
        self.assertEqual(result["rank_q_iou_final_score"], 2)
        self.assertEqual(result["rank_q_iou_gt_class"], 2)
        self.assertAlmostEqual(result["top1_best_iou"], 0.25)
        self.assertAlmostEqual(result["top5_best_iou"], 1.0)
        self.assertAlmostEqual(result["top1_top5_iou_gap"], 0.75)
        self.assertAlmostEqual(result["gt_logit_q_iou"], 1.0)
        self.assertAlmostEqual(result["wrong_logit_q_iou"], 2.0)
        self.assertAlmostEqual(result["class_margin_q_iou"], -1.0)
        self.assertAlmostEqual(result["best_localized_gt_logit"], 1.0)
        self.assertTrue(result["has_iou_05_candidate"])
        self.assertTrue(result["joint_success_iou05_score05"])
    def test_context_mask_keeps_expanded_ring_at_alpha(self):
        features = [torch.ones(1, 1, 4, 4)]
        targets = [{"boxes": torch.tensor([[2.0, 2.0, 6.0, 6.0]])}]
        privileged, masks = privilege_features(
            features,
            targets,
            image_hw=(8, 8),
            background_weight=0.2,
            context_weight=0.5,
            context_scale=2.0,
        )

        expected = torch.full((4, 4), 0.5)
        expected[1:3, 1:3] = 1.0
        torch.testing.assert_close(masks[0][0, 0], expected)
        torch.testing.assert_close(privileged[0][0, 0], expected)

    def test_image_score_metrics_separate_positive_and_background_queries(self):
        boxes = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [0.6, 0.6, 0.9, 0.9],
            [0.0, 0.0, 0.1, 0.1],
        ])
        logits = torch.tensor([[2.0, 0.0], [3.0, -1.0], [-2.0, -2.0]])
        positive = image_score_metrics(
            boxes, logits, torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        )
        negative = image_score_metrics(boxes, logits, torch.empty(0, 4))

        self.assertEqual(positive["positive_query_count"], 1)
        self.assertEqual(positive["background_query_count"], 2)
        self.assertEqual(positive["background_fp_count_05"], 1)
        self.assertEqual(negative["positive_query_count"], 0)
        self.assertEqual(negative["background_query_count"], 3)
        self.assertEqual(negative["background_fp_count_05"], 2)
        self.assertAlmostEqual(negative["top1_query_score"], torch.sigmoid(torch.tensor(3.0)).item())


if __name__ == "__main__":
    unittest.main()
