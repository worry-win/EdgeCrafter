import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ablation.evaluate_cmp5L_oracle_gated_l1_attention import (  # noqa: E402
    mix_guided_attention,
    oracle_query_gate,
)
from scripts.ablation.count_cmp5L_attention_detection_outcomes import (  # noqa: E402
    greedy_detection_counts,
)


class OracleGatedL1AttentionTest(unittest.TestCase):
    def test_gate_a_and_b_use_best_iou_gt_and_l1_top_class(self):
        l1_boxes = torch.tensor([[
            [0.25, 0.25, 0.30, 0.30],
            [0.75, 0.75, 0.30, 0.30],
            [0.50, 0.50, 0.10, 0.10],
        ]])
        gt_boxes = [torch.tensor([
            [0.10, 0.10, 0.40, 0.40],
            [0.60, 0.60, 0.90, 0.90],
        ])]
        gt_labels = [torch.tensor([1, 2])]
        l1_logits = torch.tensor([[
            [0.0, 3.0, 1.0],
            [3.0, 0.0, 1.0],
            [0.0, 0.0, 3.0],
        ]])

        gate_a, match_a = oracle_query_gate(
            l1_boxes, l1_logits, gt_boxes, gt_labels, gate="localization"
        )
        gate_b, match_b = oracle_query_gate(
            l1_boxes, l1_logits, gt_boxes, gt_labels, gate="localization_class"
        )

        self.assertEqual(gate_a.tolist(), [[True, True, False]])
        self.assertEqual(gate_b.tolist(), [[True, False, False]])
        self.assertEqual(match_a.tolist(), [[0, 1, 0]])
        self.assertEqual(match_b.tolist(), [[0, 1, 0]])

    def test_only_reliable_queries_receive_guided_attention(self):
        original = torch.tensor([[[[0.8, 0.2]], [[0.7, 0.3]], [[0.6, 0.4]]]])
        guided = torch.tensor([[[[0.1, 0.9]], [[0.2, 0.8]], [[0.3, 0.7]]]])
        gate = torch.tensor([[True, False, True]])

        mixed = mix_guided_attention(original, guided, gate)

        torch.testing.assert_close(mixed[:, 0], guided[:, 0])
        torch.testing.assert_close(mixed[:, 1], original[:, 1])
        torch.testing.assert_close(mixed[:, 2], guided[:, 2])

    def test_fixed_threshold_counts_duplicate_and_wrong_class_as_false_positives(self):
        pred_boxes = torch.tensor([
            [0.1, 0.1, 0.4, 0.4],
            [0.1, 0.1, 0.4, 0.4],
            [0.6, 0.6, 0.9, 0.9],
        ])
        pred_logits = torch.tensor([
            [4.0, -4.0],
            [3.0, -4.0],
            [4.0, -4.0],
        ])
        gt_boxes = torch.tensor([[0.1, 0.1, 0.4, 0.4], [0.6, 0.6, 0.9, 0.9]])
        gt_labels = torch.tensor([0, 1])

        counts = greedy_detection_counts(
            pred_boxes, pred_logits, gt_boxes, gt_labels, score_threshold=0.5
        )

        self.assertEqual(counts, {"tp": 1, "fp": 2, "fn": 1})


if __name__ == "__main__":
    unittest.main()
