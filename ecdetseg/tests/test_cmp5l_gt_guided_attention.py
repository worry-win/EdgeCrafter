import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ablation.evaluate_cmp5L_gt_guided_attention import (  # noqa: E402
    apply_gt_union_attention_prior,
)


class GTGuidedAttentionTest(unittest.TestCase):
    def test_soft_prior_uses_union_of_all_gt_boxes_and_renormalizes(self):
        weights = torch.full((1, 2, 1, 4), 0.25)
        locations = torch.tensor([[
            [[[0.20, 0.20], [0.35, 0.20], [0.50, 0.50], [0.99, 0.99]]],
            [[[0.80, 0.80], [0.65, 0.80], [0.50, 0.50], [0.10, 0.90]]],
        ]])
        gt_boxes = [torch.tensor([
            [0.10, 0.10, 0.30, 0.30],
            [0.70, 0.70, 0.90, 0.90],
        ])]

        guided, prior = apply_gt_union_attention_prior(
            weights,
            locations,
            gt_boxes,
            box_weight=1.0,
            ring_weight=0.5,
            background_weight=0.2,
            ring_scale=1.5,
        )

        expected_prior = torch.tensor([[[[1.0, 0.5, 0.2, 0.2]], [[1.0, 0.5, 0.2, 0.2]]]])
        torch.testing.assert_close(prior, expected_prior)
        torch.testing.assert_close(guided.sum(-1), torch.ones_like(guided.sum(-1)))
        torch.testing.assert_close(guided, expected_prior / expected_prior.sum(-1, keepdim=True))

    def test_all_one_prior_is_bitwise_identity(self):
        torch.manual_seed(4)
        weights = torch.randn(2, 3, 2, 5).softmax(-1)
        locations = torch.rand(2, 3, 2, 5, 2)
        gt_boxes = [torch.tensor([[0.2, 0.2, 0.7, 0.7]]), torch.empty(0, 4)]

        guided, prior = apply_gt_union_attention_prior(
            weights,
            locations,
            gt_boxes,
            box_weight=1.0,
            ring_weight=1.0,
            background_weight=1.0,
        )

        self.assertTrue(torch.equal(guided, weights))
        self.assertTrue(torch.equal(prior, torch.ones_like(prior)))

    def test_hard_prior_preserves_negative_images_and_empty_heads(self):
        weights = torch.tensor([
            [[[0.2, 0.3, 0.5]]],
            [[[0.1, 0.2, 0.7]]],
        ])
        locations = torch.tensor([
            [[[[0.0, 0.0], [0.1, 0.1], [0.9, 0.9]]]],
            [[[[0.0, 0.0], [0.1, 0.1], [0.9, 0.9]]]],
        ])
        gt_boxes = [torch.tensor([[0.45, 0.45, 0.55, 0.55]]), torch.empty(0, 4)]

        guided, prior = apply_gt_union_attention_prior(
            weights,
            locations,
            gt_boxes,
            box_weight=1.0,
            ring_weight=0.0,
            background_weight=0.0,
        )

        self.assertEqual(float(prior[0].sum()), 0.0)
        self.assertTrue(torch.equal(guided[0], weights[0]))
        self.assertTrue(torch.equal(prior[1], torch.ones_like(prior[1])))
        self.assertTrue(torch.equal(guided[1], weights[1]))


if __name__ == "__main__":
    unittest.main()
