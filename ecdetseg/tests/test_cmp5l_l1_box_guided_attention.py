import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ablation.evaluate_cmp5L_l1_box_guided_attention import (  # noqa: E402
    apply_box_attention_prior,
    layer_indices,
)


class L1BoxGuidedAttentionTest(unittest.TestCase):
    def test_soft_prior_uses_each_query_box_and_renormalizes_per_head(self):
        weights = torch.tensor([[[[0.25, 0.25, 0.25, 0.25]]]])
        locations = torch.tensor([[[[
            [0.50, 0.50],  # box
            [0.75, 0.50],  # 1.5x ring
            [0.90, 0.50],  # far background
            [0.10, 0.10],  # far background
        ]]]])
        l1_boxes = torch.tensor([[[0.50, 0.50, 0.40, 0.40]]])

        guided, prior = apply_box_attention_prior(
            weights, locations, l1_boxes,
            box_weight=1.0, ring_weight=0.5, background_weight=0.2,
            ring_scale=1.5,
        )

        torch.testing.assert_close(prior, torch.tensor([[[[1.0, 0.5, 0.2, 0.2]]]]))
        torch.testing.assert_close(guided.sum(-1), torch.ones_like(guided.sum(-1)))
        torch.testing.assert_close(
            guided, torch.tensor([[[[1.0 / 1.9, 0.5 / 1.9, 0.2 / 1.9, 0.2 / 1.9]]]])
        )

    def test_all_one_prior_is_bitwise_identity(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 3, 2, 5)
        weights = logits.softmax(-1)
        locations = torch.rand(2, 3, 2, 5, 2)
        boxes = torch.rand(2, 3, 4)

        guided, prior = apply_box_attention_prior(
            weights, locations, boxes,
            box_weight=1.0, ring_weight=1.0, background_weight=1.0,
            ring_scale=1.5,
        )

        self.assertTrue(torch.equal(prior, torch.ones_like(prior)))
        self.assertTrue(torch.equal(guided, weights))

    def test_hard_prior_falls_back_to_original_if_head_has_no_inside_point(self):
        weights = torch.tensor([[[[0.2, 0.3, 0.5]]]])
        locations = torch.tensor([[[[[0.0, 0.0], [0.1, 0.1], [0.9, 0.9]]]]])
        boxes = torch.tensor([[[0.5, 0.5, 0.1, 0.1]]])

        guided, prior = apply_box_attention_prior(
            weights, locations, boxes,
            box_weight=1.0, ring_weight=0.0, background_weight=0.0,
            ring_scale=1.5,
        )

        self.assertEqual(float(prior.sum()), 0.0)
        self.assertTrue(torch.equal(guided, weights))

    def test_layer_names_map_to_actual_indices_without_touching_l1_or_l4(self):
        self.assertEqual(layer_indices("l2"), (1,))
        self.assertEqual(layer_indices("l3"), (2,))
        self.assertEqual(layer_indices("l2_l3"), (1, 2))


if __name__ == "__main__":
    unittest.main()
