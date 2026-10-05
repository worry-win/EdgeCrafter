"""Behavioral gates for the cmp5L A–F validation runner."""

import unittest
from types import SimpleNamespace

import torch

from engine.edgecrafter.utils import deformable_attention_core_func_v2
from scripts.ablation.diag_cmp5L_adaptive_suppression import (
    build_suppression as historical_build_suppression,
    suppression_core,
)
from scripts.ablation.evaluate_cmp5L_af_validation import (
    build_af_coeff,
    json_compatible_args,
    normal_bg_dependency,
    region_mask_with_outside,
    require_tensor_subset_coverage,
    topk_query_predictions,
)


class TestAFDefinition(unittest.TestCase):
    def test_region_priority_and_outside_are_orthogonal(self):
        locations = torch.tensor(
            [[[[0.5, 0.5], [0.65, 0.5], [0.9, 0.9], [1.2, 0.5]]]],
            dtype=torch.float32,
        )
        boxes = torch.tensor([[0.4, 0.4, 0.6, 0.6]], dtype=torch.float32)
        region, outside = region_mask_with_outside(locations, boxes, scale=1.5)
        self.assertEqual(region.flatten().tolist(), [2, 1, 0, 0])
        self.assertEqual(outside.flatten().tolist(), [False, False, False, True])

    def test_dependency_is_point_sum_then_head_mean_then_layer_mean(self):
        # Two heads: far-BG masses 0.75 and 0.25 => layer mass 0.5.
        attention = torch.tensor([[[[0.25, 0.75], [0.75, 0.25]]]])
        region = torch.tensor([[[[2, 0], [2, 0]]]])
        result = normal_bg_dependency([attention, attention], [region, region])
        self.assertTrue(torch.allclose(result, torch.tensor([[0.5]])))

    def test_af_coefficients_preserve_historical_empty_image_privilege_only_where_defined(self):
        region = torch.zeros(2, 1, 1, 2, dtype=torch.long)
        g_gt = torch.ones(2, 1)
        g_pred = torch.ones(2, 1)
        has_gt = torch.tensor([True, False])
        expected_identity = torch.ones(1, 1, 2)
        for condition in ("B", "C", "D", "E"):
            coeff = build_af_coeff(condition, g_gt, g_pred, region, has_gt)
            self.assertTrue(torch.equal(coeff[1], expected_identity), condition)
        coeff_f = build_af_coeff("F", g_gt, g_pred, region, has_gt)
        self.assertTrue(torch.allclose(coeff_f[1], torch.full((1, 1, 2), 0.2)))

    def test_c_matches_historical_dep_gate_on_nonempty_image(self):
        region = torch.tensor([[[[2, 1, 0]], [[1, 0, 2]]]])
        g_gt = torch.tensor([[True, False]])
        coeff = build_af_coeff("C", g_gt, g_gt.float(), region, torch.tensor([True]))
        historical = torch.stack([
            historical_build_suppression(
                "dep-gate", None, torch.tensor([0.3, 0.1]), region[0], 0.2
            )
        ])
        self.assertTrue(torch.allclose(coeff, historical))


class TestCoreAndPredictions(unittest.TestCase):
    def test_manifest_args_convert_runtime_objects_to_json_values(self):
        args = SimpleNamespace(
            device=torch.device("cuda"),
            output=__import__("pathlib").Path("out"),
            limit=12,
        )
        self.assertEqual(
            json_compatible_args(args),
            {"device": "cuda", "output": "out", "limit": 12},
        )

    def test_tensor_subset_gate_rejects_incomplete_smoke_coverage(self):
        with self.assertRaisesRegex(RuntimeError, "missing fixed tensor subset"):
            require_tensor_subset_coverage({1, 3, 10, 11}, {1, 3})
        require_tensor_subset_coverage({1, 3, 10, 11}, {1, 3, 10, 11})
        require_tensor_subset_coverage({1, 3, 10, 11}, {"1", "3", "10", "11"})

    def test_all_one_coeff_matches_original_core(self):
        torch.manual_seed(7)
        batch, heads, channels, queries = 2, 2, 4, 3
        shapes = torch.tensor([[2, 2], [1, 2]])
        num_points = [2, 1]
        value = [
            torch.randn(batch, heads, channels, 4),
            torch.randn(batch, heads, channels, 2),
        ]
        locations = torch.rand(batch, queries, heads, sum(num_points), 2)
        attention = torch.randn(batch, queries, heads, sum(num_points)).softmax(-1)
        expected = deformable_attention_core_func_v2(
            value, shapes, locations, attention, num_points
        )
        actual = suppression_core(
            value,
            shapes,
            locations,
            attention,
            num_points,
            torch.ones(batch, queries, heads, sum(num_points)),
        )
        self.assertLessEqual(float((actual - expected).abs().max()), 1e-6)

    def test_topk_predictions_keep_query_identity(self):
        output = {
            "pred_logits": torch.tensor([[[0.0, 4.0], [3.0, 0.0]]]),
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.25, 0.25, 0.1, 0.1]]]),
        }
        result = topk_query_predictions(output, torch.tensor([[100, 200]]), topk=3)[0]
        self.assertEqual(result["query_ids"].tolist(), [0, 1, 0])
        self.assertEqual(result["labels"].tolist(), [1, 0, 0])
        self.assertEqual(result["boxes"].shape, (3, 4))


if __name__ == "__main__":
    unittest.main()
