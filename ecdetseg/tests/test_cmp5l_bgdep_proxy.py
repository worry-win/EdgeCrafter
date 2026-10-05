import math
import unittest

import numpy as np
import torch

from scripts.ablation.cmp5L_bgdep_proxy import (
    binary_metrics,
    point_suppression_coeff,
    regression_metrics,
)


class BGDependencyProxyTest(unittest.TestCase):
    def test_proxy_metrics_separate_ranking_and_threshold_quality(self):
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        scores = np.asarray([0.1, 0.4, 0.35, 0.8], dtype=np.float64)
        metrics = binary_metrics(labels, scores, threshold=0.5, high_threshold=0.75)

        self.assertTrue(math.isclose(metrics["roc_auc"], 0.75))
        self.assertTrue(math.isclose(metrics["pr_auc"], (1.0 + 2.0 / 3.0) / 2.0))
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["precision_high_confidence"], 1.0)
        self.assertEqual(metrics["recall_high_confidence"], 0.5)

        reg = regression_metrics(
            np.asarray([0.0, 1.0, 2.0]), np.asarray([0.0, 1.0, 3.0])
        )
        self.assertTrue(math.isclose(reg["mae"], 1.0 / 3.0))
        self.assertGreater(reg["pearson"], 0.98)
        self.assertEqual(reg["spearman"], 1.0)

    def test_point_suppression_preserves_ring_and_lambda_one_is_identity(self):
        far_probability = torch.tensor([[[[0.0, 0.5, 1.0]]]])
        query_dependency = torch.tensor([[1.0]])

        point_only = point_suppression_coeff(far_probability)
        joint = point_suppression_coeff(far_probability, query_dependency)
        disabled = point_suppression_coeff(far_probability, enabled=False)

        torch.testing.assert_close(point_only, torch.tensor([[[[1.0, 0.6, 0.2]]]]))
        torch.testing.assert_close(joint, point_only)
        torch.testing.assert_close(disabled, torch.ones_like(far_probability))

        # A point classified as Ring/FG (far probability zero) is never suppressed.
        self.assertEqual(point_only[..., 0].item(), 1.0)
