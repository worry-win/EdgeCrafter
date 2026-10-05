import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ablation.analyze_cmp5L_internal_behavior import (
    _ffn_metrics,
    query_prototype_metrics,
    summarize,
)


class InternalBehaviorAnalysisTest(unittest.TestCase):
    def test_summary_reports_robust_statistics_and_seeded_ci(self):
        values = np.arange(1, 9, dtype=np.float64)
        result = summarize(values, seed=3, bootstrap=200)
        self.assertEqual(result["n"], 8)
        self.assertAlmostEqual(result["mean"], 4.5)
        self.assertAlmostEqual(result["median"], 4.5)
        self.assertEqual(result["iqr"], [2.75, 6.25])
        self.assertLess(result["ci95"][0], result["mean"])
        self.assertGreater(result["ci95"][1], result["mean"])

    def test_ffn_reports_energy_active_fractions(self):
        data = {
            "ffn_activation": np.zeros((3, 6, 2, 4), dtype=np.float32),
        }
        data["ffn_activation"][:, 4, :, 0] = 3
        data["ffn_activation"][:, 4, :, 1:] = 1
        metrics = _ffn_metrics(data)
        self.assertEqual(metrics["active50_fraction"].shape, (3, 2))
        self.assertTrue(np.all(metrics["active50_fraction"] == 0.25))
        self.assertTrue(np.all(metrics["active90_fraction"] == 0.75))

    def test_query_prototypes_are_fit_without_using_failure_labels_as_features(self):
        responsible = np.array([
            [[0.0, 0.0]], [[0.2, 0.0]], [[3.0, 0.0]], [[4.0, 0.0]],
        ])
        background = np.array([
            [[5.0, 0.0]], [[5.0, 0.0]], [[5.0, 0.0]], [[5.0, 0.0]],
        ])
        success = np.array([True, True, False, False])
        result = query_prototype_metrics(responsible, background, success)
        self.assertLess(result["distance_to_correct_prototype"][0, 0],
                        result["distance_to_correct_prototype"][3, 0])
        self.assertEqual(result["prototype_margin"].shape, (4, 1))


if __name__ == "__main__":
    unittest.main()
