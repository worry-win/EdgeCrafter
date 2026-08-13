from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np


ECDETSEG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ECDETSEG_ROOT))

from engine.solver.ec_engine import summarize_pr_curve_f1  # noqa: E402


class EcEngineF1Test(unittest.TestCase):
    def test_summary_reports_iou50_iou95_and_sweep_mean(self):
        recalls = np.linspace(0.0, 1.0, 101)
        iou_thresholds = np.linspace(0.50, 0.95, 10)
        precision = np.empty((10, 101, 1, 1, 3), dtype=np.float64)
        per_iou_precision = np.linspace(0.1, 1.0, 10)
        for index, value in enumerate(per_iou_precision):
            precision[index, :, :, :, :] = value

        coco_eval = SimpleNamespace(
            eval={"precision": precision},
            params=SimpleNamespace(recThrs=recalls, iouThrs=iou_thresholds),
        )
        summary = summarize_pr_curve_f1(coco_eval)
        expected_f1 = 2 * per_iou_precision / (per_iou_precision + 1.0)

        self.assertAlmostEqual(summary["f1_iou50"], expected_f1[0])
        self.assertAlmostEqual(summary["f1_iou95"], expected_f1[-1])
        self.assertAlmostEqual(summary["f1_iou50_95_mean"], expected_f1.mean())


if __name__ == "__main__":
    unittest.main()
