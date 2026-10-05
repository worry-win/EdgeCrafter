import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ablation.cmp5L_privileged_metrics import metrics_row


class PrivilegedFullEvalTest(unittest.TestCase):
    def test_metrics_row_uses_project_coco_and_pr_curve_contract(self):
        stats = {
            "coco_eval_bbox": [0.41, 0.73, 0.81],
            "yolo_f1_iou50": {
                "precision": 0.76,
                "recall": 0.69,
                "f1": 0.7232,
                "map50": 0.73,
            },
            "macro_f1_iou50": 0.701,
        }

        row = metrics_row("strong", stats, n_images=2011)

        self.assertEqual(row["condition"], "strong")
        self.assertEqual(row["n_images"], 2011)
        self.assertEqual(row["map50"], 0.73)
        self.assertEqual(row["precision"], 0.76)
        self.assertEqual(row["recall"], 0.69)
        self.assertEqual(row["f1"], 0.7232)
        self.assertEqual(row["macro_f1_iou50"], 0.701)


if __name__ == "__main__":
    unittest.main()
