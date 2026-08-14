from pathlib import Path
import sys
import unittest


ECDETSEG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ECDETSEG_ROOT))

from engine.solver.metrics_format import (  # noqa: E402
    format_yolo_per_class_metrics_table,
)


class MetricsFormatTest(unittest.TestCase):
    def test_formats_yolo_per_class_metrics_in_category_order(self):
        table = format_yolo_per_class_metrics_table({
            2: {
                "name": "two",
                "precision": 0.2,
                "recall": 0.3,
                "f1": 0.4,
                "f1_iou95": 0.05,
                "f1_iou50_95": 0.25,
                "map50": 0.6,
                "confidence": 0.7,
            },
            0: {"name": "zero"},
        })

        self.assertTrue(table.startswith("[YOLO Per Class]\n"))
        self.assertLess(table.index("0: zero"), table.index("2: two"))
        self.assertIn("  0.2000", table)
        self.assertIn("  0.7000", table)


if __name__ == "__main__":
    unittest.main()
