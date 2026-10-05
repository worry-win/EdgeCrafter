import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ablation.dump_cmp5L_internal_behavior import (
    _select_gradient_ann_ids,
    classify_lesion,
)


class LesionSelectionTest(unittest.TestCase):
    def test_groups_use_class_agnostic_localisation_and_gt_class_score(self):
        self.assertEqual(classify_lesion([True] * 5, [True] * 5), "agreement_easy")
        self.assertEqual(
            classify_lesion([False, True, False, False, False], [True] * 5),
            "rank_fixable",
        )
        self.assertEqual(
            classify_lesion([False, True, False, False, False], [False, True, True, True, True]),
            "localization_fixable",
        )
        self.assertEqual(
            classify_lesion([True, False, True, False, True], [True] * 5),
            "student_correct_disagreement",
        )
        self.assertEqual(classify_lesion([False] * 5, [True] * 5), "common_failure")

    def test_gradient_subset_is_stratified_and_seeded(self):
        rows = [
            {"ann_id": i, "group": "agreement_easy" if i < 8 else "common_failure",
             "gt_class": i % 4}
            for i in range(16)
        ]
        first = _select_gradient_ann_ids(rows, 8, seed=9)
        second = _select_gradient_ann_ids(rows, 8, seed=9)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        picked_groups = {rows[i]["group"] for i in first}
        self.assertEqual(picked_groups, {"agreement_easy", "common_failure"})


if __name__ == "__main__":
    unittest.main()
