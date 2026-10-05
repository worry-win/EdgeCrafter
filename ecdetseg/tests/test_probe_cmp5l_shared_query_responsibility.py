import unittest

from scripts.ablation.probe_cmp5L_shared_query_responsibility import (
    compare_responsibility,
    select_stratified_image_ids,
)


class SharedQueryResponsibilityTests(unittest.TestCase):
    def test_stratified_selection_covers_each_class_and_empty_images(self):
        ground_truth = {
            "images": [{"id": item} for item in range(1, 13)],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 0},
                {"id": 2, "image_id": 2, "category_id": 0},
                {"id": 3, "image_id": 3, "category_id": 1},
                {"id": 4, "image_id": 4, "category_id": 1},
                {"id": 5, "image_id": 5, "category_id": 2},
                {"id": 6, "image_id": 6, "category_id": 2},
                {"id": 7, "image_id": 7, "category_id": 3},
                {"id": 8, "image_id": 8, "category_id": 3},
            ],
        }

        selected = select_stratified_image_ids(ground_truth, per_class=2, empty=2, total=10)

        self.assertEqual(selected, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_responsibility_comparison_separates_slot_anchor_and_detection(self):
        baseline = [
            {"key": "1:0:3", "class": 3, "query_slot": 4, "anchor_index": 40, "hit": True},
            {"key": "2:0:2", "class": 2, "query_slot": 5, "anchor_index": 50, "hit": False},
        ]
        candidate = [
            {"key": "1:0:3", "class": 3, "query_slot": 4, "anchor_index": 41, "hit": False},
            {"key": "2:0:2", "class": 2, "query_slot": 6, "anchor_index": 50, "hit": True},
        ]

        result = compare_responsibility(baseline, candidate)

        self.assertEqual(result["all"]["count"], 2)
        self.assertEqual(result["all"]["same_query_slot"], 1)
        self.assertEqual(result["all"]["same_anchor_index"], 1)
        self.assertEqual(result["all"]["rescued"], 1)
        self.assertEqual(result["all"]["destroyed"], 1)
        self.assertEqual(result["class_3"]["destroyed"], 1)
        self.assertEqual(result["class_2"]["rescued"], 1)


if __name__ == "__main__":
    unittest.main()
