"""Behavioral contracts for KQ-B's grouped same-slot classification KD."""

import unittest

import torch
import torch.nn.functional as F

from scripts.ablation.cmp5L_shared_query_kd import (
    KQSpec,
    final_layer_hungarian_matches,
    grouped_sigmoid_output_kd,
    resolve_kq_spec,
    validate_kq_config,
)


class KQBGroupedKDTest(unittest.TestCase):
    def test_registered_kqb_spec_uses_prelocked_calibrated_weight(self):
        spec = resolve_kq_spec("KQB")
        self.assertEqual(spec.teacher_memory_mode, "privileged_np")
        self.assertEqual(spec.layers, (0, 1, 2, 3))
        self.assertEqual(spec.weight, .5)
        self.assertEqual(spec.output_weight, 3.643302750228453)
        self.assertEqual(spec.num_queries, 300)

    def test_kqb_config_rejects_all_query_or_nonzero_empty_supervision(self):
        spec = KQSpec("KQB", True, "privileged_np", (0, 1, 2, 3), weight=.5, output_weight=1.23)
        config = {
            "enabled": True, "teacher_memory_mode": "privileged_np",
            "layers": [0, 1, 2, 3], "weight": .5, "num_queries": 300,
            "share_student_initial_query": True, "share_student_topk": True,
            "detach_teacher": True, "loss": "cosine", "output_weight": 1.23,
            "output_loss": "sigmoid_soft_bce_grouped", "output_layer": 3,
            "output_all_normal_queries": False,
            "output_grouping": "hungarian_lesions_top20_far_negatives",
            "negative_topk": 20, "negative_max_iou": .3,
            "empty_image_output_kd": "zero",
        }
        validate_kq_config(config, spec)
        for key, value in (("output_all_normal_queries", True), ("empty_image_output_kd", "all_queries")):
            invalid = dict(config, **{key: value})
            with self.assertRaises(ValueError):
                validate_kq_config(invalid, spec)

    def test_uses_detached_final_normal_outputs_for_detection_matcher(self):
        class Matcher:
            def __init__(self):
                self.received = None

            def __call__(self, predictions, targets):
                self.received = predictions
                return {"indices": [(torch.tensor([1]), torch.tensor([0]))]}

        matcher = Matcher()
        logits = torch.randn(1, 2, 4, requires_grad=True)
        boxes = torch.rand(1, 2, 4, requires_grad=True)
        outputs = {"pred_logits": logits, "pred_boxes": boxes, "dn_outputs": [object()]}
        target = [{"boxes": torch.rand(1, 4), "labels": torch.tensor([2])}]
        matched = final_layer_hungarian_matches(matcher, outputs, target, normal_query_count=2)
        self.assertEqual(matched[0][0].tolist(), [1])
        self.assertEqual(set(matcher.received), {"pred_logits", "pred_boxes"})
        self.assertFalse(matcher.received["pred_logits"].requires_grad)
        self.assertFalse(matcher.received["pred_boxes"].requires_grad)
        with self.assertRaises(ValueError):
            final_layer_hungarian_matches(matcher, outputs, target, normal_query_count=3)

    def test_matched_low_score_lesion_and_top_far_negatives_only(self):
        # The first two student queries are DN; the last five are normal.
        student = torch.tensor(
            [
                [[9., 9.], [9., 9.], [-3., -2.], [3., -1.], [4., 0.], [-1., 1.], [-2., -2.]],
                [[9., 9.], [9., 9.], [2., 2.], [1., 1.], [0., 0.], [-1., -1.], [-2., -2.]],
            ], requires_grad=True,
        )
        teacher = torch.zeros(2, 5, 2, requires_grad=True)
        boxes = torch.tensor(
            [
                [[.5, .5, .2, .2], [.1, .1, .1, .1], [.5, .5, .2, .2],
                 [.85, .85, .1, .1], [.8, .1, .1, .1]],
                [[.5, .5, .2, .2]] * 5,
            ]
        )
        targets = [
            {"boxes": torch.tensor([[.5, .5, .2, .2]]), "labels": torch.tensor([1])},
            {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)},
        ]
        matches = [
            (torch.tensor([0]), torch.tensor([0])),
            (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
        ]

        result = grouped_sigmoid_output_kd(
            student, teacher, boxes, targets, matches,
            normal_query_count=5, negative_topk=2, negative_max_iou=0.3,
        )
        expected_pos = F.binary_cross_entropy_with_logits(
            student[0, 2:3], teacher[0, 0:1].detach().sigmoid(),
        )
        expected_neg = F.binary_cross_entropy_with_logits(
            student[0, [3, 5]], teacher[0, [1, 3]].detach().sigmoid(),
        )
        expected = (0.5 * expected_pos + 0.5 * expected_neg) / 2
        self.assertTrue(torch.allclose(result["loss"], expected))
        self.assertEqual(result["lesion_indices"], [[0], []])
        self.assertEqual(result["negative_indices"], [[1, 3], []])
        self.assertEqual(result["lesion_count"], 1)
        self.assertEqual(result["negative_count"], 2)
        self.assertEqual(result["empty_image_count"], 1)
        self.assertEqual(result["matched_class_counts"], {"0": 0, "1": 1})
        self.assertEqual(result["lesion_probability_samples"][0]["query_index"], 0)
        self.assertEqual(result["low_score_lesion_count_below_0_5"], 1)
        result["loss"].backward()
        self.assertIsNone(teacher.grad)
        self.assertEqual(float(student.grad[:, :2].abs().sum()), 0.0)
        self.assertEqual(float(student.grad[1].abs().sum()), 0.0)
        self.assertEqual(float(student.grad[0, [4, 6]].abs().sum()), 0.0)
        self.assertGreater(float(student.grad[0, 2].abs().sum()), 0.0)
        self.assertGreater(float(student.grad[0, [3, 5]].abs().sum()), 0.0)

    def test_empty_image_is_zero_and_missing_negative_keeps_half_weight(self):
        student = torch.zeros(1, 2, 1, requires_grad=True)
        teacher = torch.zeros(1, 2, 1, requires_grad=True)
        boxes = torch.tensor([[[.5, .5, .2, .2], [.5, .5, .2, .2]]], requires_grad=True)
        positive = [{"boxes": torch.tensor([[.5, .5, .2, .2]]), "labels": torch.tensor([0])}]
        match = [(torch.tensor([0]), torch.tensor([0]))]
        result = grouped_sigmoid_output_kd(
            student, teacher, boxes, positive, match,
            normal_query_count=2, global_image_count=3, ddp_world_size=2,
        )
        # No hard negative is eligible; its half is zero, not reassigned.
        self.assertEqual(result["negative_count"], 0)
        self.assertAlmostEqual(float(result["loss"].detach()), .5 * 2 / 3 * float(torch.log(torch.tensor(2.))), places=6)
        result["loss"].backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNone(boxes.grad)

        empty = [{"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}]
        empty_match = [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))]
        zero = grouped_sigmoid_output_kd(
            student, teacher, boxes, empty, empty_match, normal_query_count=2,
        )
        self.assertEqual(float(zero["loss"]), 0.0)
        self.assertEqual(zero["empty_image_count"], 1)


if __name__ == "__main__":
    unittest.main()
