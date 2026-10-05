"""Behavioral contracts for the independent KQ-R and KQ-U objectives."""

import unittest

import torch
import torch.nn.functional as F

from scripts.ablation.cmp5L_shared_query_kd import (
    KQSpec, candidate_ranking_kd, one_way_protected_kd, resolve_kq_spec,
    validate_kq_config,
)


class KQRCandidateRankingTest(unittest.TestCase):
    def test_uses_same_class_far_candidates_and_filters_teacher_reverse_pairs(self):
        # Two DN slots precede five normal slots. Normal slot 0 is matched to GT class 1.
        student = torch.tensor([[[9., 9.], [9., 9.], [0., 1.],
                                 [0., 0.], [0., 3.], [0., -1.], [0., 2.]]], requires_grad=True)
        teacher = torch.tensor([[[2., 2.], [2., 2.], [0., 2.],
                                 [0., 0.], [0., 4.], [0., -1.], [0., 1.]]], requires_grad=True)
        boxes = torch.tensor([[[.5, .5, .2, .2], [.1, .1, .1, .1],
                               [.8, .8, .1, .1], [.1, .8, .1, .1], [.8, .1, .1, .1]]])
        targets = [{"boxes": torch.tensor([[.5, .5, .2, .2]]), "labels": torch.tensor([1])}]
        matches = [(torch.tensor([0]), torch.tensor([0]))]
        result = candidate_ranking_kd(student, teacher, boxes, targets, matches,
                                      normal_query_count=5, negative_topk=3,
                                      negative_max_iou=.3)
        # Top 3 class-1 competitors are 2, 4, 1; teacher reverses only slot 2.
        self.assertEqual(result["candidate_pair_count"], 3)
        self.assertEqual(result["valid_pair_count"], 2)
        self.assertEqual(result["reverse_pair_count"], 1)
        self.assertEqual(result["matched_class_counts"]["1"], 1)
        self.assertEqual(result["valid_class_counts"]["1"], 1)
        expected_d_student = student[0, 2, 1] - student[0, [6, 3], 1]
        expected_d_teacher = teacher[0, 2, 1] - teacher[0, [6, 3], 1]
        expected = F.binary_cross_entropy_with_logits(
            expected_d_student, expected_d_teacher.detach().sigmoid(), reduction="mean")
        self.assertTrue(torch.allclose(result["loss"], expected))
        result["loss"].backward()
        self.assertIsNone(teacher.grad)
        self.assertEqual(float(student.grad[0, :2].abs().sum()), 0.)
        self.assertEqual(float(student.grad[0, 4].abs().sum()), 0.)
        self.assertGreater(float(student.grad[0, 2].abs().sum()), 0.)


class KQUOneWayProtectionTest(unittest.TestCase):
    def test_only_raises_matched_gt_class_and_lowers_far_negative_classes(self):
        student = torch.tensor([
            [[9., 9.], [-1., -1.], [1., 1.], [2., 2.], [0., 0.]],
            [[9., 9.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]],
        ], requires_grad=True)
        teacher = torch.tensor([
            [[5., 5.], [3., 3.], [0., 0.], [3., -2.], [1., 1.]],
            [[5., 5.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]],
        ], requires_grad=True)
        boxes = torch.tensor([
            [[.5, .5, .2, .2], [.2, .2, .2, .2], [.8, .8, .1, .1], [.1, .8, .1, .1]],
            [[.5, .5, .2, .2]] * 4,
        ])
        targets = [
            {"boxes": torch.tensor([[.5, .5, .2, .2], [.2, .2, .2, .2]]), "labels": torch.tensor([1, 0])},
            {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)},
        ]
        matches = [
            (torch.tensor([0, 1]), torch.tensor([0, 1])),
            (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
        ]
        result = one_way_protected_kd(student, teacher, boxes, targets, matches,
                                      normal_query_count=4, negative_topk=1,
                                      negative_max_iou=.3)
        # Lesion 0 class 1 activates; lesion 1 class 0 is teacher-lower and inactive.
        # Hard negative slot 2 wins max-class score; only its class 1 is teacher-lower.
        expected_pos = F.binary_cross_entropy_with_logits(student[0, 1, 1], teacher[0, 1, 1].detach().sigmoid()) / 2
        expected_neg = F.binary_cross_entropy_with_logits(student[0, 3, 1], teacher[0, 3, 1].detach().sigmoid()) / 2
        expected = (0.5 * expected_pos + 0.5 * expected_neg) / 2
        self.assertTrue(torch.allclose(result["loss"], expected))
        self.assertEqual(result["lesion_count"], 2)
        self.assertEqual(result["lesion_active_count"], 1)
        self.assertEqual(result["negative_count"], 1)
        self.assertEqual(result["negative_active_dimension_count"], 1)
        self.assertEqual(result["empty_image_count"], 1)
        result["loss"].backward()
        self.assertIsNone(teacher.grad)
        self.assertEqual(float(student.grad[:, 0].abs().sum()), 0.)
        self.assertEqual(float(student.grad[1].abs().sum()), 0.)
        self.assertEqual(float(student.grad[0, 2].abs().sum()), 0.)
        self.assertLess(float(student.grad[0, 1, 1]), 0.)
        self.assertGreater(float(student.grad[0, 3, 1]), 0.)


class NewArmConfigTest(unittest.TestCase):
    def test_registered_weights_match_fixed_no_step_calibration(self):
        self.assertEqual(resolve_kq_spec("KQR").output_weight, 2.322074686584331)
        self.assertEqual(resolve_kq_spec("KQU").output_weight, 6.196734780239296)
        for group in ("KQR", "KQU"):
            spec = resolve_kq_spec(group)
            self.assertEqual(spec.teacher_memory_mode, "privileged_np")
            self.assertEqual(spec.layers, (0, 1, 2, 3))
            self.assertEqual(spec.weight, .5)
            self.assertEqual(spec.num_queries, 300)

    def test_new_arms_reject_old_bidirectional_output_kd(self):
        for group, objective, grouping in (
            ("KQR", "teacher_positive_pairwise_rank", "hungarian_lesions_classwise_top20_far_candidates"),
            ("KQU", "one_way_sigmoid_soft_bce", "hungarian_lesions_top20_far_negatives"),
        ):
            spec = KQSpec(group, True, "privileged_np", (0, 1, 2, 3), output_weight=1.0)
            config = {
                "enabled": True, "teacher_memory_mode": "privileged_np",
                "layers": [0, 1, 2, 3], "weight": .5, "num_queries": 300,
                "share_student_initial_query": True, "share_student_topk": True,
                "detach_teacher": True, "loss": "cosine", "output_weight": 1.0,
                "output_loss": objective, "output_layer": 3,
                "output_all_normal_queries": False, "output_grouping": grouping,
                "negative_topk": 20, "negative_max_iou": .3,
                "empty_image_output_kd": "zero",
            }
            validate_kq_config(config, spec)
            with self.assertRaises(ValueError):
                validate_kq_config(dict(config, output_loss="sigmoid_soft_bce_grouped"), spec)
            with self.assertRaises(ValueError):
                validate_kq_config(dict(config, output_all_normal_queries=True), spec)


if __name__ == "__main__":
    unittest.main()
