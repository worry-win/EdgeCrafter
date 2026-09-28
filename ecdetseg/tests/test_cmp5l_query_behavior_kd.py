import sys
import unittest
import copy
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ecdetseg"))

from engine.edgecrafter.utils import deformable_attention_core_func_v2
from engine.edgecrafter.decoder import ECTransformer
from scripts.ablation.cmp5L_query_behavior_kd import (
    adaptive_farbg_coefficients,
    compute_behavior_kd_losses,
    replay_frozen_teacher,
    run_student_with_capture,
    sampled_value_suppression_core,
    summarize_behavior_batch,
)
from scripts.ablation.summarize_cmp5L_qbeh_pilot import _loss_summary, decide_go_no_go


class QueryBehaviorKDContractTest(unittest.TestCase):
    def test_all_one_suppression_is_exact_normal_core_parity(self):
        torch.manual_seed(7)
        batch, heads, channels, queries = 2, 2, 3, 5
        shapes = [[2, 3], [1, 2]]
        points = [2, 1]
        values = [
            torch.randn(batch, heads, channels, height * width)
            for height, width in shapes
        ]
        locations = torch.rand(batch, queries, heads, sum(points), 2)
        weights = torch.softmax(
            torch.randn(batch, queries, heads, sum(points)), dim=-1
        )
        coeff = torch.ones_like(weights)

        expected = deformable_attention_core_func_v2(
            values, shapes, locations, weights, points
        )
        actual = sampled_value_suppression_core(
            values, shapes, locations, weights, points, coeff
        )

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_adaptive_teacher_suppresses_only_high_dependency_far_background(self):
        # Query 0 samples only far background; query 1 samples only inside GT.
        locations = torch.tensor(
            [[
                [[[0.05, 0.05], [0.10, 0.10], [0.90, 0.90]]],
                [[[0.45, 0.45], [0.50, 0.50], [0.55, 0.55]]],
            ]],
            dtype=torch.float32,
        )
        weights = torch.full((1, 2, 1, 3), 1.0 / 3.0)
        gt_boxes = [torch.tensor([[0.4, 0.4, 0.6, 0.6]])]

        coefficients, bg_dependency = adaptive_farbg_coefficients(
            [locations, locations, locations],
            [weights, weights, weights],
            gt_boxes,
            threshold=0.20,
            far_bg_scale=0.20,
        )

        torch.testing.assert_close(bg_dependency, torch.tensor([[1.0, 0.0]]))
        for coefficient in coefficients:
            torch.testing.assert_close(
                coefficient[0, 0], torch.full((1, 3), 0.20)
            )
            torch.testing.assert_close(
                coefficient[0, 1], torch.ones(1, 3)
            )

    def test_behavior_losses_gate_positive_and_asymmetrically_suppress_negatives(self):
        student_in = torch.zeros(1, 3, 2, requires_grad=True)
        student_out = torch.tensor(
            [[[1.0, 0.0], [0.2, 0.0], [0.0, 0.2]]], requires_grad=True
        )
        teacher_in = torch.zeros(1, 3, 2, requires_grad=True)
        teacher_out = torch.tensor(
            [[[0.0, 1.0], [0.1, 0.0], [0.0, 0.1]]], requires_grad=True
        )
        student_trace = {
            "query_inputs": [student_in],
            "query_outputs": [student_out],
            "relative_sampling": [
                torch.ones(1, 3, 1, 2, 2, requires_grad=True)
            ],
        }
        teacher_trace = {
            "query_inputs": [teacher_in],
            "query_outputs": [teacher_out],
            "relative_sampling": [
                torch.zeros(1, 3, 1, 2, 2, requires_grad=True)
            ],
        }
        student_predictions = {
            "pred_logits": torch.tensor(
                [[[0.0, -1.0], [3.0, -2.0], [-3.0, -2.0]]],
                requires_grad=True,
            ),
            "pred_boxes": torch.tensor(
                [[[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1],
                  [0.8, 0.8, 0.1, 0.1]]],
                requires_grad=True,
            ),
        }
        teacher_predictions = {
            "pred_logits": torch.tensor(
                [[[2.0, -2.0], [0.0, -2.0], [-1.0, -2.0]]],
                requires_grad=True,
            ),
            "pred_boxes": torch.tensor(
                [[[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1],
                  [0.8, 0.8, 0.1, 0.1]]],
                requires_grad=True,
            ),
        }
        targets = [{
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        }]
        matches = [(torch.tensor([0]), torch.tensor([0]))]

        losses, stats = compute_behavior_kd_losses(
            student_trace,
            teacher_trace,
            student_predictions,
            teacher_predictions,
            targets,
            matches,
            distill_layers=(0,),
        )

        torch.testing.assert_close(losses["loss_query_update"], torch.tensor(1.0))
        self.assertGreater(
            float(losses["loss_negative_behavior"].detach()), 0.0
        )
        self.assertGreater(float(losses["loss_sampling"].detach()), 0.0)
        self.assertEqual(stats["teacher_better_count"], 1)
        self.assertEqual(stats["negative_count"], 2)

        sum(losses.values()).backward()
        self.assertIsNotNone(student_out.grad)
        self.assertIsNotNone(student_predictions["pred_logits"].grad)
        self.assertIsNone(teacher_out.grad)
        self.assertIsNone(teacher_predictions["pred_logits"].grad)

    def test_real_decoder_normal_replay_has_shared_init_and_prediction_parity(self):
        torch.manual_seed(11)
        student = ECTransformer(
            num_classes=2,
            hidden_dim=16,
            num_queries=5,
            feat_channels=[16, 16, 16],
            feat_strides=[8, 16, 32],
            num_levels=3,
            num_points=[1, 1, 1],
            nhead=4,
            num_layers=4,
            dim_feedforward=32,
            dropout=0.0,
            num_denoising=0,
            eval_spatial_size=None,
        ).train()
        teacher = copy.deepcopy(student).eval()
        teacher.requires_grad_(False)
        features = [
            torch.randn(1, 16, 4, 4, requires_grad=True),
            torch.randn(1, 16, 2, 2, requires_grad=True),
            torch.randn(1, 16, 1, 1, requires_grad=True),
        ]
        targets = [{
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        }]

        student_predictions, student_trace, replay_inputs = run_student_with_capture(
            student, features, targets
        )
        teacher_predictions, teacher_trace, diagnostics = replay_frozen_teacher(
            teacher,
            replay_inputs,
            targets,
            privileged=False,
        )

        self.assertEqual(diagnostics["initial_query_max_abs_diff"], 0.0)
        self.assertEqual(diagnostics["initial_reference_max_abs_diff"], 0.0)
        self.assertTrue(diagnostics["initial_query_exact"])
        self.assertTrue(diagnostics["initial_reference_exact"])
        torch.testing.assert_close(
            teacher_predictions["pred_logits"],
            student_predictions["pred_logits"],
            rtol=0,
            atol=1e-6,
        )
        torch.testing.assert_close(
            teacher_predictions["pred_boxes"],
            student_predictions["pred_boxes"],
            rtol=0,
            atol=1e-6,
        )
        self.assertEqual(len(student_trace["query_inputs"]), 4)
        self.assertEqual(len(teacher_trace["query_outputs"]), 4)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

    def test_mechanism_summary_uses_unmatched_negatives_and_matched_positives(self):
        student_predictions = {
            "pred_logits": torch.tensor([[[2.0, -1.0], [1.0, 0.0], [-2.0, -1.0]]]),
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1], [0.9, 0.9, 0.1, 0.1]]]),
        }
        teacher_predictions = {
            "pred_logits": torch.tensor([[[3.0, -2.0], [0.0, -1.0], [-3.0, -2.0]]]),
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1], [0.9, 0.9, 0.1, 0.1]]]),
        }
        student_trace = {
            "query_inputs": [torch.zeros(1, 3, 2)],
            "query_outputs": [torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])],
            "relative_sampling": [torch.ones(1, 3, 1, 1, 2)],
        }
        teacher_trace = {
            "query_inputs": [torch.zeros(1, 3, 2)],
            "query_outputs": [torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 1.0]]])],
            "relative_sampling": [torch.zeros(1, 3, 1, 1, 2)],
        }
        targets = [{
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        }]
        summary = summarize_behavior_batch(
            student_trace,
            teacher_trace,
            student_predictions,
            teacher_predictions,
            targets,
            [(torch.tensor([0]), torch.tensor([0]))],
            distill_layers=(0,),
            high_confidence=0.5,
        )

        self.assertEqual(summary["negative"]["count"], 2)
        self.assertEqual(summary["positive"]["count"], 1)
        self.assertEqual(summary["layers"]["0"]["count"], 1)
        self.assertAlmostEqual(summary["layers"]["0"]["delta_cosine_sum"], 1.0)
        self.assertAlmostEqual(summary["layers"]["0"]["relative_sampling_l1_sum"], 1.0)
        self.assertAlmostEqual(summary["positive"]["student_iou_sum"], 1.0)

    def test_go_no_go_rules_compare_each_kd_arm_to_its_control(self):
        metrics = {
            "A": {"ap50": 0.750, "precision": 0.60},
            "B": {"ap50": 0.756, "precision": 0.60},
            "C": {"ap50": 0.754, "precision": 0.62},
            "D": {"ap50": 0.762, "precision": 0.61},
            "E": {"ap50": 0.761, "precision": 0.61},
        }
        mechanism = {
            "A": {"negative_queries": {"mean_student_max_class_score": 0.10}},
            "C": {"negative_queries": {"mean_student_max_class_score": 0.08}},
        }
        verdict = decide_go_no_go(metrics, mechanism)
        self.assertEqual(verdict["query_kd"], "GO")
        self.assertEqual(verdict["negative_behavior_kd"], "GO")
        self.assertEqual(verdict["combined"], "GO")
        self.assertEqual(verdict["sampling_kd"], "NO-GO")

    def test_report_extracts_real_training_loss_and_kd_trajectory(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.txt"
            path.write_text(
                '{"epoch":0,"train_loss":26.0,"train_loss_qbeh_query":0.003,"test_coco_eval_bbox":[0.3,0.6]}\n'
                '{"epoch":1,"train_loss":21.0,"train_loss_qbeh_query":0.002,"test_coco_eval_bbox":[0.4,0.7]}\n',
                encoding="utf-8",
            )
            summary = _loss_summary(path)
        self.assertEqual(summary["first_loss"], 26.0)
        self.assertEqual(summary["last_loss"], 21.0)
        self.assertEqual(summary["first_ap50"], 0.6)
        self.assertEqual(summary["last_ap50"], 0.7)
        self.assertEqual(summary["first_query_kd"], 0.003)
        self.assertEqual(summary["last_query_kd"], 0.002)


if __name__ == "__main__":
    unittest.main()
