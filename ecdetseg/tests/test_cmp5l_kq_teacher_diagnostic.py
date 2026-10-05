import math
import unittest

import torch

from scripts.ablation.cmp5L_kq_teacher_diagnostic import (
    fixed_student_query_groups,
    gradient_pair_stats,
    grouped_loss_accounting,
    restore_trainable_parameter_mask,
)
from scripts.ablation.diagnose_cmp5L_kq_gradients import _gradient_map


class KQTeacherDiagnosticContractTest(unittest.TestCase):
    def test_fixed_student_groups_do_not_change_when_teacher_outputs_change(self):
        # q0 is a correct class-0 TP; q1 is a high-score low-IoU FP; q2 is ordinary.
        logits = torch.tensor([[[8.0, -8.0], [7.0, -7.0], [-7.0, -8.0]]])
        boxes = torch.tensor([[[0.50, 0.50, 0.20, 0.20], [0.10, 0.10, 0.10, 0.10], [0.80, 0.80, 0.10, 0.10]]])
        targets = [{"boxes": torch.tensor([[0.50, 0.50, 0.20, 0.20]]), "labels": torch.tensor([0])}]
        groups = fixed_student_query_groups(logits, boxes, targets)
        self.assertEqual(groups["tp"].tolist(), [[True, False, False]])
        self.assertEqual(groups["high_score_fp"].tolist(), [[False, True, False]])
        self.assertEqual(groups["ordinary_unmatched"].tolist(), [[False, False, True]])
        self.assertTrue(torch.equal(groups["all"], torch.ones_like(groups["all"])))

    def test_group_loss_contribution_keeps_global_denominator(self):
        loss = torch.tensor([[1.0, 3.0, 6.0, 10.0]])
        mask = torch.tensor([[True, False, True, False]])
        row = grouped_loss_accounting(loss, mask)
        self.assertEqual(row["count"], 2)
        self.assertAlmostEqual(row["group_mean"], 3.5)
        self.assertAlmostEqual(row["global_denominator_contribution"], 1.75)
        self.assertAlmostEqual(row["fraction_of_total_loss"], 7.0 / 20.0)

    def test_gradient_stats_distinguish_missing_zero_and_nonzero(self):
        both = gradient_pair_stats([torch.tensor([3.0, 4.0])], [torch.tensor([0.0, 5.0])])
        self.assertEqual(both["det_status"], "nonzero")
        self.assertEqual(both["kd_status"], "nonzero")
        self.assertAlmostEqual(both["det_norm"], 5.0)
        self.assertAlmostEqual(both["kd_norm"], 5.0)
        self.assertAlmostEqual(both["cosine"], 0.8)
        zero = gradient_pair_stats([torch.zeros(2)], [torch.ones(2)])
        self.assertEqual(zero["det_status"], "zero")
        self.assertIsNone(zero["cosine"])
        missing = gradient_pair_stats([None], [torch.ones(2)])
        self.assertEqual(missing["det_status"], "missing")
        self.assertTrue(math.isfinite(missing["kd_norm"]))
        partial = gradient_pair_stats(
            [torch.ones(2), None], [None, torch.ones(3)]
        )
        self.assertEqual(partial["det_status"], "nonzero")
        self.assertEqual(partial["kd_status"], "nonzero")
        self.assertAlmostEqual(partial["cosine"], 0.0)

    def test_constant_group_loss_reports_missing_gradients(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        constant_loss = torch.tensor(0.0)
        gradients = _gradient_map(constant_loss, [("parameter", parameter)])
        self.assertEqual(gradients, {"parameter": None})

    def test_ema_copy_recovers_training_models_parameter_mask(self):
        training_model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        training_model[0].weight.requires_grad_(False)
        ema_copy = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        ema_copy.requires_grad_(False)
        audit = restore_trainable_parameter_mask(ema_copy, training_model)
        self.assertEqual(audit["trainable_parameter_tensors"], 3)
        self.assertFalse(ema_copy[0].weight.requires_grad)
        self.assertTrue(ema_copy[0].bias.requires_grad)
        self.assertTrue(ema_copy[1].weight.requires_grad)
        self.assertTrue(ema_copy[1].bias.requires_grad)


if __name__ == "__main__":
    unittest.main()
