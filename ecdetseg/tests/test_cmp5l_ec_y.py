import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.ablation.train_cmp5L_ec_y import (
    late_kd_factor, prepare_late_teacher, summarize_fixed_teacher_scores,
)


class LateKDTest(unittest.TestCase):
    def test_taper_is_one_at_50_and_zero_from_80(self):
        self.assertEqual(late_kd_factor('Y1', 49, 0, 100), 1.0)
        self.assertEqual(late_kd_factor('Y1', 50, 0, 100), 1.0)
        self.assertAlmostEqual(late_kd_factor('Y1', 65, 0, 100), .5)
        self.assertAlmostEqual(late_kd_factor('Y1', 79, 99, 100), .0003333333333333333)
        self.assertEqual(late_kd_factor('Y1', 80, 0, 100), 0.0)
        self.assertEqual(late_kd_factor('Y3', 99, 0, 100), 0.0)
        self.assertEqual(late_kd_factor('Y0', 99, 0, 100), 1.0)
        self.assertEqual(late_kd_factor('Y2', 99, 0, 100), 1.0)

    def test_frozen_teacher_is_separate_from_live_evaluation_ema_and_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            ema_module = torch.nn.Linear(3, 2)
            solver = SimpleNamespace(ema=SimpleNamespace(module=ema_module, updates=10),
                                     last_epoch=49, frozen_teacher_sha=None)
            runtime = SimpleNamespace(y_arm='Y2', teacher=ema_module,
                                      solver=solver, output=Path(directory),
                                      args=SimpleNamespace(init_sha256='seed42'),
                                      fixed_teacher_state=None)
            prepare_late_teacher(runtime, 50)
            self.assertIsNot(runtime.teacher, solver.ema.module)
            self.assertFalse(any(p.requires_grad for p in runtime.teacher.parameters()))
            initial = {k: v.clone() for k, v in runtime.teacher.state_dict().items()}
            with torch.no_grad():
                for value in solver.ema.module.parameters():
                    value.add_(1)
            prepare_late_teacher(runtime, 51)
            for key, value in initial.items():
                self.assertTrue(torch.equal(runtime.teacher.state_dict()[key], value))
            resumed = SimpleNamespace(y_arm='Y2', teacher=solver.ema.module,
                                      solver=SimpleNamespace(ema=solver.ema, last_epoch=50,
                                                             frozen_teacher_sha=solver.frozen_teacher_sha),
                                      output=Path(directory),
                                      args=SimpleNamespace(init_sha256='seed42'),
                                      fixed_teacher_state=None)
            prepare_late_teacher(resumed, 51)
            for key, value in initial.items():
                self.assertTrue(torch.equal(resumed.teacher.state_dict()[key], value))

    def test_fixed_diagnostic_reports_matched_gt_probability_gap(self):
        student = torch.tensor([[[0., 0.], [0., 0.]]])
        teacher = torch.tensor([[[0., 2.], [0., 0.]]])
        targets = [{'labels': torch.tensor([1])}]
        matches = [(torch.tensor([0]), torch.tensor([0]))]
        report = summarize_fixed_teacher_scores(student, teacher, targets, matches)
        self.assertEqual(report['matched_class_1']['count'], 1)
        self.assertAlmostEqual(report['matched_class_1']['student_mean'], .5)
        self.assertGreater(report['matched_class_1']['teacher_minus_student_mean'], .38)


if __name__ == '__main__':
    unittest.main()
