"""Behavioral contracts for cmp5L G/H validation-only experiments."""

import unittest

import torch

from scripts.ablation.analyze_cmp5L_gh_validation import (
    build_h_evaluation_contrasts,
    query_transition_summary,
    select_conservative_threshold,
)
from scripts.ablation.evaluate_cmp5L_gh_validation import (
    H_THRESHOLDS,
    all_point_coeff,
    build_g_gates,
    build_h_gates,
    gate_change_audit,
    stratified_image_split,
)


class TestGAttribution(unittest.TestCase):
    def test_g_conditions_correct_only_the_named_error_on_positive_images(self):
        g_gt = torch.tensor([[False, False, True, True], [False, True, False, True]])
        g_pred = torch.tensor([[False, True, False, True], [True, False, False, True]])
        has_gt = torch.tensor([True, False])

        gates = build_g_gates(g_gt, g_pred, has_gt)

        self.assertTrue(torch.equal(gates["G0"], g_pred))
        self.assertEqual(gates["G1"][0].tolist(), [False, False, False, True])
        self.assertEqual(gates["G2"][0].tolist(), [False, True, True, True])
        self.assertTrue(torch.equal(gates["G3"][0], g_gt[0]))
        for condition in ("G1", "G2", "G3"):
            self.assertTrue(torch.equal(gates[condition][1], g_pred[1]))

    def test_all_point_coeff_identity_and_suppression_are_exact(self):
        gate = torch.tensor([[False, True], [True, False]])
        coeff = all_point_coeff(gate, heads=3, points=5)
        self.assertEqual(tuple(coeff.shape), (2, 2, 3, 5))
        self.assertTrue(torch.equal(coeff[0, 0], torch.ones(3, 5)))
        self.assertTrue(torch.allclose(coeff[0, 1], torch.full((3, 5), 0.2)))

    def test_gate_change_audit_counts_only_positive_fs_and_ms(self):
        g_gt = torch.tensor([[False, False, True, True], [False, True, False, True]])
        g_pred = torch.tensor([[False, True, False, True], [True, False, False, True]])
        has_gt = torch.tensor([True, False])
        gates = build_g_gates(g_gt, g_pred, has_gt)
        audit = gate_change_audit(gates, g_gt, g_pred, has_gt)
        self.assertEqual(audit["G1_changed"], 1)
        self.assertEqual(audit["G2_changed"], 1)
        self.assertEqual(audit["G3_changed"], 2)
        self.assertEqual(audit["fs_count"], 1)
        self.assertEqual(audit["ms_count"], 1)


class TestHOperatingPoints(unittest.TestCase):
    def test_h_thresholds_are_prelocked_and_use_strict_greater_than(self):
        self.assertEqual(H_THRESHOLDS, (0.05, 0.20, 0.50, 0.80, 0.95))
        probabilities = torch.tensor([[0.05, 0.05001, 0.20, 0.95, 0.99]])
        gates = build_h_gates(probabilities)
        self.assertEqual(gates["H005"].tolist(), [[False, True, True, True, True]])
        self.assertEqual(gates["H020"].tolist(), [[False, False, False, True, True]])
        self.assertEqual(gates["H095"].tolist(), [[False, False, False, False, True]])

    def test_image_split_is_deterministic_disjoint_and_balanced_by_three_strata(self):
        image_ids = list(range(1, 13))
        positive_ids = {1, 2, 3, 4, 5, 6, 7, 8}
        duct_ids = {1, 2, 3}
        first = stratified_image_split(image_ids, positive_ids, duct_ids, seed=20260919)
        second = stratified_image_split(image_ids, positive_ids, duct_ids, seed=20260919)
        self.assertEqual(first, second)
        calibration = set(first["calibration"])
        evaluation = set(first["evaluation"])
        self.assertFalse(calibration & evaluation)
        self.assertEqual(calibration | evaluation, set(image_ids))
        for stratum_ids in (set(image_ids) - positive_ids, duct_ids, positive_ids - duct_ids):
            self.assertLessEqual(
                abs(len(calibration & stratum_ids) - len(evaluation & stratum_ids)), 1
            )

    def test_calibration_rule_requires_all_three_floors_and_breaks_ties_to_higher_t(self):
        baseline = {"ap50": 0.60, "fixed_f1": 0.70, "duct_ap50": 0.40}
        candidates = {
            "H005": {"threshold": 0.05, "ap50": 0.62, "fixed_f1": 0.69, "duct_ap50": 0.42},
            "H020": {"threshold": 0.20, "ap50": 0.61, "fixed_f1": 0.70, "duct_ap50": 0.40},
            "H050": {"threshold": 0.50, "ap50": 0.61, "fixed_f1": 0.71, "duct_ap50": 0.41},
            "H080": {"threshold": 0.80, "ap50": 0.59, "fixed_f1": 0.72, "duct_ap50": 0.43},
            "H095": {"threshold": 0.95, "ap50": 0.60, "fixed_f1": 0.70, "duct_ap50": 0.39},
        }
        result = select_conservative_threshold(baseline, candidates)
        self.assertEqual(result["selected"], "H050")
        self.assertEqual(result["eligible"], ["H020", "H050"])

    def test_evaluation_bootstrap_keeps_historical_contrast_when_none_is_selected(self):
        self.assertEqual(
            build_h_evaluation_contrasts(None),
            {"H005_minus_A": ("A", "H005")},
        )
        self.assertEqual(
            set(build_h_evaluation_contrasts("H080")),
            {"H080_minus_A", "H080_minus_H005", "H005_minus_A"},
        )


class TestQueryAttribution(unittest.TestCase):
    def test_transition_summary_separates_reassignment_from_rescue_and_destruction(self):
        left = {1: 5, 2: 7, 3: None, 4: 9, 5: None}
        right = {1: 5, 2: 8, 3: 6, 4: None, 5: None}
        result = query_transition_summary(left, right)
        self.assertEqual(
            result,
            {
                "stable_same_query": 1,
                "stable_reassigned": 1,
                "rescued": 1,
                "destroyed": 1,
                "stable_missed": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
