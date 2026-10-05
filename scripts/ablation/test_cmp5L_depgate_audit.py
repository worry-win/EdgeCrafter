"""Contract tests for isolating an S1 Student head from its detector weights."""

import unittest

import torch

from scripts.ablation.evaluate_cmp5L_depgate_audit import (
    split_model_head,
    summarize_gate_logits,
)


class TestCheckpointSplit(unittest.TestCase):
    def test_head_is_required_and_detector_weights_remain_unchanged(self):
        state = {
            "backbone.weight": object(),
            "depgate_head.net.0.weight": object(),
            "depgate_head.net.0.bias": object(),
        }
        detector, head = split_model_head(state, require_head=True)
        self.assertEqual(list(detector), ["backbone.weight"])
        self.assertEqual(list(head), ["net.0.weight", "net.0.bias"])
        self.assertIs(detector["backbone.weight"], state["backbone.weight"])

    def test_gate_on_refuses_a_baseline_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "depgate_head"):
            split_model_head({"backbone.weight": object()}, require_head=True)

    def test_gate_telemetry_reports_query_level_lambda(self):
        telemetry = summarize_gate_logits([[torch.tensor([0.0, 1.0986123])]], strength=0.2)
        self.assertEqual(telemetry[0]["n"], 2)
        self.assertAlmostEqual(telemetry[0]["prob"]["mean"], 0.625, places=5)
        self.assertAlmostEqual(telemetry[0]["lambda"]["min"], 0.85, places=5)
        self.assertAlmostEqual(telemetry[0]["lambda"]["max"], 0.90, places=5)


if __name__ == "__main__":
    unittest.main()
