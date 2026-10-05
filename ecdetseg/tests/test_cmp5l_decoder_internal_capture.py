import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ecdetseg"))

from engine.edgecrafter.decoder import ECTransformer
from scripts.ablation.probe_cmp5L_decoder_internal_tensors import DecoderInternalCapture
from scripts.ablation.dump_cmp5L_internal_behavior import capture_ffn_gradient_importance


class DecoderInternalCaptureTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = ECTransformer(
            num_classes=4,
            hidden_dim=32,
            num_queries=5,
            feat_channels=[32, 32, 32],
            feat_strides=[8, 16, 32],
            num_levels=3,
            num_points=[1, 2, 1],
            nhead=4,
            num_layers=2,
            dim_feedforward=64,
            dropout=0.0,
            eval_spatial_size=None,
            eval_idx=-1,
        ).eval()
        self.feats = [
            torch.randn(1, 32, 8, 8),
            torch.randn(1, 32, 4, 4),
            torch.randn(1, 32, 2, 2),
        ]

    def test_capture_preserves_eval_outputs_and_replays_every_layer(self):
        with torch.no_grad():
            baseline = self.model(self.feats)
            with DecoderInternalCapture(self.model) as capture:
                observed = self.model(self.feats)
            replay = capture.replay_decoder_heads()

        self.assertFalse(self.model.training)
        self.assertTrue(all(not module.training for module in self.model.modules()))
        torch.testing.assert_close(observed["pred_logits"], baseline["pred_logits"], rtol=0, atol=0)
        torch.testing.assert_close(observed["pred_boxes"], baseline["pred_boxes"], rtol=0, atol=0)
        torch.testing.assert_close(replay["lqe_logits"][-1], observed["pred_logits"])
        torch.testing.assert_close(replay["boxes"][-1], observed["pred_boxes"])
        self.assertEqual(len(capture.layers), 2)

    def test_contract_exposes_attention_query_ffn_and_reference_stages(self):
        with torch.no_grad(), DecoderInternalCapture(self.model) as capture:
            self.model(self.feats)

        for layer in capture.layers:
            self.assertEqual(tuple(layer["reference_in"].shape), (1, 5, 1, 4))
            self.assertEqual(tuple(layer["sampling_offsets"].shape), (1, 5, 4, 4, 2))
            self.assertEqual(tuple(layer["sampling_locations"].shape), (1, 5, 4, 4, 2))
            self.assertEqual(tuple(layer["attention_weights"].shape), (1, 5, 4, 4))
            torch.testing.assert_close(
                layer["attention_weights"].sum(-1),
                torch.ones_like(layer["attention_weights"].sum(-1)),
            )
            for key in (
                "query_in", "self_attn_output", "self_attn_residual",
                "cross_attn_output", "cross_attn_residual", "ffn_linear1",
                "ffn_activation", "ffn_linear2", "query_out",
            ):
                self.assertIn(key, layer)
            self.assertFalse(torch.equal(layer["ffn_linear1"], layer["ffn_activation"]))

    def test_ffn_gradient_importance_is_finite_without_leaving_eval_mode(self):
        # Real YAML checkpoints may materialize fixed FDR scales as integer
        # Parameters. They must remain frozen because integer tensors cannot
        # participate in autograd.
        self.model.decoder.integer_scale = torch.nn.Parameter(
            torch.tensor([4], dtype=torch.int64), requires_grad=False
        )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        items = [{
            "batch_index": 0,
            "query_id": 0,
            "gt_class": 1,
            "gt_box_cxcywh": torch.tensor([0.5, 0.5, 0.2, 0.3]),
        }]
        importance = capture_ffn_gradient_importance(self.model, self.feats, items)
        self.assertEqual(tuple(importance.shape), (1, 2, 64))
        self.assertTrue(torch.isfinite(importance).all())
        self.assertGreater(float(importance.sum()), 0.0)
        self.assertFalse(self.model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in self.model.parameters()))


if __name__ == "__main__":
    unittest.main()
