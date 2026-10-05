import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from engine.edgecrafter.decoder import (
    QueryGateHead,
    TransformerDecoder,
    compute_query_gate_coefficients,
)
from engine.edgecrafter.utils import deformable_attention_core_func_v2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ablation.cmp5L_qg_behavior_distillation import (
    attach_query_gate_head,
    balanced_selection_bce,
    bernoulli_behavior_kd,
    broadcast_soft_gate_coeff,
    compose_qg_loss,
    match_competitive_candidates,
    match_selection_queries,
    max_named_buffer_change,
    slice_normal_query_layers,
    temporary_query_gate_state,
)


class QueryGateBehaviorDistillationContractTest(unittest.TestCase):
    def test_normal_query_layer_slice_excludes_dn_prefix(self):
        layers = [
            torch.arange(2 * 7 * 3).reshape(2, 7, 3) + 100 * index
            for index in range(4)
        ]
        normal = slice_normal_query_layers(layers, normal_queries=5)
        self.assertEqual(len(normal), 4)
        for source, sliced in zip(layers, normal):
            self.assertTrue(torch.equal(sliced, source[:, -5:]))
        with self.assertRaisesRegex(ValueError, "normal query count"):
            slice_normal_query_layers(layers, normal_queries=8)

    def test_qg_loss_composition_uses_locked_group_weights(self):
        detection = torch.tensor(2.0, requires_grad=True)
        selection = torch.tensor(3.0, requires_grad=True)
        behavior = torch.tensor(5.0, requires_grad=True)

        expected = {
            "QG0": 2.0,
            "QG1": 5.0,
            "QG2": 7.5,
            "QG3": 7.5,
        }
        for group, total in expected.items():
            result = compose_qg_loss(
                detection,
                selection,
                behavior,
                group=group,
            )
            self.assertAlmostEqual(float(result["total"].detach()), total)

        result = compose_qg_loss(
            detection,
            selection,
            behavior,
            group="QG3",
        )
        result["total"].backward()
        self.assertEqual(float(detection.grad), 1.0)
        self.assertEqual(float(selection.grad), 1.0)
        self.assertEqual(float(behavior.grad), 0.5)

    def test_buffer_audit_handles_boolean_and_numeric_tensors(self):
        before = {
            "flag": torch.tensor([True, False]),
            "scale": torch.tensor([1.0, 2.0]),
        }
        unchanged = [(name, value.clone()) for name, value in before.items()]
        self.assertEqual(max_named_buffer_change(before, unchanged), 0.0)
        changed = [
            ("flag", torch.tensor([False, False])),
            ("scale", torch.tensor([1.0, 2.25])),
        ]
        self.assertEqual(max_named_buffer_change(before, changed), 1.0)

    def test_temporary_gate_state_restores_after_success_and_exception(self):
        module = nn.Module()
        module.query_gate_enabled = True
        with temporary_query_gate_state(module, False):
            self.assertFalse(module.query_gate_enabled)
        self.assertTrue(module.query_gate_enabled)
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with temporary_query_gate_state(module, False):
                self.assertFalse(module.query_gate_enabled)
                raise RuntimeError("deliberate")
        self.assertTrue(module.query_gate_enabled)

    def test_transformer_decoder_reuses_one_l0_gate_on_l0_to_l2_only(self):
        class RecordingLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.cross_attn = nn.Module()
                self.cross_attn.num_heads = 2
                self.cross_attn.num_points_list = [1]
                self.sample_coefficients = None

            def forward(
                self,
                target,
                reference_points,
                value,
                spatial_shapes,
                attn_mask=None,
                query_pos_embed=None,
                sample_coefficients=None,
            ):
                self.sample_coefficients = sample_coefficients
                return target

        class ZeroHead(nn.Module):
            def __init__(self, width):
                super().__init__()
                self.width = width

            def forward(self, value):
                return value.new_zeros(value.shape[0], value.shape[1], self.width)

        class CountingHead(QueryGateHead):
            def __init__(self):
                super().__init__(in_dim=4, hidden_dim=3, prior_probability=0.01)
                self.calls = 0

            def forward(self, query):
                self.calls += 1
                return super().forward(query)

        decoder = TransformerDecoder(
            hidden_dim=4,
            decoder_layer=RecordingLayer(),
            decoder_layer_wide=RecordingLayer(),
            segmentation_head=None,
            num_layers=4,
            num_head=2,
            reg_max=1,
            reg_scale=1.0,
            up=1.0,
            eval_idx=3,
            use_fdr_decode=False,
            use_aux_distribution=False,
            use_lqe=False,
            use_pre_outputs=False,
        )
        decoder.query_gate_head = CountingHead()
        decoder.query_gate_enabled = True
        decoder.train()
        target = torch.randn(1, 7, 4)
        reference = torch.zeros(1, 7, 4)
        memory = torch.zeros(1, 1, 4)
        shapes = torch.tensor([[1, 1]])
        decoder(
            None,
            target,
            reference,
            memory,
            shapes,
            bbox_head=nn.ModuleList([ZeroHead(4) for _ in range(4)]),
            score_head=nn.ModuleList([ZeroHead(2) for _ in range(4)]),
            query_pos_head=ZeroHead(4),
            pre_bbox_head=None,
            integral=None,
            up=1.0,
            reg_scale=1.0,
            dn_meta={"dn_num_split": [2, 5]},
            continuous_bbox_head=nn.ModuleList([ZeroHead(4) for _ in range(4)]),
        )
        self.assertEqual(decoder.query_gate_head.calls, 1)
        self.assertEqual(tuple(decoder.last_query_gate_logits.shape), (1, 5))
        for layer in decoder.layers[:3]:
            self.assertIsNotNone(layer.sample_coefficients)
            self.assertEqual(tuple(layer.sample_coefficients.shape), (1, 7, 2, 1))
        self.assertIsNone(decoder.layers[3].sample_coefficients)
        self.assertTrue(torch.equal(
            decoder.layers[0].sample_coefficients,
            decoder.layers[2].sample_coefficients,
        ))

    def test_attached_gate_is_registered_and_survives_ema_style_copy_and_strict_load(self):
        class DummyDetector(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = nn.Module()
                self.decoder.decoder = nn.Module()
                self.decoder.decoder.hidden_dim = 4

        student = DummyDetector()
        head = attach_query_gate_head(student, hidden_dim=3, prior_probability=0.01)
        self.assertIs(student.decoder.decoder.query_gate_head, head)
        self.assertTrue(any("query_gate_head" in key for key in student.state_dict()))
        ema_copy = __import__("copy").deepcopy(student)
        restored = DummyDetector()
        attach_query_gate_head(restored, hidden_dim=3, prior_probability=0.01)
        restored.load_state_dict(ema_copy.state_dict(), strict=True)
        query = torch.randn(2, 5, 4)
        self.assertTrue(torch.equal(head(query), restored.decoder.decoder.query_gate_head(query)))
        with self.assertRaisesRegex(RuntimeError, "already installed"):
            attach_query_gate_head(student, hidden_dim=3, prior_probability=0.01)

    def test_native_gate_is_computed_once_from_l0_normal_queries_and_reused(self):
        class CountingHead(QueryGateHead):
            def __init__(self):
                super().__init__(in_dim=4, hidden_dim=3, prior_probability=0.01)
                self.calls = 0

            def forward(self, query):
                self.calls += 1
                return super().forward(query)

        head = CountingHead()
        query = torch.randn(2, 7, 4)
        result = compute_query_gate_coefficients(
            query,
            head,
            dn_meta={"dn_num_split": [2, 5]},
            heads=2,
            points=3,
        )
        self.assertEqual(head.calls, 1)
        self.assertEqual(tuple(result["logits"].shape), (2, 5))
        self.assertEqual(tuple(result["coefficients"].shape), (2, 7, 2, 3))
        self.assertTrue(torch.equal(result["coefficients"][:, :2], torch.ones(2, 2, 2, 3)))
        self.assertTrue(torch.allclose(result["probability"], torch.full((2, 5), 0.01)))
        self.assertTrue(torch.allclose(result["coefficients"][:, 2:], torch.full((2, 5, 2, 3), 0.992)))

    def test_attention_core_applies_optional_coeff_after_sampling(self):
        value = (torch.tensor([[[[2.0]]]]),)
        shapes = torch.tensor([[1, 1]])
        locations = torch.tensor([[[[[0.5, 0.5]]]]])
        attention = torch.ones(1, 1, 1, 1)
        normal = deformable_attention_core_func_v2(
            value, shapes, locations, attention, [1]
        )
        gated = deformable_attention_core_func_v2(
            value,
            shapes,
            locations,
            attention,
            [1],
            sample_coefficients=torch.full((1, 1, 1, 1), 0.2),
        )
        self.assertTrue(torch.allclose(normal, torch.tensor([[[2.0]]])))
        self.assertTrue(torch.allclose(gated, 0.2 * normal))

    def test_competitive_matching_uses_score_class_iou_and_excludes_positives(self):
        teacher_logits = torch.tensor([
            [[6.0, -6.0], [-6.0, 5.0], [-8.0, -8.0]],
            [[-8.0, -8.0], [-8.0, -8.0], [-8.0, -8.0]],
        ])
        teacher_boxes = torch.tensor([
            [[0.2, 0.2, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2], [0.5, 0.5, 0.1, 0.1]],
            [[0.2, 0.2, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2], [0.5, 0.5, 0.1, 0.1]],
        ])
        student_logits = torch.tensor([
            [[-5.0, 4.0], [5.5, -5.0], [4.0, -4.0]],
            [[-8.0, -8.0], [-8.0, -8.0], [-8.0, -8.0]],
        ])
        student_boxes = torch.tensor([
            [[0.71, 0.70, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]],
            [[0.2, 0.2, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2], [0.5, 0.5, 0.1, 0.1]],
        ])
        positive = torch.tensor([[0, 1, 0]], dtype=torch.long)
        result = match_competitive_candidates(
            student_logits,
            student_boxes,
            teacher_logits,
            teacher_boxes,
            positive_pairs=positive,
            top_k=20,
            score_threshold=0.05,
            min_iou=0.30,
        )
        self.assertTrue(torch.equal(result["pairs"], torch.tensor([[0, 0, 1]])))
        self.assertEqual(result["student_candidate_count"], 2)
        self.assertEqual(result["teacher_candidate_count"], 1)
        self.assertEqual(result["records"][0]["top_class"], 1)
        self.assertGreaterEqual(result["records"][0]["iou"], 0.30)

    def test_behavior_kd_detaches_teacher_separates_pools_and_averages_layers(self):
        teacher = torch.tensor(
            [[[2.0, -1.0], [-2.0, 1.0], [0.5, -0.5]]], requires_grad=True
        )
        student = torch.zeros_like(teacher, requires_grad=True)
        positive = torch.tensor([[0, 0, 0]], dtype=torch.long)
        competitive = torch.tensor([[0, 1, 1]], dtype=torch.long)

        one_layer = bernoulli_behavior_kd(
            [student], [teacher], positive_pairs=positive, competitive_pairs=competitive
        )
        four_layers = bernoulli_behavior_kd(
            [student, student, student, student],
            [teacher, teacher, teacher, teacher],
            positive_pairs=positive,
            competitive_pairs=competitive,
        )
        self.assertGreater(float(one_layer["loss"].detach()), 0.0)
        self.assertTrue(torch.allclose(four_layers["loss"], one_layer["loss"]))
        self.assertEqual(one_layer["positive_count"], 1)
        self.assertEqual(one_layer["competitive_count"], 1)
        one_layer["loss"].backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)

        empty = torch.empty((0, 3), dtype=torch.long)
        no_pairs = bernoulli_behavior_kd(
            [student], [teacher], positive_pairs=empty, competitive_pairs=empty
        )
        self.assertEqual(float(no_pairs["loss"].detach()), 0.0)
        no_pairs["loss"].backward()

    def test_selection_query_mapping_prioritizes_gt_then_matches_permuted_tokens(self):
        teacher_query = torch.eye(3)
        teacher_ref = torch.tensor([
            [0.1, 0.1, 0.1, 0.1],
            [0.5, 0.5, 0.1, 0.1],
            [0.9, 0.9, 0.1, 0.1],
        ])
        permutation = torch.tensor([2, 0, 1])
        student_query = teacher_query[permutation]
        student_ref = teacher_ref[permutation]
        result = match_selection_queries(
            teacher_query,
            teacher_ref,
            student_query,
            student_ref,
            teacher_gt_to_query={0: 0},
            student_gt_to_query={0: 1},
            min_cosine=0.5,
            min_reference_iou=0.3,
        )
        pairs = {(item["teacher_query"], item["student_query"]): item["source"] for item in result}
        self.assertEqual(pairs[(0, 1)], "gt_identity")
        self.assertEqual(pairs[(1, 2)], "init_token_reference")
        self.assertEqual(pairs[(2, 0)], "init_token_reference")
        self.assertNotEqual({(i, i) for i in range(3)}, set(pairs))

    def test_soft_gate_coeff_protects_dn_prefix_and_has_locked_range(self):
        probability = torch.tensor([[0.0, 1.0, 0.25]])
        coeff = broadcast_soft_gate_coeff(
            probability, total_queries=5, normal_queries=3, heads=2, points=4
        )
        self.assertEqual(tuple(coeff.shape), (1, 5, 2, 4))
        self.assertTrue(torch.equal(coeff[:, :2], torch.ones_like(coeff[:, :2])))
        expected = torch.tensor([1.0, 0.2, 0.8]).view(1, 3, 1, 1).expand(1, 3, 2, 4)
        self.assertTrue(torch.allclose(coeff[:, -3:], expected))
        self.assertGreaterEqual(float(coeff.min()), 0.2 - 1e-6)
        self.assertLessEqual(float(coeff.max()), 1.0)

    def test_selection_loss_balances_present_groups_and_handles_empty_group(self):
        logits = torch.tensor([[-2.0, 2.0, 0.0]], requires_grad=True)
        labels = torch.tensor([[0.0, 1.0, 1.0]])
        valid = torch.tensor([[True, True, False]])
        result = balanced_selection_bce(logits, labels, valid)
        expected_protect = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, :1], labels[:, :1])
        expected_suppress = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, 1:2], labels[:, 1:2])
        self.assertTrue(torch.allclose(result["loss"], 0.5 * (expected_protect + expected_suppress)))
        self.assertEqual(result["protect_count"], 1)
        self.assertEqual(result["suppress_count"], 1)

        only_protect = balanced_selection_bce(logits, labels, torch.tensor([[True, False, False]]))
        self.assertTrue(torch.allclose(only_protect["loss"], expected_protect))
        none = balanced_selection_bce(logits, labels, torch.zeros_like(valid))
        self.assertEqual(float(none["loss"]), 0.0)
        none["loss"].backward()
        self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()
