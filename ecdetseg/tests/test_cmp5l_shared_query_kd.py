import unittest
from dataclasses import replace

import torch

from scripts.ablation.cmp5L_shared_query_kd import (
    compose_shared_query_loss,
    resolve_kq_spec,
    validate_kq_config,
    SharedQueryReplay,
    cosine_query_kd,
    layer_query_audit,
    normal_query_slice,
    shared_query_alignment_audit,
    sigmoid_output_kd,
)


class SharedQueryKDContractTest(unittest.TestCase):
    def test_kq_group_contract_is_locked(self):
        expected = {
            "KQ0": (False, "none", ()),
            "KQ1": (True, "privileged_np", (3,)),
            "KQ2": (True, "privileged_np", (0, 1, 2, 3)),
            "KQ3": (True, "normal", (0, 1, 2, 3)),
        }
        for group, values in expected.items():
            spec = resolve_kq_spec(group)
            self.assertEqual(
                (spec.enabled, spec.teacher_memory_mode, spec.layers), values
            )
            self.assertEqual(spec.weight, 0.5)
            self.assertEqual(spec.num_queries, 300)
        with self.assertRaises(ValueError):
            resolve_kq_spec("KQ4")

    def test_total_loss_uses_locked_half_weight_and_kq0_has_no_kd(self):
        detection = torch.tensor(4.0, requires_grad=True)
        kd = torch.tensor(2.0, requires_grad=True)
        self.assertEqual(float(compose_shared_query_loss(detection, kd, resolve_kq_spec("KQ2"))), 5.0)
        control = compose_shared_query_loss(detection, None, resolve_kq_spec("KQ0"))
        self.assertIs(control, detection)
        with self.assertRaises(ValueError):
            compose_shared_query_loss(detection, None, resolve_kq_spec("KQ1"))

    def test_kqw_only_increases_four_layer_hidden_weight(self):
        spec = resolve_kq_spec("KQW")
        self.assertEqual(spec.teacher_memory_mode, "privileged_np")
        self.assertEqual(spec.layers, (0, 1, 2, 3))
        self.assertEqual(spec.weight, 5.0)
        self.assertEqual(spec.output_weight, 0.0)
        detection = torch.tensor(4.0)
        hidden = torch.tensor(2.0)
        self.assertEqual(
            float(compose_shared_query_loss(detection, hidden, spec)), 14.0
        )

    def test_sigmoid_output_kd_uses_all_classes_normal_suffix_and_detaches_teacher(self):
        # Two DN queries precede the three normal queries in the Student tensor.
        student = torch.tensor(
            [[[9.0, -9.0], [8.0, -8.0], [0.2, -0.3], [0.5, 0.7], [-0.4, 0.1]]],
            requires_grad=True,
        )
        teacher = torch.tensor(
            [[[0.1, -0.2], [0.6, 0.8], [-0.5, 0.2]]],
            requires_grad=True,
        )
        result = sigmoid_output_kd(student, teacher, normal_query_count=3)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            student[:, -3:], teacher.detach().sigmoid(), reduction="mean"
        )
        self.assertTrue(torch.allclose(result["loss"], expected))
        self.assertEqual(result["query_count"], 3)
        self.assertEqual(result["class_count"], 2)
        result["loss"].backward()
        self.assertTrue(torch.equal(student.grad[:, :2], torch.zeros_like(student.grad[:, :2])))
        self.assertGreater(float(student.grad[:, -3:].abs().sum()), 0.0)
        self.assertIsNone(teacher.grad)

    def test_total_loss_keeps_hidden_and_output_weights_separate(self):
        spec = replace(resolve_kq_spec("KQ2"), output_weight=1.25)
        detection = torch.tensor(4.0)
        hidden = torch.tensor(2.0)
        output = torch.tensor(3.0)
        total = compose_shared_query_loss(
            detection, hidden, spec, output_kd_loss=output
        )
        self.assertEqual(float(total), 4.0 + 0.5 * 2.0 + 1.25 * 3.0)
        with self.assertRaises(ValueError):
            compose_shared_query_loss(detection, hidden, spec)
        with self.assertRaises(ValueError):
            compose_shared_query_loss(
                detection, hidden, resolve_kq_spec("KQ2"), output_kd_loss=output
            )

    def test_kqo_contract_locks_calibrated_full_output_weight(self):
        spec = resolve_kq_spec("KQO")
        self.assertEqual(spec.teacher_memory_mode, "privileged_np")
        self.assertEqual(spec.layers, (0, 1, 2, 3))
        self.assertEqual(spec.weight, 0.5)
        self.assertEqual(spec.output_weight, 50.82931828609598)

    def test_resolved_yaml_contract_cannot_silently_disagree_with_group(self):
        config = {
            "enabled": True,
            "teacher_memory_mode": "privileged_np",
            "layers": [3],
            "weight": 0.5,
            "num_queries": 300,
            "share_student_initial_query": True,
            "share_student_topk": True,
            "detach_teacher": True,
            "loss": "cosine",
            "output_weight": 0.0,
            "output_loss": "none",
            "output_layer": 3,
            "output_all_normal_queries": True,
        }
        validate_kq_config(config, resolve_kq_spec("KQ1"))
        for key, bad in (("weight", 1.0), ("layers", [0, 1, 2, 3]), ("share_student_topk", False)):
            changed = dict(config)
            changed[key] = bad
            with self.assertRaises(ValueError):
                validate_kq_config(changed, resolve_kq_spec("KQ1"))

    def test_normal_query_slice_excludes_dn_prefix(self):
        value = torch.arange(2 * 7 * 3).reshape(2, 7, 3)
        self.assertTrue(torch.equal(normal_query_slice(value, 5), value[:, -5:]))
        with self.assertRaises(ValueError):
            normal_query_slice(value, 8)

    def test_cosine_kd_detaches_teacher_and_averages_selected_layers(self):
        teacher = [torch.randn(1, 3, 4, requires_grad=True) for _ in range(4)]
        student = [torch.zeros_like(value, requires_grad=True) for value in teacher]
        result = cosine_query_kd(student, teacher, layers=(1, 3))
        self.assertEqual(result["layers"], [1, 3])
        self.assertGreater(float(result["loss"]), 0.0)
        result["loss"].backward()
        self.assertIsNotNone(student[1].grad)
        self.assertIsNotNone(student[3].grad)
        self.assertIsNone(teacher[1].grad)
        self.assertIsNone(teacher[3].grad)

    def test_alignment_requires_exact_query_reference_and_topk_identity(self):
        query = torch.randn(2, 3, 4)
        reference = torch.randn(2, 3, 4)
        indices = torch.tensor([[2, 1, 0], [0, 2, 1]])
        student = SharedQueryReplay(query, reference, indices, [], [], normal_query_count=3)
        teacher = SharedQueryReplay(query.clone(), reference.clone(), indices.clone(), [], [], normal_query_count=3)
        audit = shared_query_alignment_audit(student, teacher)
        self.assertTrue(all(audit[key]["exact"] for key in ("initial_query", "initial_reference_unactivated", "topk_indices")))
        teacher.topk_indices[0, 0] = 1
        self.assertFalse(shared_query_alignment_audit(student, teacher)["topk_indices"]["exact"])

    def test_layer_audit_checks_full_decoder_output_shapes(self):
        student = [torch.randn(2, 3, 4) for _ in range(4)]
        teacher = [value.clone() for value in student]
        rows = layer_query_audit(student, teacher)
        self.assertEqual([row["shape"] for row in rows], [[2, 3, 4]] * 4)
        self.assertTrue(all(abs(row["cosine_mean"] - 1.0) < 1e-6 for row in rows))


if __name__ == "__main__":
    unittest.main()
