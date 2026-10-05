import sys
import unittest
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ablation.cmp5L_privileged_decision_kd import (
    bernoulli_kl_from_logits,
    build_counterfactual_gates,
    build_mixed_oracle_output,
    build_stage1_subset_manifest,
    mean_layer_kd,
    max_named_buffer_change,
    run_with_all_score_heads,
    select_counterfactual_cases,
    select_outcome_targets,
    subset_coco_from_manifest,
)


class PrivilegedDecisionKDContractTest(unittest.TestCase):
    def test_counterfactual_gates_are_subset_and_exclusion_of_full_gate(self):
        full = torch.tensor([[True, False, True, True, False]])
        selected, excluded = build_counterfactual_gates(full, [[0, 1, 4]])
        self.assertTrue(torch.equal(selected, torch.tensor([[True, False, False, False, False]])))
        self.assertTrue(torch.equal(excluded, torch.tensor([[False, False, True, True, False]])))
        self.assertTrue(torch.equal(selected | excluded, full))
        with self.assertRaises(IndexError):
            build_counterfactual_gates(full, [[5]])

    def test_counterfactual_case_selection_is_deterministic_and_category_limited(self):
        rows = [
            {
                "image_id": 3,
                "positive": [{"kind": "repair", "gt_index": 0, "teacher_query": 7, "gt_label": 3}],
                "negative": [{"kind": "suppress", "teacher_query": 9, "class_index": 1}],
                "normal_gt_matches": {"1": 2},
                "privileged_gt_matches": {"1": 4},
            },
            {
                "image_id": 1,
                "positive": [{"kind": "protect", "gt_index": 0, "teacher_query": 2, "gt_label": 0}],
                "negative": [],
                "normal_gt_matches": {"0": 2},
                "privileged_gt_matches": {},
            },
            {
                "image_id": 2,
                "positive": [{"kind": "repair", "gt_index": 0, "teacher_query": 1, "gt_label": 1}],
                "negative": [],
                "normal_gt_matches": {},
                "privileged_gt_matches": {"0": 1},
            },
        ]
        left = select_counterfactual_cases(rows, per_category=1)
        right = select_counterfactual_cases(list(reversed(rows)), per_category=1)
        self.assertEqual(left, right)
        self.assertEqual(left["repair"][0]["image_id"], 2)
        self.assertEqual(left["protect"][0]["image_id"], 1)
        self.assertEqual(left["suppress"][0]["image_id"], 3)
        self.assertEqual(left["mixed"][0]["image_id"], 3)
        self.assertEqual(left["mixed"][0]["query_ids"], [7, 9])

    def test_buffer_change_audit_handles_boolean_and_numeric_buffers(self):
        before = {
            "flag": torch.tensor([True, False]),
            "value": torch.tensor([1.0, 2.0]),
        }
        unchanged = [("flag", before["flag"].clone()), ("value", before["value"].clone())]
        self.assertEqual(max_named_buffer_change(before, unchanged), 0.0)
        changed = [("flag", torch.tensor([False, False])), ("value", torch.tensor([1.0, 2.5]))]
        self.assertEqual(max_named_buffer_change(before, changed), 1.0)

    def test_all_layer_score_capture_restores_eval_state_and_final_output(self):
        class FakeEC(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = nn.Module()
                self.dec_score_head = nn.ModuleList([nn.Linear(2, 1) for _ in range(4)])

        model = FakeEC().eval()
        features = torch.tensor([[1.0, 2.0]])

        def branch():
            values = []
            layer_count = 4 if model.decoder.training else 1
            layer_indices = range(4) if layer_count == 4 else (3,)
            for index in layer_indices:
                values.append(model.dec_score_head[index](features))
            return {"pred_logits": values[-1]}

        expected = branch()["pred_logits"]
        output, layers = run_with_all_score_heads(model, branch)
        self.assertFalse(model.decoder.training)
        self.assertTrue(all(not head.training for head in model.dec_score_head))
        self.assertEqual(len(layers), 4)
        torch.testing.assert_close(output["pred_logits"], expected)
        torch.testing.assert_close(layers[-1], expected)

    def test_all_layer_capture_prefers_post_lqe_logits(self):
        class AddOne(nn.Module):
            def forward(self, logits, _corners):
                return logits + 1.0

        class FakeEC(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = nn.Module()
                self.decoder.lqe_layers = nn.ModuleList([AddOne() for _ in range(4)])
                self.dec_score_head = nn.ModuleList([nn.Linear(2, 1) for _ in range(4)])

        model = FakeEC().eval()
        features = torch.tensor([[1.0, 2.0]])

        def branch():
            values = []
            for index in range(4):
                raw = model.dec_score_head[index](features)
                values.append(model.decoder.lqe_layers[index](raw, None))
            return {"pred_logits": values[-1]}

        output, layers = run_with_all_score_heads(model, branch)
        torch.testing.assert_close(layers[-1], output["pred_logits"])
        for index, layer in enumerate(layers):
            raw = model.dec_score_head[index](features)
            torch.testing.assert_close(layer, raw + 1.0)

    def test_bernoulli_self_distillation_is_zero_and_teacher_is_detached(self):
        student = torch.tensor([[1.5, -0.7]], requires_grad=True)
        teacher = student.detach().clone().requires_grad_(True)
        loss = bernoulli_kl_from_logits(student, teacher)
        self.assertAlmostEqual(float(loss.detach()), 0.0, places=7)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)

    def test_four_layer_kd_is_layer_mean_not_fourfold_sum(self):
        student_layers = [torch.tensor([[0.0]], requires_grad=True) for _ in range(4)]
        teacher_layers = [torch.tensor([[2.0]]) for _ in range(4)]
        masks = [torch.ones_like(layer, dtype=torch.bool) for layer in student_layers]
        single = bernoulli_kl_from_logits(student_layers[0], teacher_layers[0], masks[0])
        combined = mean_layer_kd(student_layers, teacher_layers, masks)
        torch.testing.assert_close(combined, single)

    def test_result_selection_protects_repairs_and_suppresses_only_reduced_error_class(self):
        # q0/N protects GT0; q1/P repairs GT1; q2 is a conservative false
        # positive whose class-2 score is reduced by P; q3 is not reduced.
        normal = {
            "pred_logits": torch.tensor([[
                [5.0, -5.0, -5.0],
                [-5.0, -5.0, -5.0],
                [-5.0, -5.0, 4.0],
                [-5.0, 3.0, -5.0],
            ]]),
            "pred_boxes": torch.tensor([[
                [0.25, 0.25, 0.20, 0.20],
                [0.75, 0.75, 0.20, 0.20],
                [0.05, 0.05, 0.05, 0.05],
                [0.95, 0.05, 0.05, 0.05],
            ]]),
        }
        privileged = {
            "pred_logits": torch.tensor([[
                [-5.0, -5.0, -5.0],
                [-5.0, 5.0, -5.0],
                [-5.0, -5.0, -2.0],
                [-5.0, 4.0, -5.0],
            ]]),
            "pred_boxes": normal["pred_boxes"].clone(),
        }
        targets = [{
            "labels": torch.tensor([0, 1]),
            "boxes": torch.tensor([
                [0.25, 0.25, 0.20, 0.20],
                [0.75, 0.75, 0.20, 0.20],
            ]),
        }]

        selected = select_outcome_targets(normal, privileged, targets)[0]
        self.assertEqual(
            [(item["kind"], item["gt_index"], item["teacher_query"], item["branch"])
             for item in selected["positive"]],
            [("protect", 0, 0, "N"), ("repair", 1, 1, "P")],
        )
        self.assertEqual(
            [(item["teacher_query"], item["class_index"])
             for item in selected["negative"]],
            [(2, 2)],
        )
        self.assertEqual(selected["counts"]["uncertain_negative"], 1)

        mixed = build_mixed_oracle_output(normal, privileged, [selected])
        torch.testing.assert_close(mixed["pred_boxes"], normal["pred_boxes"])
        # Protect stays N, repair copies the complete P class vector, and
        # suppression copies only the verified erroneous class dimension.
        torch.testing.assert_close(mixed["pred_logits"][0, 0], normal["pred_logits"][0, 0])
        torch.testing.assert_close(mixed["pred_logits"][0, 1], privileged["pred_logits"][0, 1])
        self.assertEqual(
            float(mixed["pred_logits"][0, 2, 2]),
            float(privileged["pred_logits"][0, 2, 2]),
        )
        self.assertEqual(
            float(mixed["pred_logits"][0, 3, 1]),
            float(normal["pred_logits"][0, 3, 1]),
        )

    def test_empty_or_zero_pair_kd_is_differentiable_zero(self):
        student = torch.randn(2, 3, requires_grad=True)
        teacher = torch.randn(2, 3)
        mask = torch.zeros_like(student, dtype=torch.bool)
        loss = bernoulli_kl_from_logits(student, teacher, mask)
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(student.grad)

    def test_stage1_subset_manifest_is_deterministic_disjoint_and_stratified(self):
        categories = [{"id": i, "name": str(i)} for i in range(4)]
        images = [
            {"id": i, "file_name": f"{i}.png", "width": 100, "height": 100}
            for i in range(1, 41)
        ]
        annotations = []
        ann_id = 1
        # 1-8 empty; four class blocks thereafter. Even ids get two boxes,
        # and the first box is small, creating annotation-only hard cases.
        for image_id in range(9, 41):
            category = min((image_id - 9) // 8, 3)
            count = 2 if image_id % 2 == 0 else 1
            for index in range(count):
                side = 5 if index == 0 else 20
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category,
                    "bbox": [10 + index, 10 + index, side, side],
                    "area": side * side,
                    "iscrowd": 0,
                })
                ann_id += 1
        coco = {"images": images, "annotations": annotations, "categories": categories}
        requested = {"empty": 2, "duct": 2, "lymph": 2, "cystic": 2, "solid": 2, "hard": 2}

        left = build_stage1_subset_manifest(coco, seed=17, requested=requested)
        right = build_stage1_subset_manifest(coco, seed=17, requested=requested)
        self.assertEqual(left, right)
        rows = left["images"]
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({row["image_id"] for row in rows}), 12)
        counts = {name: 0 for name in requested}
        for row in rows:
            counts[row["stratum"]] += 1
        self.assertEqual(counts, requested)
        self.assertEqual(len(left["manifest_sha256"]), 64)

        subset = subset_coco_from_manifest(coco, left)
        selected_ids = {row["image_id"] for row in rows}
        self.assertEqual({image["id"] for image in subset["images"]}, selected_ids)
        self.assertTrue(
            all(annotation["image_id"] in selected_ids for annotation in subset["annotations"])
        )
        self.assertEqual(subset["categories"], categories)


if __name__ == "__main__":
    unittest.main()
