"""First V tracer: arm identity and shared two-GPU contract."""
import unittest
import torch
from types import SimpleNamespace

from scripts.ablation import train_cmp5L_ec_v as v
from scripts.ablation.cmp5L_ec_v_losses import (
    layerwise_ranking_kd, candidate_set_kd, np_increment_ranking_kd,
    quality_filtered_box_kd,
)


class VContractTests(unittest.TestCase):
    def test_reference_recipes_have_distinct_teacher_modes(self):
        self.assertEqual(v.recipe('V0'), ('ECX0', 'Y0'))
        self.assertEqual(v.recipe('V1'), ('ECX3', 'Y1'))
        with self.assertRaises(ValueError):
            v.recipe('V8')

    def test_two_gpu_batch_contract(self):
        valid = {
            'train_dataloader': {'total_batch_size': 32,
                                 'dataset': {'transforms': {'mosaic_epoch': 24, 'stop_epoch': 98}},
                                 'collate_fn': {'mixup_epoch': 24}},
            'gradient_accumulation_steps': 1, 'sync_bn': True,
            'epochs': 100, 'early_stop_patience': 0,
        }
        v.validate_v_execution(valid, world_size=2)
        valid['gradient_accumulation_steps'] = 2
        with self.assertRaises(RuntimeError):
            v.validate_v_execution(valid, world_size=2)

    def test_v2_averages_four_independent_ranking_layers(self):
        class Matcher:
            def __call__(self, outputs, targets):
                return {'indices': [(torch.tensor([0]), torch.tensor([0]))]}

        target = [{'boxes': torch.tensor([[.5, .5, .2, .2]]), 'labels': torch.tensor([0])}]
        boxes = torch.tensor([[[.5,.5,.2,.2],[.1,.1,.1,.1],[.9,.9,.1,.1]]])
        teacher = torch.tensor([[[3.], [0.], [-1.]]]).repeat(4,1,1,1)
        layers = [torch.tensor([[[1.], [0.], [-1.]]], requires_grad=True) for _ in range(4)]
        outputs = {'pred_logits':layers[-1], 'pred_boxes':boxes,
                   'aux_outputs':[{'pred_logits':layers[i], 'pred_boxes':boxes} for i in range(3)]}
        result = layerwise_ranking_kd(outputs, teacher, target, Matcher(), global_image_count=1, ddp_world_size=1,
                                      normal_query_count=3, negative_topk=2)
        self.assertEqual(len(result['layer_losses']), 4)
        self.assertAlmostEqual(float(result['loss']), float(torch.stack(result['layer_losses']).mean()), places=6)
        result['loss'].backward()
        self.assertTrue(all(x.grad is not None and torch.isfinite(x.grad).all() for x in layers))

    def test_v2_teacher_layer_capture_restores_eval_state(self):
        inner = torch.nn.Sequential(torch.nn.Linear(2, 2))
        inner.eval()
        teacher = SimpleNamespace(decoder=SimpleNamespace(decoder=inner))
        with v.teacher_all_layer_outputs(teacher):
            self.assertTrue(inner.training)
            self.assertFalse(inner[0].training)
        self.assertFalse(inner.training)
        with self.assertRaisesRegex(RuntimeError, 'sentinel'):
            with v.teacher_all_layer_outputs(teacher):
                raise RuntimeError('sentinel')
        self.assertFalse(inner.training)

    def test_v3_softmax_is_over_candidates_and_skips_reverse_teacher(self):
        student = torch.tensor([[[.1, -4.], [2., 9.], [0., 8.]]], requires_grad=True)
        teacher = torch.tensor([[[3., -4.], [1., 9.], [0., 8.]]])
        boxes = torch.tensor([[[.5,.5,.2,.2],[.1,.1,.1,.1],[.9,.9,.1,.1]]])
        target = [{'boxes':torch.tensor([[.5,.5,.2,.2]]), 'labels':torch.tensor([0])}]
        matched = [(torch.tensor([0]), torch.tensor([0]))]
        result = candidate_set_kd(student, teacher, boxes, target, matched,
                                  global_image_count=1, ddp_world_size=1,
                                  normal_query_count=3, negative_topk=2)
        self.assertEqual(result['valid_pair_count'], 2)
        self.assertGreater(float(result['loss']), 0)
        result['loss'].backward()
        self.assertEqual(float(student.grad[...,1].abs().sum()), 0.)
        reverse = candidate_set_kd(student.detach(), teacher.flip(1), boxes, target, matched,
                                   global_image_count=1, ddp_world_size=1,
                                   normal_query_count=3, negative_topk=2)
        self.assertEqual(reverse['valid_pair_count'], 0)
        self.assertEqual(float(reverse['loss']), 0.)

    def test_v5_weights_only_positive_np_increment_without_changing_denominator(self):
        student = torch.tensor([[[.1], [0.]]], requires_grad=True)
        privileged = torch.tensor([[[2.], [0.]]])
        normal = torch.tensor([[[1.], [0.]]])
        boxes = torch.tensor([[[.5,.5,.2,.2],[.1,.1,.1,.1]]])
        target = [{'boxes':torch.tensor([[.5,.5,.2,.2]]), 'labels':torch.tensor([0])}]
        matched = [(torch.tensor([0]), torch.tensor([0]))]
        result = np_increment_ranking_kd(student, privileged, normal, boxes, target, matched,
                                         normal_query_count=2, negative_topk=1,
                                         global_image_count=1, ddp_world_size=1)
        self.assertEqual(result['valid_pair_count'], 1)
        self.assertAlmostEqual(result['weight_sum'], 1.)
        result['loss'].backward()
        self.assertGreater(float(student.grad.abs().sum()), 0.)
        zero = np_increment_ranking_kd(student, normal, privileged, boxes, target, matched,
                                       normal_query_count=2, negative_topk=1,
                                       global_image_count=1, ddp_world_size=1)
        self.assertEqual(float(zero['loss']), 0.)

    def test_v4_only_distills_teacher_box_that_improves_same_gt(self):
        student = torch.tensor([[[.5,.5,.5,.5]]], requires_grad=True)
        teacher = torch.tensor([[[.5,.5,.2,.2]]])
        target = [{'boxes':torch.tensor([[.5,.5,.2,.2]]), 'labels':torch.tensor([3])}]
        matched = [(torch.tensor([0]), torch.tensor([0]))]
        result = quality_filtered_box_kd(student, teacher, target, matched,
                                         global_image_count=1, ddp_world_size=1)
        self.assertEqual(result['active_count'], 1)
        self.assertEqual(result['active_class_counts']['3'], 1)
        self.assertGreater(float(result['loss']), 0.)
        result['loss'].backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        rejected = quality_filtered_box_kd(teacher, teacher, target, matched,
                                           global_image_count=1, ddp_world_size=1)
        self.assertEqual(rejected['active_count'], 0)
        self.assertEqual(float(rejected['loss']), 0.)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA device-index regression test')
    def test_v4_cpu_matcher_indices_with_cuda_boxes(self):
        student = torch.tensor([[[.5, .5, .5, .5]]], device='cuda', requires_grad=True)
        teacher = torch.tensor([[[.5, .5, .2, .2]]], device='cuda')
        target = [{'boxes': torch.tensor([[.5, .5, .2, .2]], device='cuda'),
                   'labels': torch.tensor([3], device='cuda')}]
        matches = [(torch.tensor([0]), torch.tensor([0]))]
        result = quality_filtered_box_kd(student, teacher, target, matches,
                                         global_image_count=1, ddp_world_size=1)
        self.assertEqual(result['active_count'], 1)

    def test_v6_teacher_view_changes_only_background_strength(self):
        before = v.ACTIVE
        try:
            v.ACTIVE = SimpleNamespace(args=SimpleNamespace(v_arm='V6'))
            warm = v.teacher_runtime(object(), 'ECX3')
            self.assertEqual(warm.spec.teacher_memory_mode, 'privileged_np')
            self.assertEqual(warm.spec.teacher_background_weight, .5)
            v.ACTIVE = SimpleNamespace(args=SimpleNamespace(v_arm='V1'))
            standard = v.teacher_runtime(object(), 'ECX3')
            self.assertEqual(standard.spec.teacher_background_weight, .2)
        finally:
            v.ACTIVE = before

    def test_v7_student_intervention_does_not_taper_with_kd(self):
        self.assertEqual(v.v7_intervention_factor(base_ramp=1., late_kd_factor=0.), 1.)
        self.assertEqual(v.v7_intervention_factor(base_ramp=.4, late_kd_factor=.2), .4)
        self.assertEqual(v.v7_intervention_factor(base_ramp=0., late_kd_factor=1., smoke=True), .5)

    def test_calibration_coverage_is_global_across_ranks(self):
        pooled, classes = v.aggregate_calibration_coverage(
            [[.1, .2], [.3, .4]],
            [{'0': 1, '1': 0, '2': 1, '3': 0},
             {'0': 0, '1': 1, '2': 0, '3': 1}])
        self.assertEqual(len(pooled), 4)
        self.assertEqual(classes, {'0': 1, '1': 1, '2': 1, '3': 1})
        with self.assertRaisesRegex(RuntimeError, 'insufficient effective calibration'):
            v.aggregate_calibration_coverage(
                [[.1], [.2]],
                [{'0': 1, '1': 0, '2': 0, '3': 0},
                 {'0': 0, '1': 1, '2': 1, '3': 1}])

    def test_v4_box_calibration_waits_for_epoch10_student(self):
        self.assertFalse(v.should_calibrate_box('V4', smoke=True))
        self.assertTrue(v.should_calibrate_box('V4', smoke=False))
        self.assertFalse(v.should_calibrate_box('V1', smoke=False))

    def test_v7_native_forward_binds_to_calibration_copy(self):
        class Tiny(torch.nn.Module):
            def forward(self, value):
                return value + self.offset
        source = Tiny()
        source.offset = 1
        original = source.forward
        source.forward = lambda value: original(value) + 100
        copied = __import__('copy').deepcopy(source)
        copied.offset = 7
        with v.student_native_forward(copied):
            self.assertEqual(copied.forward(1), 8)
        self.assertEqual(copied.forward(1), 102)

    def test_v7_encoder_prediction_identity_not_loss_equality(self):
        tensor = torch.randn(1, 3, 4)
        normal = {'enc_aux_outputs': [{'pred_boxes': tensor}]}
        replay = {'enc_aux_outputs': [{'pred_boxes': tensor}]}
        v.assert_shared_encoder_predictions(normal, replay)
        replay['enc_aux_outputs'][0]['pred_boxes'] = tensor.clone()
        with self.assertRaisesRegex(RuntimeError, 'encoder prediction'):
            v.assert_shared_encoder_predictions(normal, replay)


if __name__ == '__main__':
    unittest.main()
