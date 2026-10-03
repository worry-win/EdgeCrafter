"""Dataset adaptation contracts for the six-class kidney EC-V study."""
import unittest

import torch

from scripts.ablation.train_cmp5L_ec_v import aggregate_calibration_coverage
from scripts.ablation.cmp5L_ec_v_losses import quality_filtered_box_kd


class KidneyVClassTests(unittest.TestCase):
    def test_calibration_requires_all_six_classes(self):
        counts = {str(i): 1 for i in range(6)}
        _, merged = aggregate_calibration_coverage(
            [[1.0, 1.1, 0.9]], [counts], num_classes=6)
        self.assertEqual(merged, counts)
        counts['5'] = 0
        with self.assertRaisesRegex(RuntimeError, 'insufficient effective calibration'):
            aggregate_calibration_coverage([[1.0, 1.1, 0.9]], [counts], num_classes=6)

    def test_box_kd_counts_kidney_class_five(self):
        result = quality_filtered_box_kd(
            torch.tensor([[[.5, .5, .5, .5]]], requires_grad=True),
            torch.tensor([[[.5, .5, .2, .2]]]),
            [{'boxes': torch.tensor([[.5, .5, .2, .2]]),
              'labels': torch.tensor([5])}],
            [(torch.tensor([0]), torch.tensor([0]))],
            global_image_count=1, ddp_world_size=1, num_classes=6)
        self.assertEqual(result['active_class_counts']['5'], 1)

class KidneyVConfigTests(unittest.TestCase):
    def test_all_v_configs_resolve_to_six_class_csg_dataset(self):
        from pathlib import Path
        from engine.core.yaml_utils import load_config

        root = Path(__file__).resolve().parents[1] / 'configs' / 'ecdet'
        data = Path('/opt/public/wangzhiwei/Ultrasound_Data/COCO_Output/肾脏')
        for index in range(9):
            with self.subTest(arm=index):
                cfg = load_config(str(root / f'ecdet_l_dinov2s_kidney_ECV{index}.yml'), {})
                self.assertEqual(cfg['ec_v_arm'], f'V{index}')
                self.assertEqual(cfg['num_classes'], 6)
                self.assertEqual(cfg['train_dataloader']['dataset']['img_folder'], str(data / 'images_CSG_200m'))
                self.assertEqual(cfg['train_dataloader']['dataset']['ann_file'], str(data / 'annotations/Lesion_Det/v2_260729/train.json'))
                self.assertEqual(cfg['val_dataloader']['dataset']['ann_file'], str(data / 'annotations/Lesion_Det/v2_260729/val.json'))
                self.assertEqual(cfg['DinoV2Adapter']['weights_path'], '/opt/public/wanrui/model/dinov2/dinov2_vits14_reg4_pretrain.pth')
                self.assertEqual(cfg['ECTransformer']['num_layers'], 4)

class KidneyCalibrationSelectionTests(unittest.TestCase):
    def test_selects_train_only_pairs_for_each_class_and_empty_control(self):
        from scripts.ablation.prepare_kidney_ec_v import select_calibration_image_pairs

        images = [{'id': i} for i in range(20)]
        annotations = [{'image_id': 2 * c + offset, 'category_id': c}
                       for c in range(6) for offset in (0, 1)]
        pairs = select_calibration_image_pairs(
            {'images': images, 'annotations': annotations}, num_classes=6)
        self.assertEqual(len(pairs), 7)
        self.assertEqual([category for category, _ in pairs[:6]], list(range(6)))
        self.assertEqual(pairs[-1][0], None)
        self.assertTrue(all(len(ids) == 2 for _, ids in pairs))
        self.assertEqual(len({image for _, ids in pairs for image in ids}), 14)

class KidneyExpandedCalibrationSelectionTests(unittest.TestCase):
    def test_multiple_pairs_per_class_are_disjoint(self):
        from scripts.ablation.prepare_kidney_ec_v import select_calibration_image_pairs

        images = [{'id': i} for i in range(100)]
        annotations = [{'image_id': 12 * c + offset, 'category_id': c}
                       for c in range(6) for offset in range(12)]
        pairs = select_calibration_image_pairs(
            {'images': images, 'annotations': annotations},
            num_classes=6, pairs_per_class=5)
        self.assertEqual(len(pairs), 31)
        self.assertEqual([category for category, _ in pairs[:30]],
                         [category for category in range(6) for _ in range(5)])
        self.assertEqual(len({image for _, pair in pairs for image in pair}), 62)
