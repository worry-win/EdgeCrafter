import unittest
import io
import tempfile
from pathlib import Path
import numpy as np
import torch

from scripts.ablation.cmp5L_ec_fullcycle_core import (
    augmentation_mask,
    partition_detection_losses,
    stage_ramp,
    pack_numpy_rng_state,
    unpack_numpy_rng_state,
    resolve_calibration_batch_path,
    calibration_coefficient_from_rank_ratios,
)


class FullCycleCoreTests(unittest.TestCase):
    def test_calibration_pools_rank_ratios_before_locking_one_coefficient(self):
        coefficient, median = calibration_coefficient_from_rank_ratios(
            [[.10, .20, .30], [.40, .50, .60]])
        self.assertAlmostEqual(median, .35)
        self.assertAlmostEqual(coefficient, .30 / .35)

    def test_relative_manifest_batch_resolves_beside_manifest_not_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            shared = Path(directory) / 'shared'
            shared.mkdir()
            manifest = shared / 'manifest.json'
            batch = shared / 'calibration_batch_00.pt'
            batch.write_bytes(b'locked')
            self.assertEqual(
                resolve_calibration_batch_path(
                    manifest, 'outputs/ablation/run/shared/calibration_batch_00.pt'),
                batch,
            )

    def test_numpy_rng_state_survives_weights_only_checkpoint(self):
        before = np.random.get_state()
        safe = pack_numpy_rng_state(before)
        buffer = io.BytesIO()
        torch.save({'rng': safe}, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, weights_only=True)['rng']
        restored = unpack_numpy_rng_state(loaded)
        self.assertEqual(restored[0], before[0])
        self.assertTrue(np.array_equal(restored[1], before[1]))
        self.assertEqual(restored[2:], before[2:])

    def test_stage_ramp_uses_continuous_full_cycle_progress(self):
        self.assertEqual(stage_ramp(9, 99, 100, 100), 0.0)
        self.assertEqual(stage_ramp(10, 0, 100, 100), 0.0)
        self.assertAlmostEqual(stage_ramp(15, 0, 100, 100), 0.5)
        self.assertEqual(stage_ramp(20, 0, 100, 100), 1.0)
        self.assertEqual(stage_ramp(99, 99, 100, 100), 1.0)


    def test_loss_partition_preserves_encoder_once_and_rejects_unknown(self):
        terms = {
            'loss_mal_enc_0': torch.tensor(2.),
            'loss_bbox_enc_0': torch.tensor(3.),
            'loss_giou_enc_0': torch.tensor(4.),
            'loss_mal': torch.tensor(5.),
            'loss_fgl_aux_2': torch.tensor(6.),
            'loss_bbox_dn_pre': torch.tensor(7.),
        }
        encoder, decoder = partition_detection_losses(terms)
        self.assertEqual(sum(encoder.values()), 9)
        self.assertEqual(sum(decoder.values()), 18)
        with self.assertRaisesRegex(ValueError, 'unknown'):
            partition_detection_losses({'loss_surprise': torch.tensor(1.)})


    def test_augmented_gt_mask_uses_union_and_empty_identity(self):
        targets = [
            {'boxes': torch.tensor([[.25, .5, .25, .5]])},
            {'boxes': torch.empty(0, 4)},
        ]
        mask = augmentation_mask(targets, height=4, width=4, outside_coefficients=[.3, .8])
        self.assertEqual(mask.shape, (2, 1, 4, 4))
        self.assertAlmostEqual(float(mask[0].min()), .3)
        self.assertAlmostEqual(float(mask[0].max()), 1.)
        self.assertTrue(torch.equal(mask[1], torch.ones_like(mask[1])))


if __name__ == '__main__':
    unittest.main()
