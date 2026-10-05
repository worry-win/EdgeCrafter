import unittest
from pathlib import Path

import torch

from ecdetseg.engine.edgecrafter.mae_dino_backbone import MAEDinoViTBackbone


class MAEDinoViTBackboneTest(unittest.TestCase):
    def test_real_vitb_checkpoint_loads_all_twelve_blocks_and_forwards_last_two(self):
        checkpoint = Path(
            "/cobot/Code/CODE/MAE_DINO/output/"
            "dinov2_visible_cls_croped1m_iter500000_base/"
            "mae_dino-checkpoint-500000iter.pth"
        )
        if not checkpoint.is_file():
            self.skipTest(f"Checkpoint is unavailable: {checkpoint}")

        model = MAEDinoViTBackbone(
            weights_path=str(checkpoint),
            out_feature_indexes=[10, 11],
            proj_dim=256,
        ).eval()

        self.assertEqual(len(model.encoder.blocks), 12)
        self.assertEqual(model.encoder.out_feature_indexes, (10, 11))
        self.assertEqual(model.loaded_parameter_groups, ("encoder",))
        self.assertEqual(model.loaded_tensor_count, 173)
        self.assertEqual(model.loaded_element_count, 85_665_792)

        with torch.no_grad():
            features = model(torch.randn(1, 3, 128, 128))

        self.assertEqual(
            [tuple(feature.shape) for feature in features],
            [(1, 256, 16, 16), (1, 256, 8, 8), (1, 256, 4, 4)],
        )


if __name__ == "__main__":
    unittest.main()
