import unittest
import re

import torch

from ecdetseg.engine.core.yaml_utils import load_config
from ecdetseg.engine.edgecrafter.lwdetr_backbone import LWDetrBackbone


class LWDetrBackboneTest(unittest.TestCase):
    def test_lwdetr_xlarge_encoder_neck_forward_contract(self):
        model = LWDetrBackbone(weights_path=None)
        model.eval()

        with torch.no_grad():
            features = model(torch.randn(1, 3, 128, 128))

        self.assertEqual(
            [tuple(feature.shape) for feature in features],
            [(1, 384, 16, 16), (1, 384, 4, 4)],
        )

    def test_lwdetr_loader_rejects_non_lw_checkpoint(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "not_lwdetr.pth"
            torch.save({"model": {"unrelated.weight": torch.ones(1)}}, checkpoint)
            with self.assertRaisesRegex(KeyError, "backbone.0.encoder"):
                LWDetrBackbone(weights_path=str(checkpoint))

    def test_all_global_encoder_has_no_window_blocks(self):
        model = LWDetrBackbone(weights_path=None, window_block_indexes=[])
        self.assertEqual(model.encoder.window_block_indexes, ())
        self.assertFalse(any(block.window for block in model.encoder.blocks))

    def test_lw_encoder_neck_ecdet_x_config_contract(self):
        config = load_config(
            "ecdetseg/configs/ecdet/ecdet_x_lw_xlarge_encoder_neck_liver_ignore9.yml"
        )
        self.assertEqual(config["ECDet"]["backbone"], "LWDetrBackbone")
        self.assertEqual(config["ECDet"]["encoder"], "IdentityEncoder")
        self.assertEqual(config["ECTransformer"]["feat_channels"], [384, 384])
        self.assertEqual(config["ECTransformer"]["feat_strides"], [8, 32])
        self.assertEqual(config["ECTransformer"]["num_levels"], 2)
        self.assertEqual(config["ECTransformer"]["num_points"], [6, 6])
        self.assertEqual(config["ECTransformer"]["num_layers"], 4)
        self.assertEqual(config["ECTransformer"]["hidden_dim"], 256)

    def test_delete_image_allglobal_ec_lr_config_contract(self):
        config = load_config(
            "ecdetseg/configs/ecdet/"
            "ecdet_x_lw_xlarge_liver_delete_image_allglobal_ec_lr.yml"
        )
        self.assertEqual(config["num_classes"], 7)
        self.assertEqual(config["LWDetrBackbone"]["window_block_indexes"], [])
        self.assertTrue(config["train_dataloader"]["dataset"]["ann_file"].endswith(
            "/Lesion/Ignore_Delete-Image/train.json"
        ))
        self.assertTrue(config["val_dataloader"]["dataset"]["ann_file"].endswith(
            "/Lesion/Ignore_Delete-Image/valid.json"
        ))
        strict_patterns = [
            group["params"]
            for group in config["optimizer"]["params"]
            if group.get("lr") == 2.5e-6
        ]
        self.assertTrue(any(re.findall(pattern, "backbone.encoder.blocks.0.attn.qkv.weight")
                            for pattern in strict_patterns))
        self.assertTrue(any(re.findall(pattern, "backbone.projector.stages.0.0.0.conv.weight")
                            for pattern in strict_patterns))

    def test_delete_image_allglobal_discriminative_lr_config_contract(self):
        config = load_config(
            "ecdetseg/configs/ecdet/"
            "ecdet_x_lw_xlarge_liver_delete_image_allglobal_discriminative_lr.yml"
        )
        self.assertEqual(config["LWDetrBackbone"]["window_block_indexes"], [])
        encoder_lrs = {
            group["lr"]
            for group in config["optimizer"]["params"]
            if group["params"].startswith("^(?=.*backbone\\.encoder)")
        }
        projector_lrs = {
            group["lr"]
            for group in config["optimizer"]["params"]
            if group["params"].startswith("^(?=.*backbone\\.projector)")
        }
        self.assertEqual(encoder_lrs, {1e-5})
        self.assertEqual(projector_lrs, {5e-5})
        self.assertEqual(config["optimizer"]["lr"], 5e-4)
        default_no_decay_patterns = [
            group["params"]
            for group in config["optimizer"]["params"]
            if "lr" not in group
        ]
        self.assertFalse(any(
            re.findall(pattern, "backbone.encoder.blocks.0.norm1.weight")
            for pattern in default_no_decay_patterns
        ))


if __name__ == "__main__":
    unittest.main()
