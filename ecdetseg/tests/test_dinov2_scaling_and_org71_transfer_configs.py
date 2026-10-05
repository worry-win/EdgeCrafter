import importlib.util
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
TRANSFER_SCRIPT = ROOT / "scripts" / "ablation" / "create_full_detector_transfer_init.py"


def load_transfer_module():
    spec = importlib.util.spec_from_file_location("full_transfer", TRANSFER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FullDetectorTransferTest(unittest.TestCase):
    def test_reuses_all_compatible_detector_parameters_but_not_class_outputs(self):
        transfer = load_transfer_module()
        target = {
            "backbone.backbone.block.weight": torch.full((2, 2), 1.0),
            "backbone.projector.0.weight": torch.full((2, 2), 2.0),
            "encoder.fpn.weight": torch.full((2, 2), 3.0),
            "decoder.decoder.layers.0.linear1.weight": torch.full((2, 2), 4.0),
            "decoder.enc_bbox_head.layers.2.weight": torch.full((4, 2), 5.0),
            "decoder.enc_score_head.weight": torch.full((4, 2), 6.0),
            "decoder.dec_score_head.0.bias": torch.full((4,), 7.0),
            "decoder.denoising_class_embed.weight": torch.full((5, 2), 8.0),
        }
        source = {
            "backbone.backbone.block.weight": torch.full((2, 2), 11.0),
            "backbone.projector.0.weight": torch.full((2, 2), 12.0),
            "encoder.fpn.weight": torch.full((2, 2), 13.0),
            "decoder.decoder.layers.0.linear1.weight": torch.full((2, 2), 14.0),
            "decoder.enc_bbox_head.layers.2.weight": torch.full((4, 2), 15.0),
            "decoder.enc_score_head.weight": torch.full((71, 2), 16.0),
            "decoder.dec_score_head.0.bias": torch.full((71,), 17.0),
            "decoder.denoising_class_embed.weight": torch.full((72, 2), 18.0),
        }

        merged, audit = transfer.merge_full_detector_state(target, source)

        for key in (
            "backbone.backbone.block.weight",
            "backbone.projector.0.weight",
            "encoder.fpn.weight",
            "decoder.decoder.layers.0.linear1.weight",
        ):
            self.assertTrue(torch.equal(merged[key], source[key]))
        for key in (
            "decoder.enc_bbox_head.layers.2.weight",
            "decoder.enc_score_head.weight",
            "decoder.dec_score_head.0.bias",
            "decoder.denoising_class_embed.weight",
        ):
            self.assertTrue(torch.equal(merged[key], target[key]))
        self.assertEqual(set(audit["preserved_class_specific"]), {
            "decoder.enc_score_head.weight",
            "decoder.dec_score_head.0.bias",
            "decoder.denoising_class_embed.weight",
        })
        self.assertEqual(audit["preserved_output_initializers"], [
            "decoder.enc_bbox_head.layers.2.weight",
        ])

    def test_five_configs_share_the_intended_data_and_optimization_contract(self):
        from ecdetseg.engine.core.yaml_utils import load_config

        config_dir = ROOT / "ecdetseg" / "configs" / "ecdet"
        organ_configs = {
            "liver": config_dir / "ecdet_x_dinov2b_org71_liver_finetune_363_100e_es10.yml",
            "breast": config_dir / "ecdet_x_dinov2b_org71_breast_finetune_363_100e_es20.yml",
        }
        scaling_configs = {
            "s": config_dir / "ecdet_l_dinov2s_ecl_o365_decoderbody_breast_363_100e_es20.yml",
            "b": config_dir / "ecdet_l_dinov2b_ecl_o365_decoderbody_breast_363_100e_es20.yml",
            "l": config_dir / "ecdet_l_dinov2l_ecl_o365_decoderbody_breast_363_100e_es20.yml",
        }

        for organ, path in organ_configs.items():
            cfg = load_config(str(path), {})
            self.assertEqual(cfg["num_classes"], 7 if organ == "liver" else 4)
            self.assertEqual(cfg["epochs"], 100)
            self.assertEqual(cfg["early_stop_patience"], 10 if organ == "liver" else 20)
            self.assertEqual(cfg["train_dataloader"]["total_batch_size"], 32)
            self.assertEqual(cfg["train_dataloader"]["dataset"]["transforms"]["stop_epoch"], 98)
            self.assertIn(f"det_{organ}", cfg["train_dataloader"]["dataset"]["img_folder"])
            self.assertEqual(cfg["HybridEncoder"]["dim_feedforward"], 2048)
            self.assertEqual(cfg["ECTransformer"]["dim_feedforward"], 2048)

        expected_backbones = {
            "s": ("vit_small_patch14_reg4_dinov2", [10, 11]),
            "b": ("vit_base_patch14_reg4_dinov2", [10, 11]),
            "l": ("vit_large_patch14_reg4_dinov2", [22, 23]),
        }
        for tag, path in scaling_configs.items():
            cfg = load_config(str(path), {})
            backbone_name, indexes = expected_backbones[tag]
            self.assertEqual(cfg["num_classes"], 4)
            self.assertEqual(cfg["epochs"], 100)
            self.assertEqual(cfg["early_stop_patience"], 20)
            self.assertEqual(cfg["train_dataloader"]["total_batch_size"], 32)
            self.assertEqual(cfg["DinoV2Adapter"]["backbone_name"], backbone_name)
            self.assertEqual(cfg["DinoV2Adapter"]["interaction_indexes"], indexes)
            self.assertEqual(cfg["HybridEncoder"]["dim_feedforward"], 1024)
            self.assertEqual(cfg["ECTransformer"]["dim_feedforward"], 1024)
            self.assertEqual(cfg["ECTransformer"]["num_points"], [3, 6, 3])
            self.assertIn("det_breast", cfg["train_dataloader"]["dataset"]["img_folder"])

    def test_org71_retrains_use_patience_30_without_changing_the_training_contract(self):
        from ecdetseg.engine.core.yaml_utils import load_config

        config_dir = ROOT / "ecdetseg" / "configs" / "ecdet"
        configs = {
            "liver": config_dir / "ecdet_x_dinov2b_org71_liver_finetune_363_100e_es30.yml",
            "breast": config_dir / "ecdet_x_dinov2b_org71_breast_finetune_363_100e_es30.yml",
        }
        for organ, path in configs.items():
            cfg = load_config(str(path), {})
            self.assertEqual(cfg["early_stop_patience"], 30)
            self.assertEqual(cfg["epochs"], 100)
            self.assertEqual(cfg["train_dataloader"]["total_batch_size"], 32)
            self.assertEqual(cfg["num_classes"], 7 if organ == "liver" else 4)
            self.assertIn(f"det_{organ}", cfg["train_dataloader"]["dataset"]["img_folder"])
            self.assertIn("es30", cfg["output_dir"])

    def test_o365_dinov2s_gaussian_runs_match_860_and_908_except_for_training_noise(self):
        from ecdetseg.engine.core.yaml_utils import load_config

        config_dir = ROOT / "ecdetseg" / "configs" / "ecdet"
        configs = {
            "liver": load_config(str(config_dir / "ecdet_l_dinov2s_ecl_o365_liver_bg_noise0p15_363_100e_es10.yml"), {}),
            "breast": load_config(str(config_dir / "ecdet_l_dinov2s_ecl_o365_breast_bg_noise0p15_363_100e_es20.yml"), {}),
        }
        for organ, cfg in configs.items():
            self.assertEqual(cfg["num_classes"], 7 if organ == "liver" else 4)
            self.assertEqual(cfg["epochs"], 100)
            self.assertEqual(cfg["early_stop_patience"], 10 if organ == "liver" else 20)
            self.assertEqual(cfg["train_dataloader"]["total_batch_size"], 32)
            self.assertEqual(cfg["DinoV2Adapter"]["backbone_name"], "vit_small_patch14_reg4_dinov2")
            self.assertIn(f"det_{organ}", cfg["train_dataloader"]["dataset"]["img_folder"])
            self.assertIn(f"det_{organ}", cfg["val_dataloader"]["dataset"]["img_folder"])
            train_ops = cfg["train_dataloader"]["dataset"]["transforms"]["ops"]
            noise = [op for op in train_ops if op["type"] == "GTExcludedBackgroundCorruption"]
            self.assertEqual(len(noise), 1)
            self.assertEqual(noise[0]["mode"], "noise")
            self.assertEqual(float(noise[0]["noise_std"]), 0.15)
            self.assertEqual(float(noise[0]["p"]), 0.5)
            val_ops = cfg["val_dataloader"]["dataset"]["transforms"]["ops"]
            self.assertNotIn("GTExcludedBackgroundCorruption", [op["type"] for op in val_ops])

    def test_dinov2s_backbone_only_breast_is_a_strict_908_control(self):
        from ecdetseg.engine.core.yaml_utils import load_config

        config_dir = ROOT / "ecdetseg" / "configs" / "ecdet"
        control = load_config(
            str(config_dir / "ecdet_l_dinov2s_backbone_only_breast_363_100e_es20.yml"),
            {},
        )
        reference = load_config(
            str(config_dir / "ecdet_l_dinov2s_ecl_o365_decoderbody_breast_363_100e_es20.yml"),
            {},
        )

        for key in ("num_classes", "epochs", "early_stop_patience"):
            self.assertEqual(control[key], reference[key])
        for section in ("train_dataloader", "val_dataloader", "HybridEncoder", "ECTransformer", "optimizer"):
            self.assertEqual(control[section], reference[section])
        self.assertEqual(control["num_classes"], 4)
        self.assertEqual(control["epochs"], 100)
        self.assertEqual(control["early_stop_patience"], 20)
        self.assertEqual(control["train_dataloader"]["total_batch_size"], 32)
        self.assertIn("Ignore_Delete-Image/train.json", control["train_dataloader"]["dataset"]["ann_file"])
        self.assertIn("Ignore_Delete-Image/valid.json", control["val_dataloader"]["dataset"]["ann_file"])
        self.assertEqual(control["DinoV2Adapter"]["backbone_name"], "vit_small_patch14_reg4_dinov2")
        self.assertFalse(control["DinoV2Adapter"]["skip_load_backbone"])
        self.assertEqual(control["ECTransformer"]["num_layers"], 4)
        self.assertEqual(control["ECTransformer"]["num_points"], [3, 6, 3])


if __name__ == "__main__":
    unittest.main()
