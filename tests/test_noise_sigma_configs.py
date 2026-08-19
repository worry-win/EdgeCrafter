import copy
import unittest
from pathlib import Path

from ecdetseg.engine.core.yaml_utils import load_config


class NoiseSigmaConfigTest(unittest.TestCase):
    def test_intermediate_sigma_configs_change_only_noise_strength(self):
        config_root = Path("ecdetseg/configs/ecdet")
        reference = load_config(
            config_root / "ecdet_l_dinov2s_patch16_dec4_liver_delete_image_scratch_bg_noise.yml",
            {},
        )
        reference_op = next(
            op for op in reference["train_dataloader"]["dataset"]["transforms"]["ops"]
            if op.get("type") == "GTExcludedBackgroundCorruption"
        )

        for suffix, sigma in (("0p05", 0.05), ("0p10", 0.10)):
            with self.subTest(sigma=sigma):
                config = load_config(
                    config_root / f"ecdet_l_dinov2s_patch16_dec4_liver_delete_image_scratch_bg_noise_sigma{suffix}.yml",
                    {},
                )
                corruption = next(
                    op for op in config["train_dataloader"]["dataset"]["transforms"]["ops"]
                    if op.get("type") == "GTExcludedBackgroundCorruption"
                )
                self.assertEqual(corruption["noise_std"], sigma)
                self.assertEqual(
                    {key: value for key, value in corruption.items() if key != "noise_std"},
                    {key: value for key, value in reference_op.items() if key != "noise_std"},
                )
                self.assertTrue(config["DinoV2Adapter"]["skip_load_backbone"])
                self.assertEqual(config["ECTransformer"]["num_layers"], 4)
                self.assertEqual(config["train_dataloader"]["total_batch_size"], 32)

                normalized = copy.deepcopy(config)
                normalized["output_dir"] = reference["output_dir"]
                normalized_corruption = next(
                    op for op in normalized["train_dataloader"]["dataset"]["transforms"]["ops"]
                    if op.get("type") == "GTExcludedBackgroundCorruption"
                )
                normalized_corruption["noise_std"] = reference_op["noise_std"]
                self.assertEqual(normalized, reference)


if __name__ == "__main__":
    unittest.main()
