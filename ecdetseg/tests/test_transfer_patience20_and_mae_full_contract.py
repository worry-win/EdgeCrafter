import unittest
from pathlib import Path

from ecdetseg.engine.core.yaml_utils import load_config


class TransferPatience20AndMaeFullContractTest(unittest.TestCase):
    def test_three_new_transfer_contracts(self):
        cases = [
            (
                "ecdet_x_lw_xlarge_breast_org71_full_363_100e_es20.yml",
                "LWDetrBackbone",
                4,
                20,
                "ecdet_breast_lw_org71_full_4gpu_bs32_100e_es20.sbatch",
            ),
            (
                "ecdet_x_mae_dino_vitb_liver_org71_full_363_100e_es10.yml",
                "MAEDinoViTBackbone",
                7,
                10,
                "ecdet_mae_dino_vitb_liver_org71_full_4gpu_bs32_100e_es10.sbatch",
            ),
            (
                "ecdet_x_mae_dino_vitb_breast_org71_full_363_100e_es10.yml",
                "MAEDinoViTBackbone",
                4,
                10,
                "ecdet_mae_dino_vitb_breast_org71_full_4gpu_bs32_100e_es10.sbatch",
            ),
        ]
        repo_root = Path(__file__).resolve().parents[2]
        checkpoint = (
            repo_root
            / "outputs/pretrain/"
            "ecdet_x_mae_dino_vitb_multi_organ_71_363_recovery_50e_fixed_"
            "bs80_8gpu_seed42/best.pth"
        )
        self.assertTrue(checkpoint.is_file())

        for name, backbone, classes, patience, launcher_name in cases:
            config = load_config(f"ecdetseg/configs/ecdet/{name}", {})
            self.assertEqual(config["num_classes"], classes)
            self.assertEqual(config["epochs"], 100)
            self.assertEqual(config["early_stop_patience"], patience)
            self.assertEqual(config["train_dataloader"]["total_batch_size"], 32)
            self.assertEqual(
                config["train_dataloader"]["dataset"]["transforms"]["stop_epoch"],
                98,
            )
            self.assertIsNone(config[backbone]["weights_path"])
            self.assertEqual(config["ECTransformer"]["num_points"], [3, 6, 3])
            self.assertEqual(config["ECTransformer"]["num_layers"], 4)

            launcher = repo_root / "slurm" / launcher_name
            text = launcher.read_text(encoding="utf-8")
            self.assertIn("#SBATCH --partition=debug,batch", text)
            self.assertIn(
                "#SBATCH --output=" + str(repo_root) + "/outputs/slurm/%j-", text
            )
            self.assertIn("GLOBAL_BATCH=32 PER_GPU_BATCH=8", text)
            self.assertIn("-t", text)


if __name__ == "__main__":
    unittest.main()
