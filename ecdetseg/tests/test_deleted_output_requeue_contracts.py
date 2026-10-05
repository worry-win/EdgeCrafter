import os
import subprocess
import unittest
from pathlib import Path

from ecdetseg.engine.core.yaml_utils import load_config


class DeletedOutputRequeueConfigTest(unittest.TestCase):
    def test_resolved_configs_match_the_requested_schedules(self):
        config_root = "ecdetseg/configs/ecdet/"
        cases = {
            "ecdet_x_lw_xlarge_liver_org71_tune_363_100e_es10.yml": {
                "epochs": 100,
                "patience": 10,
                "batch": 32,
                "classes": 7,
                "stop_epoch": 98,
            },
            "ecdet_x_lw_xlarge_liver_org71_tune_363_ecx_official_lr_100e_es10.yml": {
                "epochs": 100,
                "patience": 10,
                "batch": 32,
                "classes": 7,
                "stop_epoch": 98,
            },
            "ecdet_x_lw_xlarge_breast_objects365_363_100e_es10.yml": {
                "epochs": 100,
                "patience": 10,
                "batch": 32,
                "classes": 4,
                "stop_epoch": 98,
            },
            "ecdet_x_mae_dino_vitb_multi_organ_71_363_50e_bs80_fixed.yml": {
                "epochs": 50,
                "patience": 0,
                "batch": 80,
                "classes": 71,
                "stop_epoch": 48,
            },
            "ecdet_x_mae_dino_vitb_liver_direct_363_100e_es10.yml": {
                "epochs": 100,
                "patience": 10,
                "batch": 32,
                "classes": 7,
                "stop_epoch": 98,
            },
        }

        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                config = load_config(config_root + filename, {})
                self.assertEqual(config["epochs"], expected["epochs"])
                self.assertEqual(
                    config["early_stop_patience"], expected["patience"]
                )
                self.assertEqual(
                    config["train_dataloader"]["total_batch_size"],
                    expected["batch"],
                )
                self.assertEqual(config["num_classes"], expected["classes"])
                self.assertEqual(
                    config["train_dataloader"]["dataset"]["transforms"][
                        "stop_epoch"
                    ],
                    expected["stop_epoch"],
                )
                self.assertEqual(config["ECTransformer"]["num_points"], [3, 6, 3])
                self.assertEqual(config["ECTransformer"]["num_layers"], 4)

    def test_launchers_expose_the_requested_resources_and_schedules(self):
        repo_root = Path(__file__).resolve().parents[2]
        cases = [
            (
                "ecdet_liver_lw_org71_363_lr_pair_4gpu_bs32_100e_es10.sbatch",
                "0",
                ["WORLD_SIZE=4", "GLOBAL_BATCH=32", "EPOCHS=100", "PATIENCE=10"],
            ),
            (
                "ecdet_liver_lw_org71_363_lr_pair_4gpu_bs32_100e_es10.sbatch",
                "1",
                ["WORLD_SIZE=4", "GLOBAL_BATCH=32", "EPOCHS=100", "PATIENCE=10"],
            ),
            (
                "ecdet_breast_lw_objects365_4gpu_bs32_100e_es10.sbatch",
                None,
                ["WORLD_SIZE=4", "GLOBAL_BATCH=32", "EPOCHS=100", "PATIENCE=10"],
            ),
            (
                "ecdet_mae_dino_vitb_multi_organ_8gpu_bs80_50e_fixed.sbatch",
                None,
                ["WORLD_SIZE=8", "GLOBAL_BATCH=80", "EPOCHS=50", "PATIENCE=0"],
            ),
            (
                "ecdet_mae_dino_vitb_liver_direct_4gpu_bs32_100e_es10.sbatch",
                None,
                ["WORLD_SIZE=4", "GLOBAL_BATCH=32", "EPOCHS=100", "PATIENCE=10"],
            ),
        ]

        for filename, array_id, markers in cases:
            with self.subTest(filename=filename, array_id=array_id):
                launcher = repo_root / "slurm" / filename
                text = launcher.read_text(encoding="utf-8")
                self.assertIn("#SBATCH --partition=debug,batch", text)
                env = os.environ.copy()
                env.update(
                    {
                        "DRY_RUN": "1",
                        "SLURM_JOB_ID": "dry-run",
                        "SLURM_JOB_PARTITION": "local",
                    }
                )
                if array_id is not None:
                    env["SLURM_ARRAY_TASK_ID"] = array_id
                result = subprocess.run(
                    ["bash", str(launcher)],
                    cwd=repo_root,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                for marker in markers:
                    self.assertIn(marker, result.stdout)


if __name__ == "__main__":
    unittest.main()
