import os
import subprocess
import unittest
from pathlib import Path

from ecdetseg.engine.core.yaml_utils import load_config


class BreastOrg71Full100EpochTest(unittest.TestCase):
    def test_full_transfer_100_epoch_contract(self):
        config = load_config(
            "ecdetseg/configs/ecdet/"
            "ecdet_x_lw_xlarge_breast_org71_full_363_100e_es10.yml",
            {},
        )
        self.assertEqual(config["num_classes"], 4)
        self.assertIsNone(config["LWDetrBackbone"]["weights_path"])
        self.assertEqual(config["train_dataloader"]["total_batch_size"], 32)
        self.assertEqual(config["val_dataloader"]["total_batch_size"], 32)
        self.assertEqual(config["epochs"], 100)
        self.assertEqual(config["early_stop_patience"], 10)
        self.assertEqual(
            config["train_dataloader"]["dataset"]["transforms"]["stop_epoch"],
            98,
        )
        self.assertEqual(config["ECTransformer"]["num_points"], [3, 6, 3])
        self.assertEqual(config["ECTransformer"]["num_layers"], 4)
        self.assertEqual(
            config["train_dataloader"]["dataset"]["ann_file"],
            "/cobot/Data/Lesion_det/det_breast/annotations/Lesion/"
            "Ignore_Delete-Image/train.json",
        )

        repo_root = Path(__file__).resolve().parents[2]
        launcher = (
            repo_root
            / "slurm"
            / "ecdet_breast_lw_org71_full_4gpu_bs32_100e_es10.sbatch"
        )
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
        result = subprocess.run(
            ["bash", str(launcher)],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "WORLD_SIZE=4 GLOBAL_BATCH=32 PER_GPU_BATCH=8 EPOCHS=100 PATIENCE=10",
            result.stdout,
        )
        self.assertIn("INIT=FULL_ORG71_CHECKPOINT:", result.stdout)
        self.assertIn("MP_SHARING_STRATEGY=file_descriptor", result.stdout)


if __name__ == "__main__":
    unittest.main()
