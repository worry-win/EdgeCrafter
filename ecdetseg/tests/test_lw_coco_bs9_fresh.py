import os
import subprocess
import unittest
from pathlib import Path

from ecdetseg.engine.core.yaml_utils import load_config


class LWCocoBatchNineFreshTest(unittest.TestCase):
    def test_batch_nine_starts_from_the_configured_initialization(self):
        config = load_config(
            "ecdetseg/configs/ecdet/"
            "ecdet_x_lw_xlarge_coco80_hybrid01_50e_es30_bs9.yml",
            {},
        )
        self.assertEqual(config["train_dataloader"]["total_batch_size"], 72)
        self.assertEqual(config["val_dataloader"]["total_batch_size"], 72)
        self.assertEqual(config["gradient_accumulation_steps"], 1)
        self.assertEqual(config["epochs"], 50)
        self.assertEqual(config["early_stop_patience"], 30)
        self.assertNotIn("resume", config["output_dir"])

        repo_root = Path(__file__).resolve().parents[2]
        launcher = (
            repo_root
            / "slurm"
            / "ecdet_x_lw_coco80_hybrid01_8gpu_bs9_50e.sbatch"
        )
        text = launcher.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --partition=debug,batch", text)
        self.assertNotIn("JOB504", text)
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
            "WORLD_SIZE=8 GLOBAL_BATCH=72 PER_GPU_BATCH=9",
            result.stdout,
        )
        self.assertIn(
            "INITIALIZATION=LW_OBJECTS365_ENCODER_ONLY "
            "EC_PROJECTOR_NECK_DECODER=RANDOM",
            result.stdout,
        )
        self.assertIn(
            "MP_SHARING_STRATEGY=file_descriptor",
            result.stdout,
        )


if __name__ == "__main__":
    unittest.main()
