import os
import subprocess
import unittest
from pathlib import Path


class BreastLwTransferLauncherTest(unittest.TestCase):
    def test_two_initialization_variants_keep_the_large_lr_contract(self):
        repo_root = Path(__file__).resolve().parents[2]
        launcher = repo_root / "slurm" / "ecdet_breast_lw_transfer_large_lr_4gpu_bs32.sbatch"
        expected = {
            0: (
                "NAME=lw_org71_full_big_lr",
                "ecdet_x_lw_xlarge_liver_org71_tune_363.yml",
                "INIT=FULL_ORG71_CHECKPOINT",
            ),
            1: (
                "NAME=lw_objects365_encoder_ecx_random_big_lr",
                "ecdet_x_lw_xlarge_liver_objects365_363_50e.yml",
                "INIT=LW_OBJECTS365_ENCODER_ONLY",
            ),
        }

        for task_id, fragments in expected.items():
            env = os.environ.copy()
            env.update(
                {
                    "DRY_RUN": "1",
                    "SLURM_ARRAY_TASK_ID": str(task_id),
                    "SLURM_ARRAY_JOB_ID": "dry-run",
                    "SLURM_JOB_ID": "dry-run",
                    "SLURM_JOB_NODELIST": "local",
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
            for fragment in fragments:
                self.assertIn(fragment, result.stdout)
            self.assertIn("CLASSES=4 WORLD_SIZE=4 GLOBAL_BATCH=32", result.stdout)
            self.assertIn("LR=LW_ENCODER_1e-5_EC_5e-4", result.stdout)
            self.assertIn("POINTS=[3,6,3] DECODER_LAYERS=4 EPOCHS=50", result.stdout)


if __name__ == "__main__":
    unittest.main()
