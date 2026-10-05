import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ecdetseg.engine.core.yaml_utils import load_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "ecdetseg/configs/ecdet/ecdet_l_dinov2b_obj365_freeze30_filtered_4gpu_bs16acc2_sqlite_w2pf1_no_mosaic_mixup.yml"
)
LAUNCHER = ROOT / "slurm/ecdet_l_dinov2b_obj365_resume_sqlite_4gpu_bs16acc2_w2pf1.sbatch"


class Obj365SqliteW2Pf1ResumeConfigTest(unittest.TestCase):
    def test_only_loader_pressure_changes_from_existing_sqlite_recipe(self):
        config = load_config(str(CONFIG), {})

        self.assertEqual(config["train_dataloader"]["num_workers"], 2)
        self.assertEqual(config["train_dataloader"]["prefetch_factor"], 1)
        self.assertFalse(config["train_dataloader"]["persistent_workers"])
        self.assertEqual(config["train_dataloader"]["total_batch_size"], 64)
        self.assertEqual(config["gradient_accumulation_steps"], 2)
        self.assertEqual(config["checkpoint_interval_steps"], 5000)
        self.assertTrue(config["skip_resume_eval"])
        self.assertEqual(config["train_dataloader"]["dataset"]["type"], "SqliteCocoDetection")
        self.assertEqual(config["train_dataloader"]["dataset"]["transforms"]["mosaic_prob"], 0.0)
        self.assertEqual(config["train_dataloader"]["collate_fn"]["mixup_prob"], 0.0)

    def test_launcher_uses_isolated_config_and_existing_checkpoint_directory(self):
        text = LAUNCHER.read_text(encoding="utf-8")

        self.assertIn("sqlite_w2pf1_no_mosaic_mixup.yml", text)
        self.assertIn("WORKERS_PER_RANK=2", text)
        self.assertIn("PREFETCH_FACTOR=1", text)
        self.assertIn("--mem=160G", text)
        self.assertIn("--nproc_per_node=4", text)
        self.assertIn("last_step.pth", text)
        self.assertIn(
            "outputs/pretrain/ecdet_l_dinov2b_obj365_freeze30_filtered_4gpu_bs16acc2_seed42",
            text,
        )


if __name__ == "__main__":
    unittest.main()
