import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "ecdetseg/configs/ecdet/ecdet_l_dinov2b_obj365_latest_full_breast_363_100e_es20.yml"
SBATCH = ROOT / "slurm/ecdet_l_dinov2b_obj365_latest_full_breast_4gpu_bs32_at_start.sbatch"


class Obj365LatestBreastTransferJobTest(unittest.TestCase):
    def test_job_snapshots_latest_best_when_allocation_starts(self):
        text = SBATCH.read_text()
        self.assertIn("#SBATCH --gres=gpu:nvidia_geforce_rtx_5090:4", text)
        self.assertIn("SOURCE_BEST=", text)
        self.assertIn('source_signature "$SOURCE_BEST"', text)
        self.assertIn('cp --reflink=auto -- "$SOURCE_BEST" "$SNAPSHOT_TMP"', text)
        self.assertIn('LOAD=(--tuning "$INIT_SNAPSHOT")', text)
        self.assertNotIn('LOAD=(--tuning "$SOURCE_BEST")', text)

    def test_config_is_four_class_breast_full_model_finetuning(self):
        text = CONFIG.read_text()
        self.assertIn("ecdet_l_dinov2b_ecl_o365_decoderbody_breast_363_100e_es20.yml", text)
        self.assertIn("num_classes: 4", text)
        self.assertIn("epochs: 100", text)
        self.assertIn("early_stop_patience: 20", text)


if __name__ == "__main__":
    unittest.main()
