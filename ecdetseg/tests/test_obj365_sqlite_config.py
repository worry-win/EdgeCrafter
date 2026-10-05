import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ecdetseg.engine.core.yaml_config import YAMLConfig
from ecdetseg.engine.core.yaml_utils import load_config


SQLITE_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "ecdetseg/configs/ecdet/ecdet_l_dinov2b_obj365_freeze30_filtered_4gpu_bs16acc2_sqlite_no_mosaic_mixup.yml"
)


class Obj365SqliteConfigTest(unittest.TestCase):
    def test_training_uses_sqlite_only_for_train_data(self):
        c = load_config(str(SQLITE_CONFIG), {})
        transforms = c["train_dataloader"]["dataset"]["transforms"]
        collate = c["train_dataloader"]["collate_fn"]

        self.assertEqual(c["train_dataloader"]["dataset"]["type"], "SqliteCocoDetection")
        self.assertEqual(
            c["train_dataloader"]["dataset"]["sqlite_file"],
            "/cobot/Code/wanrui/EdgeCrafter/outputs/obj365_validation/zhiyuan_objv2_train_filtered_existing.sqlite",
        )
        self.assertEqual(c["val_dataloader"]["dataset"]["type"], "CocoDetection")
        self.assertEqual(transforms["mosaic_prob"], 0.0)
        self.assertEqual(transforms["mosaic_epoch"], 12)
        self.assertEqual(collate["mixup_prob"], 0.0)
        self.assertEqual(collate["mixup_epoch"], 12)
        self.assertEqual(c["train_dataloader"]["num_workers"], 4)
        self.assertEqual(c["val_dataloader"]["num_workers"], 0)
        self.assertEqual(c["gradient_accumulation_steps"], 2)
        self.assertTrue(c["skip_resume_eval"])
        self.assertEqual(c["checkpoint_interval_steps"], 5000)

    def test_runtime_config_exposes_rolling_checkpoint_interval(self):
        config = YAMLConfig(str(SQLITE_CONFIG))

        self.assertEqual(config.checkpoint_interval_steps, 5000)
        self.assertTrue(config.skip_resume_eval)


if __name__ == "__main__":
    unittest.main()
