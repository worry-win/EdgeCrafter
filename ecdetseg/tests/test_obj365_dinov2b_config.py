import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ecdetseg.engine.core.yaml_utils import load_config


CONFIG = Path(__file__).resolve().parents[2] / "ecdetseg/configs/ecdet/ecdet_l_dinov2b_obj365_freeze30.yml"
RESUME_NO_MOSAIC_MIXUP_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "ecdetseg/configs/ecdet/ecdet_l_dinov2b_obj365_freeze30_filtered_4gpu_bs16acc2_no_mosaic_mixup.yml"
)


class Obj365DinoV2BaseConfigTest(unittest.TestCase):
    def test_training_contract(self):
        c = load_config(str(CONFIG), {})
        self.assertEqual(c["num_classes"], 366)
        self.assertEqual(c["epochs"], 30)
        self.assertTrue(c["freeze_backbone"])
        self.assertEqual(c["train_dataloader"]["total_batch_size"], 192)
        self.assertEqual(c["train_dataloader"]["dataset"]["transforms"]["mosaic_epoch"], 12)
        self.assertEqual(c["train_dataloader"]["collate_fn"]["mixup_epoch"], 12)
        self.assertEqual(c["checkpoint_freq"], 5)
        self.assertEqual(c["DinoV2Adapter"]["backbone_name"], "vit_base_patch14_reg4_dinov2")
        self.assertEqual(c["ECTransformer"]["num_layers"], 4)
        self.assertEqual(c["ECTransformer"]["dim_feedforward"], 1024)

    def test_resume_config_disables_mosaic_and_mixup(self):
        c = load_config(str(RESUME_NO_MOSAIC_MIXUP_CONFIG), {})
        transforms = c["train_dataloader"]["dataset"]["transforms"]
        collate = c["train_dataloader"]["collate_fn"]

        self.assertEqual(transforms["mosaic_prob"], 0.0)
        self.assertEqual(transforms["mosaic_epoch"], 12)
        self.assertEqual(collate["mixup_prob"], 0.0)
        self.assertEqual(collate["mixup_epoch"], 12)
        self.assertEqual(c["train_dataloader"]["total_batch_size"], 64)
        self.assertEqual(c["train_dataloader"]["num_workers"], 0)
        self.assertEqual(c["val_dataloader"]["num_workers"], 0)
        self.assertEqual(c["gradient_accumulation_steps"], 2)
        self.assertTrue(c["skip_resume_eval"])
        self.assertEqual(c["checkpoint_interval_steps"], 5000)


if __name__ == "__main__":
    unittest.main()
