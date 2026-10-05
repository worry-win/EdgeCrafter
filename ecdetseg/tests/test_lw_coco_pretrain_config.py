import unittest

from ecdetseg.engine.core.yaml_utils import load_config


class LWCocoPretrainConfigTest(unittest.TestCase):
    def test_coco_config_preserves_job403_training_contract(self):
        config = load_config(
            "ecdetseg/configs/ecdet/"
            "ecdet_x_lw_xlarge_coco80_hybrid01_50e_es30_bs10.yml",
            {},
        )

        self.assertEqual(config["num_classes"], 80)
        self.assertTrue(config["remap_mscoco_category"])
        self.assertEqual(config["ECDet"]["backbone"], "LWDetrBackbone")
        self.assertEqual(config["ECDet"]["encoder"], "HybridEncoder")
        self.assertEqual(config["ECDet"]["decoder"], "ECTransformer")
        self.assertEqual(config["LWDetrBackbone"]["out_feature_indexes"], [8, 9])
        self.assertEqual(config["LWDetrBackbone"]["projector_type"], "ec")
        self.assertEqual(config["ECTransformer"]["num_points"], [3, 6, 3])
        self.assertEqual(config["ECTransformer"]["num_layers"], 4)
        self.assertEqual(config["train_dataloader"]["total_batch_size"], 80)
        self.assertEqual(config["val_dataloader"]["total_batch_size"], 80)
        self.assertEqual(config["epochs"], 50)
        self.assertEqual(config["early_stop_patience"], 30)
        self.assertEqual(config["gradient_accumulation_steps"], 1)
        self.assertEqual(
            config["train_dataloader"]["dataset"]["img_folder"],
            "/cobot/Data/public/coco/train2017",
        )
        self.assertEqual(
            config["train_dataloader"]["dataset"]["ann_file"],
            "/cobot/Data/public/coco/annotations/instances_train2017.json",
        )
        self.assertEqual(
            config["val_dataloader"]["dataset"]["img_folder"],
            "/cobot/Data/public/coco/val2017",
        )
        self.assertEqual(
            config["val_dataloader"]["dataset"]["ann_file"],
            "/cobot/Data/public/coco/annotations/instances_val2017.json",
        )

        encoder_lrs = {
            group["lr"]
            for group in config["optimizer"]["params"]
            if group["params"].startswith("^(?=.*backbone\\.encoder)")
        }
        projector_lrs = {
            group["lr"]
            for group in config["optimizer"]["params"]
            if group["params"].startswith("^(?=.*backbone\\.projector)")
        }
        self.assertEqual(encoder_lrs, {1e-5})
        self.assertEqual(projector_lrs, {5e-4})
        self.assertEqual(config["optimizer"]["lr"], 5e-4)

if __name__ == "__main__":
    unittest.main()
