import unittest

from ecdetseg.engine.core.yaml_utils import load_config


class LWObjects365SamplingConfigsTest(unittest.TestCase):
    def test_two_sampling_configs_share_the_requested_training_contract(self):
        root = "ecdetseg/configs/ecdet/"
        expected = {
            "ecdet_x_lw_xlarge_liver_objects365_363_50e.yml": [3, 6, 3],
            "ecdet_x_lw_xlarge_liver_objects365_layerwise363_633_50e.yml": [
                [3, 6, 3],
                [6, 3, 3],
                [6, 3, 3],
                [3, 6, 3],
            ],
        }

        for filename, points in expected.items():
            with self.subTest(filename=filename):
                config = load_config(root + filename, {})
                self.assertEqual(config["ECDet"]["backbone"], "LWDetrBackbone")
                self.assertEqual(config["ECDet"]["encoder"], "HybridEncoder")
                self.assertEqual(config["ECDet"]["decoder"], "ECTransformer")
                self.assertTrue(config["LWDetrBackbone"]["weights_path"].endswith(
                    "LWDETR_xlarge_30e_objects365.pth"
                ))
                self.assertEqual(config["LWDetrBackbone"]["projector_type"], "ec")
                self.assertEqual(config["LWDetrBackbone"]["out_feature_indexes"], [8, 9])
                self.assertEqual(config["ECTransformer"]["num_layers"], 4)
                self.assertEqual(config["ECTransformer"]["num_points"], points)
                self.assertEqual(config["HybridEncoder"]["dim_feedforward"], 2048)
                self.assertEqual(config["ECTransformer"]["dim_feedforward"], 2048)
                self.assertEqual(config["optimizer"]["lr"], 5e-4)
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
                self.assertEqual(config["train_dataloader"]["total_batch_size"], 32)
                self.assertEqual(config["epochs"], 50)
                self.assertTrue(config["train_dataloader"]["dataset"]["ann_file"].endswith(
                    "/Ignore_Delete-Image/train.json"
                ))
                self.assertTrue(config["val_dataloader"]["dataset"]["ann_file"].endswith(
                    "/Ignore_Delete-Image/valid.json"
                ))


if __name__ == "__main__":
    unittest.main()
