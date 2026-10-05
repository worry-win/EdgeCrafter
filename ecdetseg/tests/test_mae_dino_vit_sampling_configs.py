import os
import subprocess
import unittest
from pathlib import Path

from ecdetseg.engine.core.yaml_utils import load_config


class MAEDinoViTTrainingStrategyConfigsTest(unittest.TestCase):
    def test_multi_organ_pretrain_and_direct_liver_share_the_model_contract(self):
        root = "ecdetseg/configs/ecdet/"
        expected = {
            "ecdet_x_mae_dino_vitb_multi_organ_71_363_50e_bs40.yml": {
                "classes": 71,
                "batch": 40,
                "train_ann": "/seg_all_moreOrgan/instances_train.json",
                "val_ann": "/seg_all_moreOrgan/instances_test.json",
            },
            "ecdet_x_mae_dino_vitb_liver_direct_363_50e.yml": {
                "classes": 7,
                "batch": 32,
                "train_ann": "/Ignore_Delete-Image/train.json",
                "val_ann": "/Ignore_Delete-Image/valid.json",
            },
        }

        for filename, contract in expected.items():
            with self.subTest(filename=filename):
                config = load_config(root + filename, {})
                self.assertEqual(config["num_classes"], contract["classes"])
                self.assertEqual(config["ECDet"]["backbone"], "MAEDinoViTBackbone")
                self.assertEqual(config["ECDet"]["encoder"], "HybridEncoder")
                self.assertEqual(config["ECDet"]["decoder"], "ECTransformer")
                self.assertTrue(config["MAEDinoViTBackbone"]["weights_path"].endswith(
                    "mae_dino-checkpoint-500000iter.pth"
                ))
                self.assertEqual(
                    config["MAEDinoViTBackbone"]["out_feature_indexes"],
                    [10, 11],
                )
                self.assertEqual(config["MAEDinoViTBackbone"]["proj_dim"], 256)
                self.assertEqual(config["HybridEncoder"]["in_channels"], [256, 256, 256])
                self.assertEqual(config["ECTransformer"]["num_layers"], 4)
                self.assertEqual(config["ECTransformer"]["num_points"], [3, 6, 3])
                self.assertEqual(
                    config["train_dataloader"]["total_batch_size"],
                    contract["batch"],
                )
                self.assertEqual(config["epochs"], 50)
                self.assertTrue(
                    config["train_dataloader"]["dataset"]["ann_file"].endswith(
                        contract["train_ann"]
                    )
                )
                self.assertTrue(
                    config["val_dataloader"]["dataset"]["ann_file"].endswith(
                        contract["val_ann"]
                    )
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

    def test_launcher_maps_pretraining_and_direct_finetuning(self):
        repo_root = Path(__file__).resolve().parents[2]
        launcher = (
            repo_root
            / "slurm"
            / "ecdet_mae_dino_vitb_pretrain_vs_direct_4gpu.sbatch"
        )
        expected = {
            0: (
                "STRATEGY=multi_organ_pretrain",
                "multi_organ_71_363_50e_bs40.yml",
                "CLASSES=71 WORLD_SIZE=4 GLOBAL_BATCH=40 PER_GPU_BATCH=10",
            ),
            1: (
                "STRATEGY=direct_liver_finetune",
                "liver_direct_363_50e.yml",
                "CLASSES=7 WORLD_SIZE=4 GLOBAL_BATCH=32 PER_GPU_BATCH=8",
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
            self.assertIn("POINTS=[3,6,3]", result.stdout)
            self.assertIn("VIT_DEPTH=12 OUTPUT_BLOCKS=[10,11]", result.stdout)
            self.assertIn("EC_PROJECTOR_NECK_DECODER=RANDOM", result.stdout)


if __name__ == "__main__":
    unittest.main()
