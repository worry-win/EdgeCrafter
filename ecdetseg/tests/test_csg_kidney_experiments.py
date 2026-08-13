from copy import deepcopy
from pathlib import Path
import os
import subprocess
import sys
import unittest


ECDETSEG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ECDETSEG_ROOT.parent
EC_PYTHON = Path("/home/wanrui/miniconda3/envs/ec/bin/python")
sys.path.insert(0, str(ECDETSEG_ROOT))

from engine.core.yaml_utils import load_config  # noqa: E402


CONFIG_ROOT = ECDETSEG_ROOT / "configs" / "ecdet"
CSG_IMAGE_ROOT = (
    "/opt/public/wangzhiwei/Ultrasound_Data/COCO_Output/肾脏/images_CSG_200m"
)
KIDNEY_ANNOTATION_ROOT = (
    "/opt/public/wangzhiwei/Ultrasound_Data/COCO_Output/肾脏/annotations/v2_260729"
)


def _resolved_config(name: str) -> dict:
    return load_config(str(CONFIG_ROOT / name), cfg={})


class CsgKidneyExperimentConfigTest(unittest.TestCase):
    def test_normal_dec4_changes_only_decoder_depth_and_output_directory(self):
        dec3 = _resolved_config(
            "ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729.yml"
        )
        dec4 = _resolved_config(
            "ecdet_l_dinov2s_patch16_dec4_kidney_v2_260729.yml"
        )

        self.assertEqual(dec4["ECTransformer"]["num_layers"], 4)
        self.assertTrue(dec4["output_dir"].endswith("dec4_kidney_v2_260729"))

        comparable_dec3 = deepcopy(dec3)
        comparable_dec4 = deepcopy(dec4)
        comparable_dec3.pop("__include__", None)
        comparable_dec4.pop("__include__", None)
        comparable_dec3["ECTransformer"]["num_layers"] = 4
        comparable_dec3["output_dir"] = comparable_dec4["output_dir"]
        self.assertEqual(comparable_dec4, comparable_dec3)

    def test_dec4_changes_only_decoder_depth_and_output_directory(self):
        dec3 = _resolved_config(
            "ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m.yml"
        )
        dec4 = _resolved_config(
            "ecdet_l_dinov2s_patch16_dec4_kidney_v2_260729_csg200m.yml"
        )

        self.assertEqual(dec4["ECTransformer"]["num_layers"], 4)
        self.assertTrue(
            dec4["output_dir"].endswith("dec4_kidney_v2_260729_csg200m")
        )

        comparable_dec3 = deepcopy(dec3)
        comparable_dec4 = deepcopy(dec4)
        comparable_dec3.pop("__include__", None)
        comparable_dec4.pop("__include__", None)
        comparable_dec3["ECTransformer"]["num_layers"] = 4
        comparable_dec3["output_dir"] = comparable_dec4["output_dir"]
        self.assertEqual(comparable_dec4, comparable_dec3)

    def test_ecdet_x_csg_configs_share_training_contract(self):
        coco = _resolved_config(
            "ecdet_x_kidney_v2_260729_csg200m_coco_decoder_150e.yml"
        )
        organ = _resolved_config(
            "ecdet_x_kidney_v2_260729_csg200m_organ_decoder_150e.yml"
        )

        for config in (coco, organ):
            self.assertEqual(config["num_classes"], 6)
            self.assertFalse(config["remap_mscoco_category"])
            self.assertEqual(config["ViTAdapter"]["name"], "ecvitsplus")
            self.assertEqual(config["ECTransformer"]["num_layers"], 4)
            self.assertEqual(config["epochs"], 150)
            self.assertEqual(config["early_stop_patience"], 30)
            self.assertEqual(config["early_stop_min_delta"], 0.0)
            self.assertEqual(config["train_dataloader"]["total_batch_size"], 32)
            self.assertEqual(config["val_dataloader"]["total_batch_size"], 32)
            self.assertEqual(
                config["train_dataloader"]["dataset"]["img_folder"],
                CSG_IMAGE_ROOT,
            )
            self.assertEqual(
                config["val_dataloader"]["dataset"]["img_folder"],
                CSG_IMAGE_ROOT,
            )
            self.assertEqual(
                config["train_dataloader"]["dataset"]["ann_file"],
                f"{KIDNEY_ANNOTATION_ROOT}/train.json",
            )
            self.assertEqual(
                config["val_dataloader"]["dataset"]["ann_file"],
                f"{KIDNEY_ANNOTATION_ROOT}/val.json",
            )
            self.assertEqual(
                config["train_dataloader"]["dataset"]["transforms"]["stop_epoch"],
                148,
            )

        self.assertFalse(coco.get("feataug_enable", False))
        self.assertTrue(organ["feataug_enable"])
        self.assertEqual(organ["feataug_types"], ["fc"])
        self.assertEqual(organ["feataug_prob"], 1.0)
        self.assertEqual(organ["feataug_crop_min_scale"], 0.6)
        self.assertEqual(organ["feataug_crop_max_scale"], 1.0)
        self.assertEqual(organ["feataug_loss_weight"], 1.0)
        self.assertEqual(organ["feataug_base_weight"], 1.0)
        self.assertTrue(organ["feataug_norm_total"])
        self.assertFalse(organ["use_prototypes"])
        self.assertFalse(organ["has_contrast"])

    def test_three_task_launcher_dry_run_maps_each_experiment(self):
        launcher = REPO_ROOT / "slurm" / "ecdet_csg_kidney_three_4gpu_150e.sbatch"
        expected = {
            0: (
                "dinov2s-dec4",
                "ecdet_l_dinov2s_patch16_dec4_kidney_v2_260729_csg200m.yml",
                "Tuning checkpoint: none",
            ),
            1: (
                "ecdetx-coco-decoder",
                "ecdet_x_kidney_v2_260729_csg200m_coco_decoder_150e.yml",
                "EC-1+2+coco-decoder.pth",
            ),
            2: (
                "ecdetx-organ-decoder-feataug",
                "ecdet_x_kidney_v2_260729_csg200m_organ_decoder_150e.yml",
                "EC-1+2+organ-decoder.pth",
            ),
        }

        for task_id, fragments in expected.items():
            env = os.environ.copy()
            env.update(
                {
                    "DRY_RUN": "1",
                    "SLURM_ARRAY_TASK_ID": str(task_id),
                    "SLURM_ARRAY_JOB_ID": "dry-run",
                    "SLURM_JOB_NODELIST": "local",
                    "SLURM_JOB_PARTITION": "local",
                }
            )
            result = subprocess.run(
                ["bash", str(launcher)],
                cwd=REPO_ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            for fragment in fragments:
                self.assertIn(fragment, result.stdout)

    def test_dec4_two_task_launcher_maps_normal_and_csgv2(self):
        launcher = (
            REPO_ROOT
            / "slurm"
            / "ecdet_dinov2s_dec4_kidney_v2_260729_4gpu_150e.sbatch"
        )
        expected = {
            0: (
                "normal-dec4",
                "ecdet_l_dinov2s_patch16_dec4_kidney_v2_260729.yml",
            ),
            1: (
                "csgv2-dec4",
                "ecdet_l_dinov2s_patch16_dec4_kidney_v2_260729_csg200m.yml",
            ),
        }

        for task_id, fragments in expected.items():
            env = os.environ.copy()
            env.update(
                {
                    "DRY_RUN": "1",
                    "SLURM_ARRAY_TASK_ID": str(task_id),
                    "SLURM_ARRAY_JOB_ID": "dry-run",
                    "SLURM_JOB_NODELIST": "local",
                    "SLURM_JOB_PARTITION": "local",
                }
            )
            result = subprocess.run(
                ["bash", str(launcher)],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for fragment in fragments:
                self.assertIn(fragment, result.stdout)

    def test_checkpoint_preflight_accepts_coco_and_blocks_missing_feataug(self):
        preflight = REPO_ROOT / "scripts" / "ablation" / "check_ecdet_tuning.py"
        checkpoint_root = Path(
            "/opt/public/wangzhiwei/Detect_Model/EC-1+2+decoder"
        )

        coco = subprocess.run(
            [
                str(EC_PYTHON),
                str(preflight),
                "--config",
                str(
                    CONFIG_ROOT
                    / "ecdet_x_kidney_v2_260729_csg200m_coco_decoder_150e.yml"
                ),
                "--checkpoint",
                str(checkpoint_root / "EC-1+2+coco-decoder.pth"),
            ],
            cwd=ECDETSEG_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(coco.returncode, 0, coco.stderr)
        self.assertIn("matched=764", coco.stdout)

        organ = subprocess.run(
            [
                str(EC_PYTHON),
                str(preflight),
                "--config",
                str(
                    CONFIG_ROOT
                    / "ecdet_x_kidney_v2_260729_csg200m_organ_decoder_150e.yml"
                ),
                "--checkpoint",
                str(checkpoint_root / "EC-1+2+organ-decoder.pth"),
                "--require-feataug",
            ],
            cwd=ECDETSEG_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(organ.returncode, 0)
        self.assertIn("FeatAug is required", organ.stderr)


if __name__ == "__main__":
    unittest.main()
