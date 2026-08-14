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
    def test_csg_no_cdn_changes_only_denoising_and_output_directory(self):
        baseline = _resolved_config(
            "ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m.yml"
        )
        no_cdn = _resolved_config(
            "ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m_no_cdn.yml"
        )

        self.assertEqual(no_cdn["ECTransformer"]["num_denoising"], 0)

        comparable_baseline = deepcopy(baseline)
        comparable_no_cdn = deepcopy(no_cdn)
        comparable_baseline.pop("__include__", None)
        comparable_no_cdn.pop("__include__", None)
        comparable_baseline["ECTransformer"]["num_denoising"] = 0
        comparable_baseline["output_dir"] = comparable_no_cdn["output_dir"]
        self.assertEqual(comparable_no_cdn, comparable_baseline)

    def test_csg_no_go_lsd_keeps_fdr_and_disables_go_lsd(self):
        config = _resolved_config(
            "ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m_no_go_lsd.yml"
        )

        self.assertTrue(config["ECTransformer"]["use_fdr_decode"])
        self.assertTrue(config["ECTransformer"]["use_aux_distribution"])
        self.assertTrue(config["ECTransformer"]["use_lqe"])
        self.assertTrue(config["ECCriterion"]["use_fgl"])
        self.assertFalse(config["ECCriterion"]["use_uni_set"])
        self.assertFalse(config["ECCriterion"]["use_ddf"])

    def test_csg_no_dfine_removes_the_complete_dfine_family(self):
        config = _resolved_config(
            "ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m_no_dfine.yml"
        )

        self.assertFalse(config["ECTransformer"]["use_fdr_decode"])
        self.assertFalse(config["ECTransformer"]["use_aux_distribution"])
        self.assertFalse(config["ECTransformer"]["use_lqe"])
        self.assertFalse(config["ECTransformer"]["use_pre_outputs"])
        self.assertEqual(config["ECTransformer"]["num_denoising"], 100)
        self.assertEqual(config["ECCriterion"]["losses"], ["mal", "boxes"])
        self.assertFalse(config["ECCriterion"]["use_uni_set"])
        self.assertFalse(config["ECCriterion"]["use_fgl"])
        self.assertFalse(config["ECCriterion"]["use_ddf"])

    def test_csg_continuous_go_differs_from_no_dfine_only_by_go(self):
        no_dfine = _resolved_config(
            "ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m_no_dfine.yml"
        )
        continuous_go = _resolved_config(
            "ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m_continuous_go.yml"
        )

        comparable_no_dfine = deepcopy(no_dfine)
        comparable_continuous_go = deepcopy(continuous_go)
        comparable_no_dfine.pop("__include__", None)
        comparable_continuous_go.pop("__include__", None)
        comparable_no_dfine["ECCriterion"]["use_uni_set"] = True
        comparable_no_dfine["output_dir"] = comparable_continuous_go["output_dir"]
        self.assertEqual(comparable_continuous_go, comparable_no_dfine)

    def test_csg_mosaic_focal_replaces_mal_without_disabling_mosaic(self):
        config = _resolved_config(
            "ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m_mosaic_focal.yml"
        )

        criterion = config["ECCriterion"]
        self.assertEqual(criterion["losses"], ["focal", "boxes", "local"])
        self.assertEqual(criterion["weight_dict"]["loss_focal"], 1)
        self.assertEqual(criterion["alpha"], 0.25)
        self.assertEqual(criterion["gamma"], 2.0)
        self.assertEqual(
            config["train_dataloader"]["dataset"]["transforms"]["mosaic_prob"],
            1.0,
        )

    def test_csg_no_mosaic_focal_only_disables_mosaic_from_focal_control(self):
        mosaic_focal = _resolved_config(
            "ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m_mosaic_focal.yml"
        )
        no_mosaic_focal = _resolved_config(
            "ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m_no_mosaic_focal.yml"
        )

        comparable_mosaic_focal = deepcopy(mosaic_focal)
        comparable_no_mosaic_focal = deepcopy(no_mosaic_focal)
        comparable_mosaic_focal.pop("__include__", None)
        comparable_no_mosaic_focal.pop("__include__", None)
        comparable_mosaic_focal["train_dataloader"]["dataset"]["transforms"][
            "mosaic_prob"
        ] = 0.0
        comparable_mosaic_focal["output_dir"] = comparable_no_mosaic_focal[
            "output_dir"
        ]
        self.assertEqual(comparable_no_mosaic_focal, comparable_mosaic_focal)

    def test_csg_no_dfine_no_cdn_only_disables_cdn_from_no_dfine(self):
        no_dfine = _resolved_config(
            "ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m_no_dfine.yml"
        )
        combined = _resolved_config(
            "ecdet_l_dinov2s_patch16_dec3_kidney_v2_260729_csg200m_no_dfine_no_cdn.yml"
        )

        comparable_no_dfine = deepcopy(no_dfine)
        comparable_combined = deepcopy(combined)
        comparable_no_dfine.pop("__include__", None)
        comparable_combined.pop("__include__", None)
        comparable_no_dfine["ECTransformer"]["num_denoising"] = 0
        comparable_no_dfine["output_dir"] = comparable_combined["output_dir"]
        self.assertEqual(comparable_combined, comparable_no_dfine)

    def test_csg_dec5_no_dfine_keeps_depth_and_removes_dfine(self):
        config = _resolved_config(
            "ecdet_l_dinov2s_patch16_dec5_kidney_v2_260729_csg200m_no_dfine.yml"
        )

        self.assertEqual(config["ECTransformer"]["num_layers"], 5)
        self.assertEqual(config["ECTransformer"]["num_denoising"], 100)
        self.assertFalse(config["ECTransformer"]["use_fdr_decode"])
        self.assertFalse(config["ECTransformer"]["use_aux_distribution"])
        self.assertFalse(config["ECTransformer"]["use_lqe"])
        self.assertFalse(config["ECTransformer"]["use_pre_outputs"])
        self.assertEqual(config["ECCriterion"]["losses"], ["mal", "boxes"])
        self.assertFalse(config["ECCriterion"]["use_uni_set"])
        self.assertFalse(config["ECCriterion"]["use_fgl"])
        self.assertFalse(config["ECCriterion"]["use_ddf"])

    def test_csg_ablation_2_launcher_maps_all_eight_experiments(self):
        launcher = REPO_ROOT / "slurm" / "ecdet_csgv2_ablations_2gpu_150e.sbatch"
        expected = {
            0: ("no-cdn", "csg200m_no_cdn.yml"),
            1: ("no-go-lsd", "csg200m_no_go_lsd.yml"),
            2: ("no-dfine", "csg200m_no_dfine.yml"),
            3: ("continuous-go", "csg200m_continuous_go.yml"),
            4: ("mosaic-focal", "csg200m_mosaic_focal.yml"),
            5: ("no-mosaic-focal", "csg200m_no_mosaic_focal.yml"),
            6: ("no-dfine-no-cdn", "csg200m_no_dfine_no_cdn.yml"),
            7: ("dec5-no-dfine", "dec5_kidney_v2_260729_csg200m_no_dfine.yml"),
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
            self.assertIn("Global batch size: 32 (2 GPUs x 16 samples/GPU)", result.stdout)
            self.assertIn("Epochs/patience/seed: 150/30/42", result.stdout)

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
