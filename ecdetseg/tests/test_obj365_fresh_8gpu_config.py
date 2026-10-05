import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ecdetseg.engine.core.yaml_utils import load_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "ecdetseg/configs/ecdet/ecdet_l_dinov2b_obj365_fresh_sqlite_8gpu_bs16_nomosaic_nomixup.yml"
)
LAUNCHER = ROOT / "slurm/ecdet_l_dinov2b_obj365_fresh_sqlite_8gpu_bs16_nomosaic_nomixup.sbatch"


class Obj365FreshEightGpuConfigTest(unittest.TestCase):
    def test_fresh_recipe_preserves_model_and_disables_only_mosaic_mixup(self):
        config = load_config(str(CONFIG), {})
        train = config["train_dataloader"]
        transforms = train["dataset"]["transforms"]
        op_types = [op["type"] for op in transforms["ops"]]

        self.assertEqual(
            config["output_dir"],
            "outputs/pretrain/ecdet_l_dinov2b_obj365_fresh_sqlite_8gpu_bs16_nomosaic_nomixup_seed42",
        )
        self.assertEqual(config["epochs"], 30)
        self.assertEqual(config["early_stop_patience"], 0)
        self.assertTrue(config["freeze_backbone"])
        self.assertEqual(config["gradient_accumulation_steps"], 1)
        self.assertFalse(config["skip_resume_eval"])
        self.assertEqual(config["checkpoint_interval_steps"], 5000)
        self.assertEqual(train["total_batch_size"], 128)
        self.assertEqual(config["val_dataloader"]["total_batch_size"], 128)
        self.assertEqual(train["num_workers"], 2)
        self.assertEqual(train["prefetch_factor"], 1)
        self.assertFalse(train["persistent_workers"])
        self.assertEqual(config["val_dataloader"]["num_workers"], 0)
        self.assertEqual(train["dataset"]["type"], "SqliteCocoDetection")
        self.assertTrue(
            train["dataset"]["sqlite_file"].endswith(
                "zhiyuan_objv2_train_filtered_existing.sqlite"
            )
        )
        self.assertEqual(transforms["mosaic_prob"], 0.0)
        self.assertEqual(train["collate_fn"]["mixup_prob"], 0.0)
        self.assertEqual(transforms["mosaic_epoch"], 12)
        self.assertEqual(transforms["stop_epoch"], 28)
        self.assertIn("RandomPhotometricDistort", op_types)
        self.assertIn("RandomZoomOut", op_types)
        self.assertIn("RandomIoUCrop", op_types)
        self.assertIn("RandomHorizontalFlip", op_types)
        self.assertEqual(config["num_classes"], 366)
        self.assertFalse(config["remap_mscoco_category"])
        self.assertEqual(
            config["DinoV2Adapter"]["weights_path"],
            "/cobot/Code/CODE/eomt/checkpoints/dinov2/vit_base_patch14_reg4_dinov2.pth",
        )
        self.assertFalse(config["DinoV2Adapter"]["skip_load_backbone"])
        self.assertEqual(config["ECTransformer"]["num_layers"], 4)
        self.assertEqual(config["ECTransformer"]["dim_feedforward"], 1024)
        self.assertEqual(config["ECTransformer"]["num_points"], [3, 6, 3])

    def test_launcher_is_fresh_reserved_eight_gpu_job_with_local_sqlite(self):
        text = LAUNCHER.read_text(encoding="utf-8")

        self.assertIn("#SBATCH --partition=batch", text)
        self.assertIn("#SBATCH --nodelist=cu01", text)
        self.assertIn("#SBATCH --reservation=dev1_cu01_1week", text)
        self.assertIn("#SBATCH --gres=gpu:nvidia_geforce_rtx_5090:8", text)
        self.assertIn("#SBATCH --cpus-per-task=96", text)
        self.assertIn("#SBATCH --mem=220G", text)
        self.assertIn("#SBATCH --time=6-00:00:00", text)
        self.assertIn("%j-dino-ec-obj365-fresh-8g-b16", text)
        self.assertIn("--nproc_per_node=8", text)
        self.assertIn("PER_GPU_BATCH=16", text)
        self.assertIn("EFFECTIVE_BATCH=128", text)
        self.assertIn("WORKERS_PER_RANK=2", text)
        self.assertIn("PREFETCH_FACTOR=1", text)
        self.assertIn("TRAIN_MODE=FRESH_NO_RESUME", text)
        self.assertIn("cp -- \"$TRAIN_SQLITE\" \"$LOCAL_SQLITE\"", text)
        self.assertIn("train_dataloader.dataset.sqlite_file=\"$LOCAL_SQLITE\"", text)
        self.assertIn(
            "outputs/pretrain/ecdet_l_dinov2b_obj365_fresh_sqlite_8gpu_bs16_nomosaic_nomixup_seed42",
            text,
        )
        train_section = text.split('echo "JOB_ID=', maxsplit=1)[1].split(
            'BEST="$OUTPUT/best.pth"', maxsplit=1
        )[0]
        self.assertNotIn(' -r ', train_section)


if __name__ == "__main__":
    unittest.main()
