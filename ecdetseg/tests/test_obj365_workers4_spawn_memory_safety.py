import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ecdetseg.engine.core.yaml_utils import load_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "ecdetseg/configs/ecdet/ecdet_l_dinov2b_obj365_resume_8gpu_bs20_w4_forkserver.yml"
LAUNCHER = ROOT / "slurm/ecdet_l_dinov2b_obj365_resume_8gpu_bs20_w4_forkserver_memguard.sbatch"


class Obj365Workers4SpawnMemorySafetyTest(unittest.TestCase):
    def test_workers4_uses_forkserver_without_extra_prefetch(self):
        config = load_config(str(CONFIG), {})
        train = config["train_dataloader"]

        self.assertEqual(train["num_workers"], 4)
        self.assertEqual(train["prefetch_factor"], 1)
        self.assertFalse(train["persistent_workers"])
        self.assertEqual(train["multiprocessing_context"], "forkserver")
        self.assertEqual(train["total_batch_size"], 160)

    def test_launcher_limits_allocator_threads_and_protects_node_memory(self):
        text = LAUNCHER.read_text(encoding="utf-8")

        self.assertIn("#SBATCH --mem=180G", text)
        self.assertIn("#SBATCH --gres=gpu:nvidia_geforce_rtx_5090:8", text)
        self.assertIn("ecdet_l_dinov2b_obj365_resume_8gpu_bs20_w4_forkserver.yml", text)
        self.assertIn("OMP_NUM_THREADS=1", text)
        self.assertIn("MKL_NUM_THREADS=1", text)
        self.assertIn("OPENBLAS_NUM_THREADS=1", text)
        self.assertIn("NUMEXPR_NUM_THREADS=1", text)
        self.assertIn("MALLOC_ARENA_MAX=2", text)
        self.assertIn("MEMORY_FLOOR_KB=50331648", text)
        self.assertIn('"$MEMORY_WATCHDOG" "$SLURM_JOB_ID" "$SCANCEL" &', text)
        self.assertIn('"multiprocessing_context": c["train_dataloader"].get("multiprocessing_context") == "forkserver"', text)


if __name__ == "__main__":
    unittest.main()
