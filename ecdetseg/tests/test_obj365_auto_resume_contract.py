import unittest
from pathlib import Path


class Obj365AutoResumeContractTest(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[2]

    def test_training_jobs_end_terminally_so_failure_dependency_can_fire(self):
        launchers = [
            "ecdet_l_dinov2b_obj365_fresh_sqlite_8gpu_bs20_w4_nomosaic_nomixup.sbatch",
            "ecdet_l_dinov2b_obj365_resume_sqlite_8gpu_bs20_w4_nomosaic_nomixup.sbatch",
            "ecdet_l_dinov2b_obj365_resume_sqlite_8gpu_bs20_w2_nomosaic_nomixup.sbatch",
        ]
        for filename in launchers:
            with self.subTest(filename=filename):
                text = (self.repo_root / "slurm" / filename).read_text(encoding="utf-8")
                self.assertIn("#SBATCH --no-requeue", text)
                self.assertNotIn("#SBATCH --requeue\n", text)

    def test_resume_uses_reduced_workers_and_latest_readable_checkpoint(self):
        launcher = self.repo_root / "slurm" / (
            "ecdet_l_dinov2b_obj365_resume_sqlite_8gpu_bs20_w2_nomosaic_nomixup.sbatch"
        )
        text = launcher.read_text(encoding="utf-8")
        self.assertIn("train_dataloader.num_workers=2", text)
        self.assertIn('if [[ "$STEP_LAST" -nt "$LAST" ]]', text)
        self.assertIn("torch.load(path, map_location=\"cpu\"", text)

    def test_chain_submits_one_failure_only_fallback(self):
        launcher = self.repo_root / "slurm" / "submit_obj365_bs20_w2_resume_chain.sh"
        text = launcher.read_text(encoding="utf-8")
        self.assertIn("CHAIN_LENGTH=${1:-2}", text)
        self.assertIn('--dependency="afternotok:$parent_job"', text)
        self.assertIn("--kill-on-invalid-dep=yes", text)

    def test_workers4_chain_uses_workers4_resume_launcher(self):
        launcher = self.repo_root / "slurm" / "submit_obj365_bs20_w4_resume_chain.sh"
        text = launcher.read_text(encoding="utf-8")
        self.assertIn("CHAIN_LENGTH=${1:-2}", text)
        self.assertIn(
            "ecdet_l_dinov2b_obj365_resume_sqlite_8gpu_bs20_w4_nomosaic_nomixup.sbatch",
            text,
        )
        self.assertIn('--dependency="afternotok:$parent_job"', text)
        self.assertIn("--kill-on-invalid-dep=yes", text)


if __name__ == "__main__":
    unittest.main()
