import unittest
from pathlib import Path

from scripts.analysis.liver_consensus_error_audit import analyze_image


class LiverConsensusErrorAuditTest(unittest.TestCase):
    def test_three_model_high_confidence_false_positive_is_flagged(self):
        predictions = {
            "408_0": [{"label": 2, "score": 0.91, "box": [10, 10, 30, 30]}],
            "442_0": [{"label": 2, "score": 0.85, "box": [11, 10, 31, 30]}],
            "417_0": [{"label": 2, "score": 0.88, "box": [10, 11, 30, 31]}],
            "503": [],
            "408_2": [],
        }
        clusters = analyze_image(
            predictions=predictions,
            ground_truth=[],
            ignore_boxes=[],
            consensus_iou=0.5,
            match_iou=0.5,
            ignore_iof=0.7,
            min_models=3,
            high_mean_score=0.5,
            high_min_score=0.3,
        )
        self.assertEqual(len(clusters), 1)
        cluster = clusters[0]
        self.assertEqual(cluster["support_models"], 3)
        self.assertTrue(cluster["high_confidence"])
        self.assertEqual(cluster["error_type"], "false_positive")
        self.assertTrue(cluster["is_consensus_error"])

    def test_launcher_audits_both_splits_with_three_model_consensus(self):
        repo_root = Path(__file__).resolve().parents[2]
        launcher = repo_root / "slurm" / "liver_five_model_consensus_error_audit.sbatch"
        text = launcher.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:nvidia_geforce_rtx_5090:1", text)
        self.assertIn("#SBATCH --output=" + str(repo_root) + "/outputs/slurm/%j-", text)
        self.assertIn("--split both", text)
        self.assertIn("--min-models 3", text)
        self.assertIn("408_0", text)
        self.assertIn("442_0", text)
        self.assertIn("417_0", text)
        self.assertIn("503", text)
        self.assertIn("408_2", text)


if __name__ == "__main__":
    unittest.main()
