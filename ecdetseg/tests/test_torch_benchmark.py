from pathlib import Path
import sys
import unittest


ECDETSEG_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ECDETSEG_ROOT / "tools" / "benchmark"
sys.path.insert(0, str(BENCHMARK_ROOT))

from torch_benchmark import (  # noqa: E402
    LatencySummary,
    format_comparison,
    parse_args,
    summarize_latencies,
)


class TorchBenchmarkTest(unittest.TestCase):
    def test_summary_drops_fastest_and_slowest_samples_before_mean(self):
        summary = summarize_latencies(
            [9.0, 1.0, 5.0, 3.0, 7.0],
            drop_fastest=1,
            drop_slowest=1,
        )

        self.assertEqual(summary.sample_count, 3)
        self.assertEqual(summary.mean_ms, 5.0)

    def test_comparison_prints_fp32_and_fp16_as_columns(self):
        output = format_comparison(
            fp32=LatencySummary(sample_count=900, mean_ms=10.0),
            fp16=LatencySummary(sample_count=900, mean_ms=5.0),
            batch_size=1,
        )

        self.assertIn("FP32", output)
        self.assertIn("FP16/AMP", output)
        self.assertRegex(output, r"Mean latency \(ms\).*10\.000.*5\.000")
        self.assertRegex(output, r"Throughput \(images/s\).*100\.00.*200\.00")
        self.assertRegex(output, r"Retained samples.*900.*900")

    def test_summary_rejects_trimming_all_samples(self):
        with self.assertRaisesRegex(ValueError, "must leave at least one sample"):
            summarize_latencies(
                [1.0, 2.0],
                drop_fastest=1,
                drop_slowest=1,
            )

    def test_cli_defaults_match_approved_sampling_policy(self):
        args = parse_args(["--config", "model.yml", "--checkpoint", "best.pth"])

        self.assertEqual(args.warmup, 100)
        self.assertEqual(args.iterations, 1000)
        self.assertEqual(args.drop_fastest, 50)
        self.assertEqual(args.drop_slowest, 50)
        self.assertEqual(args.batch_size, 1)


if __name__ == "__main__":
    unittest.main()
