from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

import torch.multiprocessing as mp


ECDETSEG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ECDETSEG_ROOT))

import train  # noqa: E402


class TrainMultiprocessingTest(unittest.TestCase):
    def test_main_uses_file_system_sharing_before_distributed_setup(self):
        original_strategy = mp.get_sharing_strategy()
        mp.set_sharing_strategy("file_descriptor")
        observed_strategy = None

        def observe_strategy(*_args, **_kwargs):
            nonlocal observed_strategy
            observed_strategy = mp.get_sharing_strategy()
            raise RuntimeError("stop after observing startup configuration")

        args = SimpleNamespace(print_rank=0, print_method="builtin", seed=42)
        try:
            with mock.patch.object(
                train.dist_utils,
                "setup_distributed",
                side_effect=observe_strategy,
            ):
                with self.assertRaisesRegex(RuntimeError, "stop after observing"):
                    train.main(args)
        finally:
            mp.set_sharing_strategy(original_strategy)

        self.assertEqual(observed_strategy, "file_system")


if __name__ == "__main__":
    unittest.main()
