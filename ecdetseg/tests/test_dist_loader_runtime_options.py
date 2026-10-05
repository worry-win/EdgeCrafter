import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ecdetseg.engine.data.dataloader import DataLoader
from ecdetseg.engine.misc.dist_utils import warp_loader


class DistributedLoaderRuntimeOptionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        handle = tempfile.NamedTemporaryFile(delete=False)
        handle.close()
        cls._init_file = handle.name
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{cls._init_file}",
            rank=0,
            world_size=1,
        )

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        if os.path.exists(cls._init_file):
            os.unlink(cls._init_file)

    def test_warp_loader_preserves_explicit_prefetch_settings(self):
        loader = DataLoader(
            TensorDataset(torch.arange(8)),
            batch_size=2,
            num_workers=2,
            prefetch_factor=1,
            persistent_workers=False,
            multiprocessing_context="spawn",
        )

        wrapped = warp_loader(loader, shuffle=False)

        self.assertEqual(wrapped.num_workers, 2)
        self.assertEqual(wrapped.prefetch_factor, 1)
        self.assertFalse(wrapped.persistent_workers)
        self.assertEqual(
            wrapped.multiprocessing_context.get_start_method(),
            "spawn",
        )
        batch = next(iter(wrapped))[0]
        self.assertEqual(batch.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
