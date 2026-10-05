import argparse
import os

import torch
import torch.distributed as dist
from torch.utils.data import TensorDataset

from ecdetseg.engine.data.dataloader import DataLoader
from ecdetseg.engine.misc.dist_utils import warp_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", choices=("spawn", "forkserver"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    dist.init_process_group(backend="gloo", init_method="env://")
    rank = dist.get_rank()
    loader = DataLoader(
        TensorDataset(torch.arange(256)),
        batch_size=2,
        num_workers=args.workers,
        prefetch_factor=1,
        persistent_workers=False,
        multiprocessing_context=args.context,
    )
    loader = warp_loader(loader, shuffle=False)
    batch = next(iter(loader))[0]
    if batch.numel() != 2:
        raise RuntimeError(f"rank={rank} unexpected batch shape={tuple(batch.shape)}")
    print(
        f"MULTIRANK_DATALOADER_SMOKE rank={rank} world={dist.get_world_size()} "
        f"context={args.context} workers={args.workers} pid={os.getpid()} result=ok",
        flush=True,
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
