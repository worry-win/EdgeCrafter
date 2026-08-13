"""Exercise the same NCCL and COCO object collectives used by EC training."""

import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


ECDETSEG_ROOT = Path(__file__).resolve().parents[1] / "ecdetseg"
sys.path.insert(0, str(ECDETSEG_ROOT))

from engine.data.dataset.coco_eval import all_gather as coco_all_gather  # noqa: E402


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", local_rank)

    torch.cuda.set_device(device)
    dist.init_process_group(init_method="env://")

    model = torch.nn.Sequential(
        torch.nn.Linear(2048, 2048),
        torch.nn.GELU(),
        torch.nn.Linear(2048, 2048),
    ).to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank])

    for step in range(32):
        batch = torch.randn(16, 2048, device=device)
        loss = model(batch).square().mean()
        loss.backward()
        for parameter in model.parameters():
            parameter.grad = None

        payload = {
            "rank": rank,
            "step": step,
            "img_ids": list(range(rank * 1000, rank * 1000 + 17 + step + rank)),
            "bytes": bytes(65536 + rank * 49152 + step),
        }
        gathered = coco_all_gather(payload)
        if len(gathered) != world_size:
            raise RuntimeError(
                f"all_gather returned {len(gathered)} entries for world_size={world_size}"
            )
        for gathered_rank, item in enumerate(gathered):
            if item["rank"] != gathered_rank or item["step"] != step:
                raise RuntimeError(
                    f"corrupt object at step={step}: expected rank={gathered_rank}, got={item}"
                )

        if rank == 0 and step % 8 == 0:
            print(f"collective smoke ok: step={step}", flush=True)

    dist.barrier()
    if rank == 0:
        print("EC 2-GPU NCCL + COCO collective verification ok", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
