import argparse

import torch

from ecdetseg.engine.core import YAMLConfig
from ecdetseg.engine.misc import dist_utils
from ecdetseg.engine.solver.ec_engine import train_one_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    if not dist_utils.setup_distributed(seed=42):
        raise RuntimeError("Two-process distributed initialization failed")

    config = YAMLConfig(args.config)
    device = torch.device("cuda")
    model = dist_utils.warp_model(
        config.model.to(device),
        sync_bn=config.sync_bn,
        find_unused_parameters=config.find_unused_parameters,
    )
    criterion = config.criterion.to(device)
    optimizer = config.optimizer
    data_loader = dist_utils.warp_loader(
        config.train_dataloader,
        shuffle=config.train_dataloader.shuffle,
    )
    data_loader.set_epoch(0)
    data_loader.sampler.set_epoch(0)
    batch = next(iter(data_loader))

    stats = train_one_epoch(
        False,
        None,
        model,
        criterion,
        [batch],
        optimizer,
        device,
        epoch=0,
        max_norm=config.clip_max_norm,
        print_freq=1,
        scaler=config.scaler,
    )
    if not all(torch.isfinite(torch.tensor(value)) for value in stats.values()):
        raise RuntimeError(f"Non-finite smoke statistics: {stats}")

    max_memory_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    if dist_utils.is_main_process():
        print({
            "smoke": "PASS",
            "config": args.config,
            "world_size": dist_utils.get_world_size(),
            "per_rank_batch": batch[0].shape[0],
            "max_cuda_memory_gib": round(max_memory_gib, 3),
            "loss": stats["loss"],
        })
    dist_utils.cleanup()


if __name__ == "__main__":
    main()
