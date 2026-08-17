import sys

import torch

from ecdetseg.engine.core import YAMLConfig


def main(config_path):
    config = YAMLConfig(
        config_path,
        train_dataloader={"total_batch_size": 1, "num_workers": 0},
    )
    device = torch.device("cuda")
    loader = config.train_dataloader
    loader.set_epoch(0)
    require_ignore = "neutral_ignore" in config_path
    for samples, targets in loader:
        if not require_ignore or any(len(target["ignore_boxes"]) > 0 for target in targets):
            break
    else:
        raise AssertionError("No ignore annotation found in neutral-ignore training data")
    samples = samples.to(device)
    targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
    assert all((target["labels"] < config.criterion.num_classes).all() for target in targets)
    outputs = config.model.to(device)(samples, targets=targets)
    losses = config.criterion.to(device)(outputs, targets, epoch=0, step=0, global_step=0, epoch_step=1)
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    print({
        "config": config_path,
        "num_classes": config.criterion.num_classes,
        "target_count": sum(len(target["labels"]) for target in targets),
        "ignore_count": sum(len(target["ignore_boxes"]) for target in targets),
        "loss_count": len(losses),
        "loss": float(total.detach()),
    })


if __name__ == "__main__":
    main(sys.argv[1])
