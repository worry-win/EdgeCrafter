#!/usr/bin/env python3
"""Run several real augmented distributed train steps for a transfer init."""

import argparse
import itertools
import sys

import torch
from torch.utils import _pytree

from ecdetseg.engine.core import YAMLConfig
from ecdetseg.engine.misc import dist_utils
from ecdetseg.engine.solver.ec_engine import train_one_epoch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--total-batch-size", type=int, default=32)
    args = parser.parse_args()

    if not dist_utils.setup_distributed(seed=42):
        raise RuntimeError("Distributed initialization failed")

    config = YAMLConfig(args.config)
    config.yaml_cfg["DinoV2Adapter"]["skip_load_backbone"] = True
    config.yaml_cfg["train_dataloader"]["total_batch_size"] = args.total_batch_size
    device = torch.device("cuda")
    model = config.model.to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    incompatible = model.load_state_dict(checkpoint["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(str(incompatible))

    first_nonfinite_module = []

    def record_nonfinite(name):
        def hook(_module, _inputs, output):
            if first_nonfinite_module:
                return
            for value in _pytree.tree_leaves(output):
                if torch.is_tensor(value) and torch.is_floating_point(value) and not torch.isfinite(value).all():
                    first_nonfinite_module.append(name)
                    break
        return hook

    hooks = [
        module.register_forward_hook(record_nonfinite(name))
        for name, module in model.named_modules()
        if name and not any(True for _ in module.children())
    ]

    model = dist_utils.warp_model(
        model,
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
    batches = list(itertools.islice(iter(data_loader), args.steps))
    if len(batches) != args.steps:
        raise RuntimeError(f"Expected {args.steps} batches, got {len(batches)}")

    samples0, targets0 = batches[0]
    samples0 = samples0.to(device)
    targets0 = [{key: value.to(device) for key, value in target.items()} for target in targets0]
    model.train()
    with torch.autocast(device_type=str(device), cache_enabled=True):
        outputs0 = model(samples0, targets=targets0)
    diagnostic = {}
    for key, value in outputs0.items():
        if torch.is_tensor(value):
            diagnostic[key] = {
                "finite": bool(torch.isfinite(value).all()),
                "min": float(value.nan_to_num().min()),
                "max": float(value.nan_to_num().max()),
            }
            if key == "pred_boxes":
                diagnostic[key]["min_width_height"] = float(value[..., 2:].nan_to_num().min())
    sample_tensor = samples0.tensors if hasattr(samples0, "tensors") else samples0
    report = {
        "rank": dist_utils.get_rank(),
        "input_finite": bool(torch.isfinite(sample_tensor).all()),
        "input_min": float(sample_tensor.min()),
        "input_max": float(sample_tensor.max()),
        "first_nonfinite_module": first_nonfinite_module[:1],
        "outputs": diagnostic,
    }
    print(report, file=sys.stderr, flush=True)
    for hook in hooks:
        hook.remove()
    if first_nonfinite_module or not torch.isfinite(outputs0["pred_boxes"]).all() or (outputs0["pred_boxes"][..., 2:] < 0).any():
        raise RuntimeError(f"Invalid boxes during instrumented first real-batch forward: {report}")
    del outputs0, samples0, targets0

    stats = train_one_epoch(
        False,
        None,
        model,
        criterion,
        batches,
        optimizer,
        device,
        epoch=0,
        max_norm=config.clip_max_norm,
        print_freq=1,
        scaler=config.scaler,
    )
    if not all(torch.isfinite(torch.tensor(value)) for value in stats.values()):
        raise RuntimeError(f"Non-finite smoke statistics: {stats}")
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    bad = [
        key for key, value in state.items()
        if key != "decoder.anchors"
        and torch.is_floating_point(value)
        and not torch.isfinite(value).all()
    ]
    if bad:
        raise RuntimeError(f"Non-finite parameters/buffers after smoke: {bad[:10]}")

    if dist_utils.is_main_process():
        print({
            "smoke": "PASS",
            "config": args.config,
            "checkpoint": args.checkpoint,
            "steps": args.steps,
            "world_size": dist_utils.get_world_size(),
            "per_rank_batch": batches[0][0].shape[0],
            "max_cuda_memory_gib": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3),
            "loss": stats["loss"],
        })
    dist_utils.cleanup()


if __name__ == "__main__":
    main()
