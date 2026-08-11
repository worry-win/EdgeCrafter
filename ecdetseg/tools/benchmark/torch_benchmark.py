"""Benchmark EdgeCrafter PyTorch model-forward latency on CUDA."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
import sys

import torch


@dataclass(frozen=True)
class LatencySummary:
    sample_count: int
    mean_ms: float


def summarize_latencies(
    latencies_ms: list[float],
    drop_fastest: int,
    drop_slowest: int,
) -> LatencySummary:
    if drop_fastest < 0 or drop_slowest < 0:
        raise ValueError("drop counts must be non-negative")
    if drop_fastest + drop_slowest >= len(latencies_ms):
        raise ValueError("drop counts must leave at least one sample")

    ordered = sorted(latencies_ms)
    end = len(ordered) - drop_slowest if drop_slowest else len(ordered)
    retained = ordered[drop_fastest:end]
    return LatencySummary(sample_count=len(retained), mean_ms=fmean(retained))


def format_comparison(
    fp32: LatencySummary,
    fp16: LatencySummary,
    batch_size: int,
) -> str:
    fp32_fps = batch_size * 1000.0 / fp32.mean_ms
    fp16_fps = batch_size * 1000.0 / fp16.mean_ms
    return "\n".join(
        [
            f"{'Metric':<24}{'FP32':>14}{'FP16/AMP':>16}",
            f"{'Mean latency (ms)':<24}{fp32.mean_ms:>14.3f}{fp16.mean_ms:>16.3f}",
            f"{'Throughput (images/s)':<24}{fp32_fps:>14.2f}{fp16_fps:>16.2f}",
            f"{'Retained samples':<24}{fp32.sample_count:>14d}{fp16.sample_count:>16d}",
        ]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark model-only FP32 and FP16/AMP inference latency."
    )
    parser.add_argument("--config", required=True, help="Model YAML configuration")
    parser.add_argument("--checkpoint", required=True, help="Training checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--drop-fastest", type=int, default=50)
    parser.add_argument("--drop-slowest", type=int, default=50)
    return parser.parse_args(argv)


def load_model(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, tuple[int, int]]:
    ecdetseg_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ecdetseg_root))
    from engine.core import YAMLConfig

    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    for backbone_key in ("ViTAdapter", "DinoV2Adapter"):
        if backbone_key in cfg.yaml_cfg:
            cfg.yaml_cfg[backbone_key]["skip_load_backbone"] = True

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "ema" in checkpoint and checkpoint["ema"] is not None:
        state = checkpoint["ema"]["module"]
        state_source = "ema.module"
    else:
        state = checkpoint["model"]
        state_source = "model"

    model = cfg.model
    model.load_state_dict(state)
    model = model.deploy().to(device).eval()
    height, width = cfg.yaml_cfg["eval_spatial_size"]
    print(f"Checkpoint state: {state_source}")
    return model, (int(height), int(width))


@torch.inference_mode()
def measure_latency(
    model: torch.nn.Module,
    images: torch.Tensor,
    warmup: int,
    iterations: int,
    use_amp: bool,
    drop_fastest: int,
    drop_slowest: int,
) -> LatencySummary:
    def forward() -> None:
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            model(images)

    for _ in range(warmup):
        forward()
    torch.cuda.synchronize(images.device)

    latencies_ms = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iterations):
        start.record()
        forward()
        end.record()
        end.synchronize()
        latencies_ms.append(start.elapsed_time(end))

    return summarize_latencies(latencies_ms, drop_fastest, drop_slowest)


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    if args.drop_fastest < 0 or args.drop_slowest < 0:
        raise ValueError("drop counts must be non-negative")
    if args.drop_fastest + args.drop_slowest >= args.iterations:
        raise ValueError("drop counts must leave at least one sample")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a CUDA device")
    torch.cuda.set_device(device)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cudnn.benchmark = True

    model, (height, width) = load_model(args.config, args.checkpoint, device)
    images = torch.randn(
        args.batch_size,
        3,
        height,
        width,
        device=device,
        dtype=torch.float32,
    )

    common = {
        "model": model,
        "images": images,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "drop_fastest": args.drop_fastest,
        "drop_slowest": args.drop_slowest,
    }
    fp32 = measure_latency(use_amp=False, **common)
    fp16 = measure_latency(use_amp=True, **common)

    print("\nTiming scope: cfg.model.deploy()(images) only")
    print("Excluded: preprocessing, CPU-to-GPU transfer, and postprocessing")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Input: batch={args.batch_size}, shape={height}x{width}, TF32=disabled")
    print(
        f"Sampling: warmup={args.warmup}, iterations={args.iterations}, "
        f"drop_fastest={args.drop_fastest}, drop_slowest={args.drop_slowest}"
    )
    print("FP16/AMP uses CUDA autocast with FP32 model weights\n")
    print(format_comparison(fp32, fp16, args.batch_size))


if __name__ == "__main__":
    main()
