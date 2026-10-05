"""One-batch real-checkpoint smoke test for FFN activation×gradient capture."""

import argparse
import json

import torch

from scripts.ablation.dump_cmp5L_internal_behavior import capture_ffn_gradient_importance
from scripts.ablation.probe_cmp5L_decoder_internal_tensors import _build, _normalise_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--img-folder", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", default="ema")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    args.device = torch.device(args.device)
    cfg, _solver, model = _build(args)
    samples, targets = next(iter(cfg.val_dataloader))
    samples = samples.to(args.device)
    normalized = _normalise_targets(targets, samples.shape[-2], samples.shape[-1], args.device)
    before = [parameter.requires_grad for parameter in model.parameters()]
    importance = capture_ffn_gradient_importance(model, samples, [{
        "batch_index": 0,
        "query_id": 0,
        "gt_class": int(normalized[0]["labels"][0]),
        "gt_box_cxcywh": normalized[0]["boxes"][0],
    }])
    after = [parameter.requires_grad for parameter in model.parameters()]
    print(json.dumps({
        "shape": list(importance.shape),
        "finite": bool(torch.isfinite(importance).all()),
        "sum": float(importance.sum()),
        "eval_preserved": not model.training,
        "requires_grad_restored": before == after,
    }, indent=2))


if __name__ == "__main__":
    main()
