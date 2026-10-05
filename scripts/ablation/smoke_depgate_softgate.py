"""Smoke-test the dep-gate soft-gate forward on a tiny batch (no training).

Reuses ``_build`` from probe_cmp5L_decoder_internal_tensors to load the model
correctly.  Verifies:
  1. soft-gated forward runs without error,
  2. identity parity (head -> -inf => lambda=1 => bit-for-bit NN),
  3. suppression has a real effect (head -> +inf => lambda=0.2),
  4. the real head runs and yields gate probs in (0,1).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "ecdetseg"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ablation.probe_cmp5L_decoder_internal_tensors import _build  # noqa: E402
from scripts.ablation.train_cmp5L_depgate_e2e import (  # noqa: E402
    DepGateHead,
    _run_forward_with_soft_gate,
    HEAD_IN_DIM,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ann-file", required=True)
    ap.add_argument("--img-folder", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights", choices=("ema", "model"), default="ema")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    cfg, solver, model = _build(args)
    device = torch.device(args.device)
    head = DepGateHead(HEAD_IN_DIM, 64).to(device)

    loader = cfg.val_dataloader
    samples, targets = next(iter(loader))
    samples = samples[: args.batch_size].to(device)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets[: args.batch_size]]

    with torch.no_grad():
        baseline = model(samples, targets=targets)

        class _IdentHead(torch.nn.Module):
            def forward(self, q):
                return torch.full((q.shape[0], q.shape[1], 1), float("-inf"),
                                  device=q.device, dtype=q.dtype)
        out_ident, _, _ = _run_forward_with_soft_gate(
            model.decoder, lambda: model(samples, targets=targets), _IdentHead(),
        )
        d_logits = (out_ident["pred_logits"] - baseline["pred_logits"]).abs().max().item()
        d_boxes = (out_ident["pred_boxes"] - baseline["pred_boxes"]).abs().max().item()
        print(f"identity parity: max|dlogits|={d_logits:.3e}  max|dboxes|={d_boxes:.3e}", flush=True)
        assert d_logits < 1e-4 and d_boxes < 1e-4, "identity parity FAILED"

        class _SuppressHead(torch.nn.Module):
            def forward(self, q):
                return torch.full((q.shape[0], q.shape[1], 1), float("inf"),
                                  device=q.device, dtype=q.dtype)
        out_sup, _, _ = _run_forward_with_soft_gate(
            model.decoder, lambda: model(samples, targets=targets), _SuppressHead(),
        )
        d_sup = (out_sup["pred_logits"] - baseline["pred_logits"]).abs().max().item()
        print(f"suppression effect: max|dlogits| vs baseline = {d_sup:.4f}", flush=True)
        assert d_sup > 1e-3, "soft gate had no effect"

        out_real, trace_real, _ = _run_forward_with_soft_gate(
            model.decoder, lambda: model(samples, targets=targets), head,
        )
        dhat = trace_real["dhat"][0]
        prob = torch.sigmoid(dhat).mean().item()
        print(f"real head: dhat.shape={tuple(dhat.shape)}  mean(sigmoid(dhat))={prob:.4f}", flush=True)

    print("SMOKE PASSED", flush=True)


if __name__ == "__main__":
    main()
