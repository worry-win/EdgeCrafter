"""Read-only paired evaluation of saved dep-gate Student and ordinary EMA.

This does not modify training, checkpoints, or the baseline evaluator.  Gate-on
uses the saved Student detector and its saved head; gate-off uses the *same*
Student detector.  EMA is available separately and is necessarily ungated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ecdetseg"))

from engine.core import YAMLConfig  # noqa: E402
from engine.misc import dist_utils  # noqa: E402
from engine.solver import TASKS  # noqa: E402
from scripts.ablation import evaluate_cmp5L_qbeh_checkpoint as base_eval  # noqa: E402
from scripts.ablation.train_cmp5L_depgate_e2e import (  # noqa: E402
    DepGateHead,
    HEAD_IN_DIM,
    _run_forward_with_soft_gate,
)
import scripts.ablation.train_cmp5L_depgate_e2e as depgate_train  # noqa: E402


def split_model_head(state, *, require_head):
    prefix = "depgate_head."
    detector = {k: v for k, v in state.items() if not k.startswith(prefix)}
    head = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if require_head and not head:
        raise ValueError("checkpoint has no depgate_head parameters")
    return detector, head


def summarize_gate_logits(logits_by_layer, *, strength):
    def stats(values):
        return {
            "min": float(values.min()),
            "p05": float(torch.quantile(values, 0.05)),
            "p50": float(torch.quantile(values, 0.50)),
            "mean": float(values.mean()),
            "p95": float(torch.quantile(values, 0.95)),
            "max": float(values.max()),
        }

    summary = []
    for index, batches in enumerate(logits_by_layer):
        if not batches:
            raise ValueError(f"no gate observations for decoder layer {index}")
        logits = torch.cat([b.reshape(-1).float().cpu() for b in batches])
        prob = torch.sigmoid(logits)
        lam = 1.0 - strength * prob
        summary.append({
            "layer": index,
            "n": int(logits.numel()),
            "dhat_logit": stats(logits),
            "prob": stats(prob),
            "lambda": stats(lam),
        })
    return summary


class GateOnModel(nn.Module):
    def __init__(self, detector, head):
        super().__init__()
        self.detector = detector
        self.head = head
        self.logits_by_layer = []

    def forward(self, samples):
        outputs, trace, _ = _run_forward_with_soft_gate(
            self.detector.decoder,
            lambda: self.detector(samples),
            self.head,
        )
        if not self.logits_by_layer:
            self.logits_by_layer = [[] for _ in trace["dhat"]]
        for batches, logits in zip(self.logits_by_layer, trace["dhat"]):
            if logits is None:
                raise RuntimeError("gate-on path did not capture a layer's gate output")
            batches.append(logits.detach().float().cpu().reshape(-1))
        return outputs


def build(args):
    cfg = YAMLConfig(args.config, **{
        "val_dataloader": {
            "dataset": {"ann_file": args.ann_file, "img_folder": args.img_folder},
            "num_workers": args.num_workers,
            "total_batch_size": args.batch_size,
            "shuffle": False,
            "drop_last": False,
        },
        "num_classes": 4,
        "remap_mscoco_category": False,
    })
    for name in ("ViTAdapter", "DinoV2Adapter"):
        if name in cfg.yaml_cfg:
            cfg.yaml_cfg[name]["skip_load_backbone"] = True
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver._setup()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    detector_state, head_state = split_model_head(
        state["model"], require_head=args.mode == "gate-on-model"
    )
    detector = dist_utils.de_parallel(solver.model)
    loaded = detector.load_state_dict(detector_state, strict=True)
    if "postprocessor" in state:
        solver.postprocessor.load_state_dict(state["postprocessor"], strict=True)
    if args.mode == "gate-off-ema":
        if solver.ema is None or "ema" not in state:
            raise ValueError("checkpoint has no EMA weights")
        solver.ema.load_state_dict(state["ema"], strict=True)
        model = solver.ema.module
    elif args.mode == "gate-off-model":
        model = detector
    else:
        hidden = head_state["net.0.weight"].shape[0]
        head = DepGateHead(HEAD_IN_DIM, hidden).to(args.device)
        head_loaded = head.load_state_dict(head_state, strict=True)
        model = GateOnModel(detector, head)
        args._gate_model = model
        if head_loaded.missing_keys or head_loaded.unexpected_keys:
            raise RuntimeError("gate head load had missing or unexpected keys")
    print(json.dumps({
        "event": "checkpoint_load_audit",
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "detector_key_count": len(detector_state),
        "head_key_count": len(head_state),
        "detector_missing": loaded.missing_keys,
        "detector_unexpected": loaded.unexpected_keys,
        "ema_head_key_count": sum("depgate_head" in k for k in state.get("ema", {}).get("module", {})),
    }, sort_keys=True), flush=True)
    return cfg, solver, model.to(args.device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--img-folder", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=("gate-on-model", "gate-off-model", "gate-off-ema"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--expected-images", type=int, default=2011)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    args.weights = "ema" if args.mode == "gate-off-ema" else "model"
    if args.smoke_only:
        if args.mode != "gate-on-model":
            raise ValueError("smoke-only requires gate-on-model")
        cfg, _, model = build(args)
        samples, _ = next(iter(cfg.val_dataloader))
        samples = samples.to(args.device)
        with torch.no_grad():
            baseline = model.detector(samples)
            original_strength = depgate_train.STRONG_BG
            try:
                depgate_train.STRONG_BG = 0.0
                identity = model(samples)
            finally:
                depgate_train.STRONG_BG = original_strength
            model.logits_by_layer = []
            gated = model(samples)
        result = {}
        for key in ("pred_logits", "pred_boxes"):
            result[key + "_identity_max_abs"] = float((baseline[key] - identity[key]).abs().max())
            result[key + "_gated_max_abs"] = float((baseline[key] - gated[key]).abs().max())
        if max(result[k] for k in result if k.endswith("identity_max_abs")) > 2e-6:
            raise AssertionError("all-one gate does not reproduce ordinary Student forward")
        if max(result[k] for k in result if k.endswith("gated_max_abs")) <= 0:
            raise AssertionError("saved gate has no effect")
        print(json.dumps({
            "mode": args.mode,
            "smoke": result,
            "gate_telemetry": summarize_gate_logits(
                model.logits_by_layer, strength=original_strength
            ),
        }, sort_keys=True), flush=True)
        return
    base_eval._build = build
    base_eval.evaluate(args)
    output = Path(args.out)
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["meta"]["mode"] = args.mode
    payload["meta"]["gate_head_weights"] = "model" if args.mode == "gate-on-model" else None
    if args.mode == "gate-on-model":
        payload["gate_telemetry"] = summarize_gate_logits(
            args._gate_model.logits_by_layer, strength=depgate_train.STRONG_BG
        )
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
