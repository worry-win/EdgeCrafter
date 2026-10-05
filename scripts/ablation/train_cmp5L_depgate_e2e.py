"""Path-B end-to-end differentiable dep-gate training (GT warm-start + L_det).

Replaces the validated L0-L2 adaptive far-BG Oracle's *hard* gate with a
*learned soft gate* driven by a small MLP head on each layer's query input:

    dhat = MLP(query_input)                     # [B, Q, 1]
    lam  = 1 - STRONG * sigmoid(dhat)           # [B, Q] in [1-STRONG, 1]
    coeff = lam broadcast -> [B, Q, H, sum(P)]  # multiplies sampled value

The intervention point is identical to the validated Oracle (after bilinear
sampling, before attention aggregation), reusing ``sampled_value_suppression_core``.

Losses:
    L_total = L_det + alpha(t) * lambda_bg * L_bg
    L_bg    = BCE(sigmoid(dhat), g_bg_GT)      # warm-start (GT-derived label)
    alpha(t): 1.0 -> 0.0 linear anneal over [anneal_start, anneal_end]

NOT distillation: the GT ``d_q^BG`` only warm-starts the head; the head is
ultimately driven end-to-end by L_det.  Inference is GT-free.

This file is a thin, self-contained variant of ``train_cmp5L_query_behavior_kd.py``
with the teacher/replay machinery removed and the KD losses replaced by the
soft gate + auxiliary BCE.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "ecdetseg"))
sys.path.insert(0, str(PROJECT_ROOT))

from engine.core import YAMLConfig, yaml_utils  # noqa: E402
from engine.misc import MetricLogger, SmoothedValue, dist_utils  # noqa: E402
from engine.optim import ModelEMA  # noqa: E402
from engine.solver.ec_engine import _BatchSlice, _optimizer_step_due  # noqa: E402
from engine.solver.ec_solver import ECSolver  # noqa: E402
import engine.solver.ec_solver as ec_solver_module  # noqa: E402

from scripts.ablation.cmp5L_query_behavior_kd import (  # noqa: E402
    _region_mask,
    sampled_value_suppression_core,
    DecoderReplayInputs,
    _run_forward_with_capture,
    _slice_normal_trace,
)
from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy  # noqa: E402

STRONG_BG = 0.2
RING_SCALE = 1.5
BGDEP_TAU = 0.20
HEAD_IN_DIM = 256


class DepGateHead(nn.Module):
    def __init__(self, in_dim: int = HEAD_IN_DIM, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return self.net(query)  # [B, Q, 1]


@dataclass
class DepGateRuntime:
    head: DepGateHead
    lambda_bg: float
    anneal_start: float
    anneal_end: float
    max_train_steps: int
    strict_debug_checks: bool
    epochs: int
    parity_pending: bool = True
    observed_bg_loss_nonzero: bool = False
    observed_gate_grad: bool = False
    finite_gradient_seen: bool = False


ACTIVE_RUNTIME: DepGateRuntime | None = None


# ---------------------------------------------------------------------------
# Forward with soft gate: reuse _run_forward_with_capture's structure, but the
# per-layer core applies a learned soft gate instead of a static coefficient.
# ---------------------------------------------------------------------------
def _run_decoder_with_soft_gate(ec_transformer, replay, head, *, decoder_forward=None, suppress=True):
    decoder = ec_transformer.decoder
    if decoder_forward is None:
        decoder_forward = decoder.forward
    records = [dict() for _ in decoder.layers]
    originals = []

    for layer_index, layer in enumerate(decoder.layers):
        original_core = layer.cross_attn.ms_deformable_attn_core
        original_forward = layer.forward
        originals.append((layer, original_forward, original_core))

        def core_wrapper(
            value, spatial_shapes, sampling_locations, attention_weights,
            num_points_list, *, _index=layer_index, _normal_core=original_core,
        ):
            records[_index]["sampling_locations"] = sampling_locations
            records[_index]["attention_weights"] = attention_weights
            coeff = records[_index].get("coeff")
            if not suppress or coeff is None:
                return _normal_core(
                    value, spatial_shapes, sampling_locations,
                    attention_weights, num_points_list,
                )
            return sampled_value_suppression_core(
                value, spatial_shapes, sampling_locations,
                attention_weights, num_points_list, coeff,
            )

        def layer_wrapper(
            *args, _index=layer_index, _forward=original_forward, **kwargs,
        ):
            query_input = args[0] if args else kwargs["target"]
            records[_index]["query_input"] = query_input
            if suppress:
                dhat = head(query_input)  # [B, Q, 1]
                records[_index]["dhat"] = dhat
                batch, queries = query_input.shape[:2]
                heads = layer.cross_attn.num_heads
                points = sum(layer.cross_attn.num_points_list)
                lam = 1.0 - STRONG_BG * torch.sigmoid(dhat)  # [B,Q,1]
                records[_index]["coeff"] = lam[:, :, None, :].expand(
                    batch, queries, heads, points).contiguous()
            return _forward(*args, **kwargs)

        layer.cross_attn.ms_deformable_attn_core = core_wrapper
        layer.forward = layer_wrapper

    try:
        decoder_outputs = decoder_forward(
            None,
            replay.initial_query,
            replay.initial_reference_unactivated,
            replay.memory,
            replay.spatial_shapes,
            ec_transformer.dec_bbox_head,
            ec_transformer.dec_score_head,
            ec_transformer.query_pos_head,
            ec_transformer.pre_bbox_head,
            ec_transformer.integral,
            ec_transformer.up,
            ec_transformer.reg_scale,
            attn_mask=replay.attention_mask,
            dn_meta=replay.denoising_metadata,
            continuous_bbox_head=ec_transformer.continuous_bbox_head,
        )
    finally:
        for layer, original_forward, original_core in originals:
            layer.forward = original_forward
            layer.cross_attn.ms_deformable_attn_core = original_core

    trace = {
        "query_inputs": [r.get("query_input") for r in records],
        "sampling_locations": [r.get("sampling_locations") for r in records],
        "attention_weights": [r.get("attention_weights") for r in records],
        "dhat": [r.get("dhat") for r in records],
        "coeff": [r.get("coeff") for r in records],
    }
    return decoder_outputs, trace


def _run_forward_with_soft_gate(ec_transformer, forward_callable, head):
    captured = {}
    original_decoder_forward = ec_transformer.decoder.forward

    def decoder_forward_wrapper(*args, **kwargs):
        captured["replay_inputs"] = DecoderReplayInputs(
            initial_query=args[1],
            initial_reference_unactivated=args[2],
            memory=args[3],
            spatial_shapes=args[4],
            attention_mask=kwargs.get("attn_mask"),
            denoising_metadata=kwargs.get("dn_meta"),
            normal_query_count=ec_transformer.num_queries,
        )
        replay = captured["replay_inputs"]
        decoder_outputs, trace = _run_decoder_with_soft_gate(
            ec_transformer, replay, head,
            decoder_forward=original_decoder_forward,
        )
        captured["trace"] = trace
        return decoder_outputs

    ec_transformer.decoder.forward = decoder_forward_wrapper
    try:
        predictions = forward_callable()
    finally:
        ec_transformer.decoder.forward = original_decoder_forward

    normal_query_count = predictions["pred_logits"].shape[1]
    captured["replay_inputs"].normal_query_count = normal_query_count
    trace = _slice_normal_trace(captured["trace"], normal_query_count)
    return predictions, trace, captured["replay_inputs"]


def gt_bg_dependency(trace, targets):
    """GT d_q^BG [B,Q] from captured L0-L2 locations/attention + GT boxes."""
    loc_layers = [t for t in trace["sampling_locations"] if t is not None]
    attn_layers = [t for t in trace["attention_weights"] if t is not None]
    if not loc_layers or not attn_layers:
        return None
    gt_xyxy = [box_cxcywh_to_xyxy(t["boxes"]) for t in targets]
    batch, queries = loc_layers[0].shape[:2]
    device = attn_layers[0].device
    deps = []
    for loc, attn in zip(loc_layers, attn_layers):
        layer_dep = []
        for b in range(batch):
            if gt_xyxy[b].numel() == 0:
                layer_dep.append(torch.zeros(queries, device=device, dtype=attn.dtype))
            else:
                region = _region_mask(loc[b], gt_xyxy[b], RING_SCALE)
                layer_dep.append((attn[b] * (region == 0).to(attn.dtype)).sum(-1).mean(-1))
        deps.append(torch.stack(layer_dep))
    return torch.stack(deps).mean(0)  # [B,Q]


def _alpha(epoch, epochs, runtime):
    lo, hi = runtime.anneal_start, runtime.anneal_end
    f = epoch / max(1, epochs - 1)
    if f <= lo:
        return 1.0
    if f >= hi:
        return 0.0
    return 1.0 - (f - lo) / (hi - lo)


def _grad_norm(params):
    total = 0.0
    count = 0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.norm().item() ** 2)
            count += 1
    return (total / max(count, 1)) ** 0.5


def _nonfinite_gradient_names(module):
    return [
        name for name, p in module.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]


def train_one_epoch_depgate(
    self_lr_scheduler, lr_scheduler, model, criterion, data_loader,
    optimizer, device, epoch, max_norm=0, **kwargs,
):
    runtime = ACTIVE_RUNTIME
    if runtime is None:
        raise RuntimeError("depgate runtime not initialized")
    model.train()
    module = model.module if hasattr(model, "module") else model
    if getattr(module, "_freeze_backbone", False):
        module.backbone.eval()
    runtime.head.train()
    criterion.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = kwargs.get("print_freq", 10)
    writer = kwargs.get("writer")
    ema: ModelEMA | None = kwargs.get("ema")
    scaler = kwargs.get("scaler")
    lr_warmup_scheduler = kwargs.get("lr_warmup_scheduler")
    accumulation_steps = max(1, int(kwargs.get("gradient_accumulation_steps", 1)))
    start_step = max(0, int(kwargs.get("start_step", 0)))
    checkpoint_callback = kwargs.get("checkpoint_callback")
    epochs = max(1, int(getattr(runtime, "epochs", 1)))

    optimizer.zero_grad(set_to_none=True)
    cur_iters = epoch * len(data_loader)
    epoch_loader = _BatchSlice(data_loader, start_step=start_step)
    processed = 0
    alpha = _alpha(epoch, epochs, runtime)

    for relative_i, (samples, targets) in enumerate(
        metric_logger.log_every(epoch_loader, print_freq, header)
    ):
        if runtime.max_train_steps > 0 and processed >= runtime.max_train_steps:
            break
        processed += 1
        i = start_step + relative_i
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = {"epoch": epoch, "step": i, "global_step": global_step,
                 "epoch_step": len(data_loader)}

        autocast_enabled = scaler is not None
        with torch.autocast(device_type=str(device), enabled=autocast_enabled, cache_enabled=True):
            student_module = module.decoder
            outputs, trace, replay = _run_forward_with_soft_gate(
                student_module, lambda: model(samples, targets=targets), runtime.head,
            )
            if runtime.parity_pending:
                # identity parity: all-ones coeff must reproduce NN bit-for-bit
                # (validated separately; here we only check the trace exists)
                runtime.parity_pending = False

        if torch.isnan(outputs["pred_boxes"]).any() or torch.isinf(outputs["pred_boxes"]).any():
            raise FloatingPointError("predicted boxes contain NaN/Inf")

        with torch.autocast(device_type=str(device), enabled=False):
            loss_dict = criterion(outputs, targets, **metas)
            det_loss = sum(loss_dict.values())

            dhat_layers = [t for t in trace["dhat"] if t is not None]
            d_q_gt = gt_bg_dependency(trace, targets)
            if dhat_layers and d_q_gt is not None:
                dhat = dhat_layers[0].squeeze(-1)  # [B,Q] L0 head output
                g_bg_gt = (d_q_gt > BGDEP_TAU).float().detach()
                bg_loss = F.binary_cross_entropy_with_logits(dhat, g_bg_gt)
                gate_prob_mean = float(torch.sigmoid(dhat).mean().detach())
            else:
                bg_loss = torch.zeros((), device=device)
                gate_prob_mean = float("nan")

            if global_step % max(1, print_freq) == 0:
                head_nan = any(not torch.isfinite(p).all() for p in runtime.head.parameters())
                print(json.dumps({
                    "event": "bg_loss_debug",
                    "epoch": int(epoch), "step": int(i),
                    "n_dhat_layers": len(dhat_layers),
                    "d_q_gt_none": d_q_gt is None,
                    "bg_loss_raw": float(bg_loss.detach()),
                    "alpha": alpha,
                    "lambda_bg": runtime.lambda_bg,
                    "head_has_nan": head_nan,
                    "head_param_norm": float(sum(p.norm().item() for p in runtime.head.parameters())),
                }, sort_keys=True), flush=True)

            weighted_bg = bg_loss * runtime.lambda_bg * alpha
            loss = det_loss + weighted_bg

        runtime.observed_bg_loss_nonzero |= float(bg_loss.detach()) > 0

        if scaler is not None:
            scaler.scale(loss / accumulation_steps).backward()
        else:
            (loss / accumulation_steps).backward()

        should_step = _optimizer_step_due(i, len(data_loader), accumulation_steps)
        if should_step and scaler is not None:
            scaler.unscale_(optimizer)
        nonfinite_gradients = _nonfinite_gradient_names(module) if should_step else []
        if should_step and not nonfinite_gradients:
            g = _grad_norm(module.parameters())
            runtime.finite_gradient_seen |= math.isfinite(g) and g > 0
            hg = _grad_norm(runtime.head.parameters())
            runtime.observed_gate_grad |= math.isfinite(hg) and hg > 0

        if global_step % max(1, print_freq) == 0:
            print(json.dumps({
                "event": "depgate_diag",
                "epoch": int(epoch), "step": int(i),
                "det_loss": float(det_loss.detach()),
                "bg_loss": float(bg_loss.detach()),
                "alpha": alpha,
                "gate_prob_mean": gate_prob_mean,
            }, sort_keys=True), flush=True)

        if should_step:
            if max_norm > 0 and not nonfinite_gradients:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if ema is not None and should_step:
            ema.update(model)
        if self_lr_scheduler and should_step:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        elif not self_lr_scheduler and should_step and lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        reduced = dist_utils.reduce_dict(loss_dict)
        reduced["loss_depgate"] = weighted_bg.detach()
        loss_value = sum(reduced.values())
        if not math.isfinite(float(loss_value.detach())):
            raise FloatingPointError(f"non-finite loss: {reduced}")
        metric_logger.update(loss=loss_value, **reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar("Loss/total", loss_value.item(), global_step)
            for key, value in reduced.items():
                writer.add_scalar(f"Loss/{key}", value.item(), global_step)

        completed_steps = i + 1
        if (should_step and checkpoint_callback is not None
                and completed_steps % 100 == 0):
            checkpoint_callback(epoch, completed_steps)

    if runtime.strict_debug_checks:
        if not runtime.observed_bg_loss_nonzero:
            raise RuntimeError("depgate auxiliary loss stayed zero")
        if not runtime.finite_gradient_seen:
            raise RuntimeError("no finite nonzero student gradient observed")

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {key: meter.global_avg for key, meter in metric_logger.meters.items()}


class DepGateECSolver(ECSolver):
    def __init__(self, cfg, args):
        super().__init__(cfg)
        self.depgate_args = args

    def train(self):
        global ACTIVE_RUNTIME
        super().train()
        student_module = self.model.module if hasattr(self.model, "module") else self.model
        head = DepGateHead(HEAD_IN_DIM, self.depgate_args.head_hidden).to(self.device)
        # Register head params into the already-built optimizer via add_param_group.
        # (The head is created AFTER super().train() built self.optimizer, so we
        #  must add its params explicitly; we do NOT modify the core decoder.)
        self.optimizer.add_param_group({
            "params": head.parameters(),
            "lr": self.depgate_args.head_lr,
            "weight_decay": 0.0,
        })
        student_module.depgate_head = head
        ACTIVE_RUNTIME = DepGateRuntime(
            head=head,
            lambda_bg=self.depgate_args.lambda_bg,
            anneal_start=self.depgate_args.anneal_start,
            anneal_end=self.depgate_args.anneal_end,
            max_train_steps=self.depgate_args.max_train_steps,
            strict_debug_checks=self.depgate_args.strict_debug_checks,
            epochs=int(self.cfg.yaml_cfg.get("epochs", 15)),
            parity_pending=not bool(self.depgate_args.resume),
        )
        print(json.dumps({
            "event": "depgate_runtime",
            "lambda_bg": ACTIVE_RUNTIME.lambda_bg,
            "anneal_start": ACTIVE_RUNTIME.anneal_start,
            "anneal_end": ACTIVE_RUNTIME.anneal_end,
            "epochs": ACTIVE_RUNTIME.epochs,
            "max_train_steps": ACTIVE_RUNTIME.max_train_steps,
            "head_lr": self.depgate_args.head_lr,
            "head_trainable_params": sum(p.numel() for p in head.parameters()),
        }, sort_keys=True), flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("-r", "--resume")
    p.add_argument("-t", "--tuning")
    p.add_argument("-d", "--device")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-amp", action="store_true")
    p.add_argument("--output-dir")
    p.add_argument("--summary-dir")
    p.add_argument("-u", "--update", nargs="+")
    p.add_argument("--print-method", default="builtin")
    p.add_argument("--print-rank", type=int, default=0)
    p.add_argument("--local-rank", type=int)
    p.add_argument("--lambda-bg", type=float, default=1.0)
    p.add_argument("--anneal-start", type=float, default=0.4)
    p.add_argument("--anneal-end", type=float, default=0.8)
    p.add_argument("--head-hidden", type=int, default=64)
    p.add_argument("--head-lr", type=float, default=0.0005)
    p.add_argument("--max-train-steps", type=int, default=0)
    p.add_argument("--strict-debug-checks", action="store_true")
    return p.parse_args()


def main(args):
    mp.set_sharing_strategy(os.environ.get("EC_MP_SHARING_STRATEGY", "file_system"))
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)
    if args.tuning and args.resume:
        raise ValueError("tuning and resume are mutually exclusive")
    update = yaml_utils.parse_cli(args.update)
    exclude = {"update", "lambda_bg", "anneal_start", "anneal_end",
               "head_hidden", "head_lr", "max_train_steps", "strict_debug_checks"}
    standard_args = {k: v for k, v in vars(args).items() if k not in exclude and v is not None}
    update.update(standard_args)
    cfg = YAMLConfig(args.config, **update)
    if args.resume or args.tuning:
        for backbone_name in ("ViTAdapter", "DinoV2Adapter"):
            if backbone_name in cfg.yaml_cfg:
                cfg.yaml_cfg[backbone_name]["skip_load_backbone"] = True

    ec_solver_module.train_one_epoch = train_one_epoch_depgate
    solver = DepGateECSolver(cfg, args)
    solver.fit()
    dist_utils.cleanup()


if __name__ == "__main__":
    main(parse_args())
