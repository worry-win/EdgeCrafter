"""Train cmp5L privileged query-behavior distillation pilots.

This is an opt-in entrypoint.  It does not alter the default ECDet training
path, checkpoint schema, model forward, evaluator, or EMA implementation.
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

import torch
import torch.multiprocessing as mp


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "ecdetseg"))
sys.path.insert(0, str(PROJECT_ROOT))

from engine.core import YAMLConfig, yaml_utils  # noqa: E402
from engine.misc import MetricLogger, SmoothedValue, dist_utils  # noqa: E402
from engine.optim import ModelEMA  # noqa: E402
from engine.solver.ec_engine import (  # noqa: E402
    _BatchSlice,
    _optimizer_step_due,
)
from engine.solver.ec_solver import ECSolver  # noqa: E402
import engine.solver.ec_solver as ec_solver_module  # noqa: E402

from scripts.ablation.cmp5L_query_behavior_kd import (  # noqa: E402
    compute_behavior_kd_losses,
    replay_frozen_teacher,
    run_model_with_capture,
)


@dataclass
class KDRuntime:
    teacher_decoder: torch.nn.Module
    mode: str
    lambda_query: float
    lambda_negative: float
    lambda_sampling: float
    max_train_steps: int
    strict_debug_checks: bool
    parity_pending: bool = True
    observed_query_nonzero: bool = False
    observed_negative_nonzero: bool = False
    observed_sampling_nonzero: bool = False
    finite_student_gradient_seen: bool = False


ACTIVE_RUNTIME: KDRuntime | None = None


def _load_frozen_teacher_decoder(student_decoder, checkpoint_path, device):
    teacher = copy.deepcopy(student_decoder).to(device)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    source = state["ema"]["module"] if "ema" in state else state["model"]
    prefix = "decoder."
    teacher_state = {
        key[len(prefix):]: value
        for key, value in source.items()
        if key.startswith(prefix)
    }
    missing, unexpected = teacher.load_state_dict(teacher_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "teacher decoder checkpoint mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher


def _grad_norm(parameters):
    squared = []
    for parameter in parameters:
        if parameter.grad is not None:
            squared.append(parameter.grad.detach().float().square().sum())
    if not squared:
        return 0.0
    return float(torch.stack(squared).sum().sqrt())


def _nonfinite_gradient_names(module):
    return [
        name
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
        and not torch.isfinite(parameter.grad).all()
    ]


def _matches(criterion, predictions, targets):
    result = criterion.matcher(
        {
            "pred_logits": predictions["pred_logits"].detach().float(),
            "pred_boxes": predictions["pred_boxes"].detach().float(),
        },
        targets,
    )
    return result["indices"] if isinstance(result, dict) else result


def _enabled_losses(runtime, raw_losses):
    losses = {}
    if runtime.mode in {"B", "D", "E"}:
        losses["loss_qbeh_query"] = (
            raw_losses["loss_query_update"] * runtime.lambda_query
        )
    if runtime.mode in {"C", "D", "E"}:
        losses["loss_qbeh_negative"] = (
            raw_losses["loss_negative_behavior"] * runtime.lambda_negative
        )
    if runtime.mode == "E":
        losses["loss_qbeh_sampling"] = (
            raw_losses["loss_sampling"] * runtime.lambda_sampling
        )
    return losses


def _print_diagnostics(
    *,
    epoch,
    step,
    detection_loss,
    raw_losses,
    weighted_losses,
    kd_stats,
    replay_diagnostics,
    model,
    runtime,
    include_gradients=False,
    scaler=None,
):
    module = model.module if hasattr(model, "module") else model
    payload = {
        "event": "qbeh_diagnostics",
        "epoch": int(epoch),
        "step": int(step),
        "det_loss": float(detection_loss.detach()),
        "query_kd_raw": float(raw_losses["loss_query_update"].detach()),
        "negative_kd_raw": float(raw_losses["loss_negative_behavior"].detach()),
        "sampling_kd_raw": float(raw_losses["loss_sampling"].detach()),
        "weighted": {
            key: float(value.detach()) for key, value in weighted_losses.items()
        },
        "teacher_better_count": kd_stats["teacher_better_count"],
        "negative_count": kd_stats["negative_count"],
        "shared_init": {
            "query_max_diff": replay_diagnostics[
                "initial_query_max_abs_diff"
            ],
            "reference_max_diff": replay_diagnostics[
                "initial_reference_max_abs_diff"
            ],
            "query_exact": replay_diagnostics["initial_query_exact"],
            "reference_exact": replay_diagnostics["initial_reference_exact"],
        },
        "oracle": {
            "bg_dependency_mean": replay_diagnostics.get(
                "bg_dependency_mean"
            ),
            "high_dependency_fraction": replay_diagnostics.get(
                "high_dependency_fraction"
            ),
        },
        "teacher_training": runtime.teacher_decoder.training,
        "teacher_requires_grad_count": sum(
            int(parameter.requires_grad)
            for parameter in runtime.teacher_decoder.parameters()
        ),
        "teacher_grad_count": sum(
            int(parameter.grad is not None)
            for parameter in runtime.teacher_decoder.parameters()
        ),
        "cuda_max_memory_mb": (
            round(torch.cuda.max_memory_allocated() / 1024**2, 1)
            if torch.cuda.is_available()
            else 0.0
        ),
    }
    if include_gradients:
        nonfinite = _nonfinite_gradient_names(module)
        payload["gradient_norms"] = {
            "decoder": _grad_norm(module.decoder.parameters()),
            "class_head": _grad_norm(module.decoder.dec_score_head.parameters()),
            "backbone": _grad_norm(module.backbone.parameters()),
            "teacher": _grad_norm(runtime.teacher_decoder.parameters()),
        }
        payload["nonfinite_student_gradient_count"] = len(nonfinite)
        payload["nonfinite_student_gradient_examples"] = nonfinite[:8]
        payload["amp_scale"] = float(scaler.get_scale()) if scaler is not None else None
    print(json.dumps(payload, sort_keys=True), flush=True)


def train_one_epoch_qbeh(
    self_lr_scheduler,
    lr_scheduler,
    model,
    criterion,
    data_loader,
    optimizer,
    device,
    epoch,
    max_norm=0,
    **kwargs,
):
    runtime = ACTIVE_RUNTIME
    if runtime is None:
        raise RuntimeError("KD runtime was not initialized")
    model.train()
    module = model.module if hasattr(model, "module") else model
    if getattr(module, "_freeze_backbone", False):
        module.backbone.eval()
    runtime.teacher_decoder.eval()
    criterion.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = kwargs.get("print_freq", 10)
    writer = kwargs.get("writer")
    ema: ModelEMA | None = kwargs.get("ema")
    scaler = kwargs.get("scaler")
    lr_warmup_scheduler = kwargs.get("lr_warmup_scheduler")
    accumulation_steps = max(
        1, int(kwargs.get("gradient_accumulation_steps", 1))
    )
    start_step = max(0, int(kwargs.get("start_step", 0)))
    checkpoint_interval_steps = max(
        0, int(kwargs.get("checkpoint_interval_steps", 0))
    )
    checkpoint_callback = kwargs.get("checkpoint_callback")
    optimizer.zero_grad(set_to_none=True)
    cur_iters = epoch * len(data_loader)
    epoch_loader = _BatchSlice(data_loader, start_step=start_step)
    processed = 0

    for relative_i, (samples, targets) in enumerate(
        metric_logger.log_every(epoch_loader, print_freq, header)
    ):
        if runtime.max_train_steps > 0 and processed >= runtime.max_train_steps:
            break
        processed += 1
        i = start_step + relative_i
        samples = samples.to(device)
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        global_step = epoch * len(data_loader) + i
        metas = {
            "epoch": epoch,
            "step": i,
            "global_step": global_step,
            "epoch_step": len(data_loader),
        }

        autocast_enabled = scaler is not None
        with torch.autocast(
            device_type=str(device),
            enabled=autocast_enabled,
            cache_enabled=True,
        ):
            outputs, student_trace, replay_inputs = run_model_with_capture(
                model, samples, targets
            )
            if runtime.parity_pending:
                normal_teacher, _, _ = replay_frozen_teacher(
                    runtime.teacher_decoder,
                    replay_inputs,
                    targets,
                    privileged=False,
                )
                parity = {
                    "logits": float(
                        (
                            normal_teacher["pred_logits"]
                            - outputs["pred_logits"].detach()
                        )
                        .abs()
                        .max()
                    ),
                    "boxes": float(
                        (
                            normal_teacher["pred_boxes"]
                            - outputs["pred_boxes"].detach()
                        )
                        .abs()
                        .max()
                    ),
                }
                print(json.dumps({"event": "normal_parity", **parity}), flush=True)
                if max(parity.values()) > 2e-6:
                    raise RuntimeError(f"normal teacher parity failed: {parity}")
                runtime.parity_pending = False

            teacher_predictions, teacher_trace, replay_diagnostics = (
                replay_frozen_teacher(
                    runtime.teacher_decoder,
                    replay_inputs,
                    targets,
                    privileged=True,
                )
            )
            if not (
                replay_diagnostics["initial_query_exact"]
                and replay_diagnostics["initial_reference_exact"]
            ):
                raise RuntimeError(
                    f"shared decoder initialization failed: {replay_diagnostics}"
                )

        if torch.isnan(outputs["pred_boxes"]).any() or torch.isinf(
            outputs["pred_boxes"]
        ).any():
            raise FloatingPointError("Student predicted boxes contain NaN/Inf")

        with torch.autocast(device_type=str(device), enabled=False):
            loss_dict = criterion(outputs, targets, **metas)
            detection_loss = sum(loss_dict.values())
            matches = _matches(criterion, outputs, targets)
            raw_losses, kd_stats = compute_behavior_kd_losses(
                student_trace,
                teacher_trace,
                outputs,
                teacher_predictions,
                targets,
                matches,
                distill_layers=(0, 1, 2),
            )
            weighted_losses = _enabled_losses(runtime, raw_losses)
            loss_dict.update(weighted_losses)
            loss = sum(loss_dict.values())

        runtime.observed_query_nonzero |= (
            float(raw_losses["loss_query_update"].detach()) > 0
        )
        runtime.observed_negative_nonzero |= (
            float(raw_losses["loss_negative_behavior"].detach()) > 0
        )
        runtime.observed_sampling_nonzero |= (
            float(raw_losses["loss_sampling"].detach()) > 0
        )

        if scaler is not None:
            scaler.scale(loss / accumulation_steps).backward()
        else:
            (loss / accumulation_steps).backward()

        should_step = _optimizer_step_due(i, len(data_loader), accumulation_steps)
        include_gradients = should_step and not runtime.finite_student_gradient_seen
        if should_step and scaler is not None:
            scaler.unscale_(optimizer)
        nonfinite_gradients = (
            _nonfinite_gradient_names(module) if should_step else []
        )
        if should_step and not nonfinite_gradients:
            student_grad_norm = _grad_norm(module.parameters())
            runtime.finite_student_gradient_seen |= (
                math.isfinite(student_grad_norm) and student_grad_norm > 0
            )
        if include_gradients or global_step % max(1, print_freq) == 0:
            _print_diagnostics(
                epoch=epoch,
                step=i,
                detection_loss=detection_loss,
                raw_losses=raw_losses,
                weighted_losses=weighted_losses,
                kd_stats=kd_stats,
                replay_diagnostics=replay_diagnostics,
                model=model,
                runtime=runtime,
                include_gradients=include_gradients,
                scaler=scaler,
            )

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
        if (
            should_step
            and checkpoint_callback is not None
            and checkpoint_interval_steps > 0
            and completed_steps % checkpoint_interval_steps == 0
        ):
            checkpoint_callback(epoch, completed_steps)

    if runtime.strict_debug_checks:
        required = {
            "B": (runtime.observed_query_nonzero,),
            "C": (runtime.observed_negative_nonzero,),
            "D": (
                runtime.observed_query_nonzero,
                runtime.observed_negative_nonzero,
            ),
            "E": (
                runtime.observed_query_nonzero,
                runtime.observed_negative_nonzero,
                runtime.observed_sampling_nonzero,
            ),
        }[runtime.mode]
        if not all(required):
            raise RuntimeError(
                "enabled KD loss stayed zero during debug: "
                f"q={runtime.observed_query_nonzero}, "
                f"neg={runtime.observed_negative_nonzero}, "
                f"sample={runtime.observed_sampling_nonzero}"
            )
        if any(parameter.grad is not None for parameter in runtime.teacher_decoder.parameters()):
            raise RuntimeError("teacher received gradients")
        if not runtime.finite_student_gradient_seen:
            raise RuntimeError("no finite nonzero Student gradient was observed")

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {
        key: meter.global_avg for key, meter in metric_logger.meters.items()
    }


class QueryBehaviorECSolver(ECSolver):
    def __init__(self, cfg, args):
        super().__init__(cfg)
        self.qbeh_args = args

    def train(self):
        global ACTIVE_RUNTIME
        super().train()
        student_module = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        teacher = _load_frozen_teacher_decoder(
            student_module.decoder,
            self.qbeh_args.teacher_checkpoint,
            self.device,
        )
        ACTIVE_RUNTIME = KDRuntime(
            teacher_decoder=teacher,
            mode=self.qbeh_args.qbeh_mode,
            lambda_query=self.qbeh_args.lambda_query,
            lambda_negative=self.qbeh_args.lambda_negative,
            lambda_sampling=self.qbeh_args.lambda_sampling,
            max_train_steps=self.qbeh_args.max_train_steps,
            strict_debug_checks=self.qbeh_args.strict_debug_checks,
            parity_pending=not bool(self.qbeh_args.resume),
        )
        print(
            json.dumps(
                {
                    "event": "qbeh_runtime",
                    "mode": ACTIVE_RUNTIME.mode,
                    "lambda_query": ACTIVE_RUNTIME.lambda_query,
                    "lambda_negative": ACTIVE_RUNTIME.lambda_negative,
                    "lambda_sampling": ACTIVE_RUNTIME.lambda_sampling,
                    "teacher_training": teacher.training,
                    "teacher_trainable_parameters": sum(
                        parameter.numel()
                        for parameter in teacher.parameters()
                        if parameter.requires_grad
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume")
    parser.add_argument("-t", "--tuning")
    parser.add_argument("-d", "--device")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--summary-dir")
    parser.add_argument("-u", "--update", nargs="+")
    parser.add_argument("--print-method", default="builtin")
    parser.add_argument("--print-rank", type=int, default=0)
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--qbeh-mode", choices=("B", "C", "D", "E"), required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--lambda-query", type=float, default=0.10)
    parser.add_argument("--lambda-negative", type=float, default=0.10)
    parser.add_argument("--lambda-sampling", type=float, default=0.02)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--strict-debug-checks", action="store_true")
    return parser.parse_args()


def main(args):
    mp.set_sharing_strategy(os.environ.get("EC_MP_SHARING_STRATEGY", "file_system"))
    dist_utils.setup_distributed(
        args.print_rank, args.print_method, seed=args.seed
    )
    if args.tuning and args.resume:
        raise ValueError("tuning and resume are mutually exclusive")
    update = yaml_utils.parse_cli(args.update)
    standard_args = {
        key: value
        for key, value in vars(args).items()
        if key
        not in {
            "update",
            "qbeh_mode",
            "teacher_checkpoint",
            "lambda_query",
            "lambda_negative",
            "lambda_sampling",
            "max_train_steps",
            "strict_debug_checks",
        }
        and value is not None
    }
    update.update(standard_args)
    cfg = YAMLConfig(args.config, **update)
    if args.resume or args.tuning:
        for backbone_name in ("ViTAdapter", "DinoV2Adapter"):
            if backbone_name in cfg.yaml_cfg:
                cfg.yaml_cfg[backbone_name]["skip_load_backbone"] = True

    ec_solver_module.train_one_epoch = train_one_epoch_qbeh
    solver = QueryBehaviorECSolver(cfg, args)
    solver.fit()
    dist_utils.cleanup()


if __name__ == "__main__":
    main(parse_args())
