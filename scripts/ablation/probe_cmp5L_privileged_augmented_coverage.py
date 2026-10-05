"""No-update target coverage probe on the real 15e train augmentation schedule."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from engine.core import YAMLConfig
from engine.misc import dist_utils
from engine.solver import TASKS
from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy
from scripts.ablation.cmp5L_privileged_decision_kd import max_named_buffer_change, select_outcome_targets
from scripts.ablation.evaluate_cmp5L_af_validation import checkpoint_key_audit, normal_bg_dependency, region_mask_with_outside
from scripts.ablation.probe_cmp5L_decoder_internal_tensors import DecoderInternalCapture
from scripts.ablation.probe_cmp5L_privileged_decision_targets import _run_branch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(args):
    cfg = YAMLConfig(args.config, **{
        "train_dataloader": {
            "num_workers": args.num_workers,
            "total_batch_size": args.batch_size,
            "shuffle": True,
            "drop_last": True,
        },
        "val_dataloader": {"num_workers": 0, "total_batch_size": 1},
        "num_classes": 4,
        "remap_mscoco_category": False,
    })
    for name in ("ViTAdapter", "DinoV2Adapter"):
        if name in cfg.yaml_cfg:
            cfg.yaml_cfg[name]["skip_load_backbone"] = True
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver._setup()
    solver.load_resume_state(args.checkpoint)
    model = solver.ema.module if solver.ema is not None else solver.model
    model = dist_utils.de_parallel(model).to(args.device).eval().requires_grad_(False)
    return cfg, solver, model


def evaluate(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    args.device = device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg, solver, model = build(args)
    load = checkpoint_key_audit(model, args.checkpoint, "ema")
    if load["missing_keys"] or load["unexpected_keys"]:
        raise RuntimeError(f"strict checkpoint mismatch: {load}")
    before = {name: value.detach().clone() for name, value in model.named_buffers()}
    parameter_versions = {name: int(value._version) for name, value in model.named_parameters()}
    audit = {
        "model_load": load,
        "normal_N_max_abs": {"logits": 0.0, "boxes": 0.0},
        "captured_final_max_abs": {"N": 0.0, "P": 0.0},
        "empty_P_identity_max_abs": {"logits": 0.0, "boxes": 0.0},
        "location_input_mutation_max_abs": 0.0,
        "attention_input_mutation_max_abs": 0.0,
        "historical_coeff_max_abs": {"B": [None] * 3, "C": [None] * 3},
        "coeff_min": {"A": [float("inf")] * 3, "E": [float("inf")] * 3},
        "coeff_max": {"A": [float("-inf")] * 3, "E": [float("-inf")] * 3},
        "coeff_shapes": {"A": [None] * 3, "E": [None] * 3},
        "finite_outputs": {"A": True, "E": True},
    }
    regimes = []
    total_counts = Counter()
    total_by_class = defaultdict(Counter)
    processed = 0
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for epoch in args.epochs:
            epoch_seed = args.seed + int(epoch)
            random.seed(epoch_seed)
            np.random.seed(epoch_seed)
            torch.manual_seed(epoch_seed)
            cfg.train_dataloader.set_epoch(epoch)
            if hasattr(cfg.train_dataloader.sampler, "set_epoch"):
                cfg.train_dataloader.sampler.set_epoch(epoch)
            counts = Counter()
            by_class = defaultdict(Counter)
            batch_rows = []
            for step, (samples, targets) in enumerate(cfg.train_dataloader):
                if step >= args.batches_per_epoch:
                    break
                samples = samples.to(device)
                normalized = [{"boxes": target["boxes"].as_subclass(torch.Tensor).to(device), "labels": target["labels"].to(device)} for target in targets]
                gt_xyxy = [box_cxcywh_to_xyxy(target["boxes"]) for target in normalized]
                has_gt = torch.tensor([bool(boxes.shape[0]) for boxes in gt_xyxy], device=device)
                live = model(samples)
                with DecoderInternalCapture(model.decoder) as capture:
                    hooked = model(samples)
                attention, regions = [], []
                for layer in range(3):
                    rec = capture.layers[layer]
                    attention.append(rec["attention_weights"])
                    regions.append(torch.stack([region_mask_with_outside(rec["sampling_locations"][b], gt_xyxy[b])[0] for b in range(len(targets))]))
                gate = normal_bg_dependency(attention, regions) > 0.20
                n_out, n_layers, _, _ = _run_branch(model, capture.decoder_input, "A", gt_xyxy, has_gt, gate, [], set(), audit)
                p_out, p_layers, _, _ = _run_branch(model, capture.decoder_input, "E", gt_xyxy, has_gt, gate, [], set(), audit)
                audit["normal_N_max_abs"]["logits"] = max(audit["normal_N_max_abs"]["logits"], float((live["pred_logits"] - n_out["pred_logits"]).abs().max()))
                audit["normal_N_max_abs"]["boxes"] = max(audit["normal_N_max_abs"]["boxes"], float((live["pred_boxes"] - n_out["pred_boxes"]).abs().max()))
                audit["captured_final_max_abs"]["N"] = max(audit["captured_final_max_abs"]["N"], float((n_layers[-1] - n_out["pred_logits"]).abs().max()))
                audit["captured_final_max_abs"]["P"] = max(audit["captured_final_max_abs"]["P"], float((p_layers[-1] - p_out["pred_logits"]).abs().max()))
                empty = ~has_gt
                if bool(empty.any()):
                    audit["empty_P_identity_max_abs"]["logits"] = max(audit["empty_P_identity_max_abs"]["logits"], float((p_out["pred_logits"][empty] - n_out["pred_logits"][empty]).abs().max()))
                    audit["empty_P_identity_max_abs"]["boxes"] = max(audit["empty_P_identity_max_abs"]["boxes"], float((p_out["pred_boxes"][empty] - n_out["pred_boxes"][empty]).abs().max()))
                selected = select_outcome_targets(n_out, p_out, normalized, score_threshold=0.5, iou_threshold=0.5, negative_iou_threshold=0.3)
                batch_selected = 0
                for target, row in zip(normalized, selected):
                    counts["samples"] += 1
                    counts["gt"] += len(target["labels"])
                    counts["empty_samples"] += int(len(target["labels"]) == 0)
                    current = sum(int(row["counts"].get(key, 0)) for key in ("repair", "protect", "suppress"))
                    batch_selected += current
                    counts["selected"] += current
                    for key, value in row["counts"].items():
                        counts[key] += int(value)
                    for item in row["positive"]:
                        by_class[item["kind"]][str(int(target["labels"][item["gt_index"]]))] += 1
                    for item in row["negative"]:
                        by_class["suppress"][str(item["class_index"])] += 1
                counts["batches"] += 1
                counts["zero_selected_batches"] += int(batch_selected == 0)
                batch_rows.append({"step": step, "gt": sum(len(t["labels"]) for t in normalized), "selected": batch_selected, "empty_samples": int(empty.sum())})
                processed += len(targets)
            regimes.append({"epoch": epoch, "augmentation": "mosaic+mixup" if epoch < 12 else "post-strong-augmentation", "counts": dict(counts), "by_class": {k: dict(v) for k, v in by_class.items()}, "batches": batch_rows})
            total_counts.update(counts)
            for kind, values in by_class.items(): total_by_class[kind].update(values)

    audit["teacher_buffer_max_abs_change"] = max_named_buffer_change(before, model.named_buffers())
    audit["parameter_version_changes"] = [name for name, value in model.named_parameters() if int(value._version) != parameter_versions[name]]
    parity = [*audit["normal_N_max_abs"].values(), *audit["captured_final_max_abs"].values(), *audit["empty_P_identity_max_abs"].values(), audit["location_input_mutation_max_abs"], audit["attention_input_mutation_max_abs"], audit["teacher_buffer_max_abs_change"]]
    if max(parity) > 2e-6 or audit["parameter_version_changes"]:
        raise RuntimeError(f"no-update coverage technical gate failed: {audit}")
    result = {
        "scope": "frozen original EMA; real 15e train dataloader; no optimizer/backward/update",
        "epochs": list(args.epochs),
        "batches_per_epoch": args.batches_per_epoch,
        "batch_size": args.batch_size,
        "regimes": regimes,
        "total_counts": dict(total_counts),
        "total_by_class": {k: dict(v) for k, v in total_by_class.items()},
        "selected_per_100_images": 100.0 * total_counts["selected"] / max(1, total_counts["samples"]),
        "zero_selected_batch_fraction": total_counts["zero_selected_batches"] / max(1, total_counts["batches"]),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        "audit": audit,
        "hashes": {"checkpoint": sha256(args.checkpoint), "config": sha256(args.config), "runner": sha256(__file__)},
    }
    (out_dir / "augmented_coverage.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out_dir / "COMPLETED.txt").write_text("status=passed\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("total_counts", "total_by_class", "selected_per_100_images", "zero_selected_batch_fraction", "peak_cuda_bytes", "audit")}, ensure_ascii=False), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batches-per-epoch", type=int, default=64)
    parser.add_argument("--epochs", type=int, nargs="+", default=[0, 13])
    parser.add_argument("--seed", type=int, default=20260920)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
