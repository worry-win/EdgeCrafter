"""Full-test COCO evaluation for GT-conditioned feature privilege conditions."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch


EC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EC_ROOT / "ecdetseg"))
sys.path.insert(0, str(EC_ROOT))

from engine.solver.ec_engine import (  # noqa: E402
    summarize_pr_curve_f1,
    summarize_yolo_pr_curve_metrics,
)
from scripts.ablation.probe_cmp5L_decoder_internal_tensors import _build  # noqa: E402
from scripts.ablation.probe_cmp5L_privileged_ema_oracle import (  # noqa: E402
    _condition_specs,
    privilege_features,
)
from scripts.ablation.cmp5L_privileged_metrics import metrics_row, render_markdown  # noqa: E402


def _forward_condition(model, base_features, targets, image_hw, spec):
    features, _ = privilege_features(
        base_features,
        targets,
        image_hw,
        spec["background_weight"],
        spec["context_weight"],
        spec["context_scale"],
    )
    return model.decoder(model.encoder(features))


def evaluate(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; refusing silent CPU fallback")
    args.device = device
    cfg, solver, model = _build(args)
    specs = _condition_specs(args.context_alpha, args.context_beta, args.context_scale)
    evaluators = {spec["name"]: copy.deepcopy(cfg.evaluator) for spec in specs}
    for evaluator in evaluators.values():
        evaluator.cleanup()

    seen = 0
    parity = None
    started = time.time()
    with torch.no_grad():
        for samples, targets in cfg.val_dataloader:
            samples = samples.to(device)
            base_features = model.backbone(samples)
            for spec in specs:
                outputs = _forward_condition(
                    model, base_features, targets, samples.shape[-2:], spec
                )
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                if spec["name"] == "normal" and parity is None:
                    baseline = model(samples)
                    parity = {
                        "logits": float((outputs["pred_logits"] - baseline["pred_logits"]).abs().max()),
                        "boxes": float((outputs["pred_boxes"] - baseline["pred_boxes"]).abs().max()),
                    }
                    if max(parity.values()) > 2e-6:
                        raise RuntimeError(f"normal composed-forward parity failed: {parity}")
                original_sizes = torch.stack([target["orig_size"] for target in targets]).to(device)
                results = solver.postprocessor(outputs, original_sizes)
                predictions = {
                    int(target["image_id"].item()): result
                    for target, result in zip(targets, results)
                }
                evaluators[spec["name"]].update(predictions)
            seen += len(targets)
            if args.log_every and seen % args.log_every < len(targets):
                print(f"[{seen}/{len(cfg.val_dataloader.dataset)}] {time.time() - started:.1f}s", flush=True)

    if args.expected_images and seen != args.expected_images:
        raise RuntimeError(f"evaluated {seen} images, expected {args.expected_images}")

    rows = []
    details = {}
    for spec in specs:
        name = spec["name"]
        evaluator = evaluators[name]
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()
        coco_eval = evaluator.coco_eval["bbox"]
        macro = summarize_pr_curve_f1(coco_eval)
        yolo = summarize_yolo_pr_curve_metrics(coco_eval, evaluator.coco_gt)
        stats = {
            "coco_eval_bbox": coco_eval.stats.tolist(),
            "macro_f1_iou50": macro["f1_iou50"],
            **yolo,
        }
        rows.append(metrics_row(name, stats, seen))
        details[name] = stats

    payload = {
        "meta": {
            "checkpoint": args.checkpoint,
            "weights": args.weights,
            "ann_file": args.ann_file,
            "n_images": seen,
            "conditions": specs,
            "normal_parity": parity,
            "seconds": round(time.time() - started, 2),
            "f1_definition": (
                "IoU=.50; independent best COCO PR operating point per class; "
                "macro mean precision/recall and harmonic-mean F1"
            ),
        },
        "rows": rows,
        "details": details,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_markdown(rows)
    if args.markdown_out:
        Path(args.markdown_out).write_text(markdown, encoding="utf-8")
    print(markdown)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--img-folder", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--markdown-out")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--context-alpha", type=float, default=0.5)
    parser.add_argument("--context-beta", type=float, default=0.2)
    parser.add_argument("--context-scale", type=float, default=1.5)
    parser.add_argument("--expected-images", type=int, default=2011)
    parser.add_argument("--log-every", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
