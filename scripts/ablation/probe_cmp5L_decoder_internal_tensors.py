"""Validate a non-invasive tensor contract for the cmp5L decoder.

The capture uses PyTorch forward/pre-forward hooks only.  It never changes the
model's train/eval state and never replaces a module's forward implementation.
Per-layer score/LQE outputs are replayed from the query and distribution tensors
that the genuine eval forward computed; the final replay is checked against the
model return value.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchvision


EC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EC_ROOT / "ecdetseg"))

from engine.core import YAMLConfig  # noqa: E402
from engine.misc import dist_utils  # noqa: E402
from engine.solver import TASKS  # noqa: E402
from engine.edgecrafter.utils import (  # noqa: E402
    distance2bbox,
    inverse_sigmoid,
    weighting_function,
)


def _tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"expected tensor-like module output, got {type(output).__name__}")


class DecoderInternalCapture:
    """Hook-only capture for one ECTransformer eval forward."""

    def __init__(self, ec_transformer):
        self.ec = ec_transformer
        self.td = ec_transformer.decoder
        self.handles = []
        self.layers = [dict() for _ in self.td.layers]
        self.decoder_input = {}
        self.live_final = {}

    @staticmethod
    def _save(bucket, key):
        def hook(_module, _inputs, output):
            # Some activations are in-place (SiLU in this repository).  Clone so
            # a later child module cannot mutate the value observed at this site.
            bucket[key] = _tensor(output).detach().clone()
            return None
        return hook

    def _decoder_pre(self, _module, inputs):
        for layer in self.layers:
            layer.clear()
        self.decoder_input.clear()
        self.decoder_input.update({
            "query_content": inputs[1].detach().clone(),
            "reference_unact": inputs[2].detach().clone(),
            "memory": inputs[3].detach().clone(),
            "spatial_shapes": tuple(tuple(int(v) for v in s) for s in inputs[4]),
        })
        self.live_final.clear()
        return None

    def _layer_pre(self, index):
        def hook(_module, inputs):
            self.layers[index]["query_in"] = inputs[0].detach().clone()
            self.layers[index]["reference_in"] = inputs[1].detach().clone()
            return None
        return hook

    def _cross_post(self, index, module):
        def hook(_module, inputs, output):
            rec = self.layers[index]
            query, reference_points, _value, spatial_shapes = inputs[:4]
            batch, queries = query.shape[:2]
            offsets = rec["sampling_offsets_flat"].reshape(
                batch, queries, module.num_heads, sum(module.num_points_list), 2
            )
            weight_logits = rec["attention_weight_logits_flat"].reshape(
                batch, queries, module.num_heads, sum(module.num_points_list)
            )
            weights = weight_logits.softmax(-1)
            if reference_points.shape[-1] == 4:
                point_scale = module.num_points_scale.to(query.dtype).unsqueeze(-1)
                offset = (
                    offsets
                    * point_scale
                    * reference_points[:, :, None, :, 2:]
                    * module.offset_scale
                )
                locations = reference_points[:, :, None, :, :2] + offset
            elif reference_points.shape[-1] == 2:
                normalizer = torch.as_tensor(
                    spatial_shapes, device=query.device, dtype=query.dtype
                ).flip([1]).reshape(1, 1, 1, module.num_levels, 1, 2)
                shaped = offsets.reshape(
                    batch, queries, module.num_heads, module.num_levels, -1, 2
                )
                locations = reference_points.reshape(
                    batch, queries, 1, module.num_levels, 1, 2
                ) + shaped / normalizer
                locations = locations.flatten(3, 4)
            else:
                raise ValueError("reference point width must be 2 or 4")
            rec["sampling_offsets"] = offsets.detach().clone()
            rec["attention_weight_logits"] = weight_logits.detach().clone()
            rec["attention_weights"] = weights.detach().clone()
            rec["sampling_locations"] = locations.detach().clone()
            rec["cross_attn_output"] = _tensor(output).detach().clone()
            rec["num_points_list"] = tuple(module.num_points_list)
            rec["spatial_shapes"] = tuple(tuple(int(v) for v in s) for s in spatial_shapes)
            return None
        return hook

    def __enter__(self):
        if self.ec.training or self.td.training:
            raise RuntimeError("DecoderInternalCapture requires a genuine eval-mode model")
        self.handles.append(self.td.register_forward_pre_hook(self._decoder_pre))
        for index, layer in enumerate(self.td.layers):
            rec = self.layers[index]
            self.handles.extend([
                layer.register_forward_pre_hook(self._layer_pre(index)),
                layer.self_attn.register_forward_hook(self._save(rec, "self_attn_output")),
                layer.norm1.register_forward_hook(self._save(rec, "self_attn_residual")),
                layer.cross_attn.sampling_offsets.register_forward_hook(
                    self._save(rec, "sampling_offsets_flat")
                ),
                layer.cross_attn.attention_weights.register_forward_hook(
                    self._save(rec, "attention_weight_logits_flat")
                ),
                layer.cross_attn.register_forward_hook(self._cross_post(index, layer.cross_attn)),
                layer.gateway.register_forward_hook(self._save(rec, "cross_attn_residual")),
                layer.linear1.register_forward_hook(self._save(rec, "ffn_linear1")),
                layer.activation.register_forward_hook(self._save(rec, "ffn_activation")),
                layer.linear2.register_forward_hook(self._save(rec, "ffn_linear2")),
                layer.norm2.register_forward_hook(self._save(rec, "query_out")),
            ])
            if self.ec.dec_bbox_head is not None:
                self.handles.append(
                    self.ec.dec_bbox_head[index].register_forward_hook(
                        self._save(rec, "bbox_distribution_delta")
                    )
                )
        if self.ec.pre_bbox_head is not None:
            self.handles.append(
                self.ec.pre_bbox_head.register_forward_hook(
                    self._save(self.decoder_input, "pre_bbox_delta")
                )
            )
        self.handles.append(
            self.ec.query_pos_head.register_forward_hook(
                self._save(self.decoder_input, "query_pos")
            )
        )
        final_index = self.td.eval_idx
        self.handles.append(
            self.ec.dec_score_head[final_index].register_forward_hook(
                self._save(self.live_final, "raw_logits")
            )
        )
        if self.td.lqe_layers is not None:
            self.handles.append(
                self.td.lqe_layers[final_index].register_forward_hook(
                    self._save(self.live_final, "lqe_logits")
                )
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False

    def replay_decoder_heads(self):
        """Recover every layer's raw/LQE logits and refined box in eval mode."""
        if not self.layers or "query_out" not in self.layers[-1]:
            raise RuntimeError("capture is empty; run one forward inside the context")
        initial_reference = self.decoder_input["reference_unact"].sigmoid()
        pre_delta = self.decoder_input.get("pre_bbox_delta")
        if pre_delta is None:
            raise RuntimeError("FDR replay requires the observed pre_bbox_head output")
        fdr_reference = (pre_delta + inverse_sigmoid(initial_reference)).sigmoid().detach()
        if hasattr(self.td, "project"):
            project = self.td.project
        else:
            project = weighting_function(self.ec.reg_max, self.ec.up, self.ec.reg_scale)

        raw_logits, lqe_logits, boxes, corners = [], [], [], []
        cumulative = None
        with torch.no_grad():
            for index, rec in enumerate(self.layers):
                delta = rec["bbox_distribution_delta"]
                cumulative = delta if cumulative is None else delta + cumulative
                raw = self.ec.dec_score_head[index](rec["query_out"])
                score = (
                    self.td.lqe_layers[index](raw, cumulative)
                    if self.td.lqe_layers is not None
                    else raw
                )
                distance = self.ec.integral(cumulative, project)
                box = distance2bbox(fdr_reference, distance, self.ec.reg_scale)
                raw_logits.append(raw)
                lqe_logits.append(score)
                boxes.append(box)
                corners.append(cumulative)
        return {
            "initial_reference": initial_reference,
            "fdr_reference": fdr_reference,
            "raw_logits": torch.stack(raw_logits),
            "lqe_logits": torch.stack(lqe_logits),
            "boxes": torch.stack(boxes),
            "corners": torch.stack(corners),
        }

    def cpu_snapshot(self):
        def cpu(value):
            if torch.is_tensor(value):
                return value.detach().cpu()
            if isinstance(value, dict):
                return {key: cpu(item) for key, item in value.items()}
            if isinstance(value, list):
                return [cpu(item) for item in value]
            return value
        return {
            "decoder_input": cpu(self.decoder_input),
            "layers": cpu(self.layers),
            "live_final": cpu(self.live_final),
        }


def _stats(tensor):
    value = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
    }


def _normalise_targets(targets, height, width, device):
    result = []
    scale = torch.tensor([width, height, width, height], device=device)
    for target in targets:
        boxes = target["boxes"].as_subclass(torch.Tensor).to(device)
        boxes = torchvision.ops.box_convert(boxes, "xyxy", "cxcywh") / scale
        result.append({"boxes": boxes, "labels": target["labels"].to(device)})
    return result


def _build(args):
    cfg = YAMLConfig(args.config, **{
        "output_dir": str(Path(args.out).resolve().parent / "_solver_setup"),
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
    solver.load_resume_state(args.checkpoint)
    model = solver.ema.module if args.weights == "ema" and solver.ema is not None else solver.model
    model = dist_utils.de_parallel(model).to(args.device).eval()
    return cfg, solver, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--img-folder", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--json-out")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; refusing silent CPU fallback")
    args.device = device

    cfg, solver, model = _build(args)
    records = []
    seen = 0
    for samples, targets in cfg.val_dataloader:
        if seen >= args.limit:
            break
        samples = samples.to(device)
        with torch.no_grad():
            baseline = model(samples)
            with DecoderInternalCapture(model.decoder) as capture:
                observed = model(samples)
            replay = capture.replay_decoder_heads()
        parity = {
            "hook_logits_max_abs": float((baseline["pred_logits"] - observed["pred_logits"]).abs().max()),
            "hook_boxes_max_abs": float((baseline["pred_boxes"] - observed["pred_boxes"]).abs().max()),
            "replay_logits_max_abs": float((replay["lqe_logits"][capture.td.eval_idx] - observed["pred_logits"]).abs().max()),
            "replay_boxes_max_abs": float((replay["boxes"][capture.td.eval_idx] - observed["pred_boxes"]).abs().max()),
        }
        if max(parity.values()) > 2e-6:
            raise RuntimeError(f"capture/replay changed or failed to reproduce outputs: {parity}")

        matcher_targets = _normalise_targets(
            targets, samples.shape[-2], samples.shape[-1], device
        )
        matches = solver.criterion.matcher(observed, matcher_targets)["indices"]
        snapshot = capture.cpu_snapshot()
        contract = {
            "input": _stats(samples),
            "decoder.query_content": _stats(snapshot["decoder_input"]["query_content"]),
            "decoder.initial_reference_unact": _stats(snapshot["decoder_input"]["reference_unact"]),
            "decoder.initial_reference_sigmoid": _stats(replay["initial_reference"]),
            "decoder.fdr_reference": _stats(replay["fdr_reference"]),
            "final.raw_logits": _stats(replay["raw_logits"][capture.td.eval_idx]),
            "final.lqe_logits": _stats(replay["lqe_logits"][capture.td.eval_idx]),
            "final.probability": _stats(replay["lqe_logits"][capture.td.eval_idx].sigmoid()),
            "final.box": _stats(replay["boxes"][capture.td.eval_idx]),
        }
        for index, layer in enumerate(snapshot["layers"]):
            for key in (
                "reference_in", "sampling_offsets", "sampling_locations",
                "attention_weight_logits", "attention_weights", "query_in",
                "self_attn_output", "self_attn_residual", "cross_attn_output",
                "cross_attn_residual", "ffn_linear1", "ffn_activation",
                "ffn_linear2", "query_out", "bbox_distribution_delta",
            ):
                contract[f"decoder.layer{index}.{key}"] = _stats(layer[key])
            contract[f"decoder.layer{index}.raw_logits"] = _stats(replay["raw_logits"][index])
            contract[f"decoder.layer{index}.lqe_logits"] = _stats(replay["lqe_logits"][index])
            contract[f"decoder.layer{index}.refined_box"] = _stats(replay["boxes"][index])
        record = {
            "image_ids": [int(t["image_id"].item()) for t in targets],
            "model_training": model.training,
            "decoder_training": model.decoder.decoder.training,
            "eval_idx": capture.td.eval_idx,
            "num_points_list": list(capture.td.layers[0].cross_attn.num_points_list),
            "parity": parity,
            "hungarian_matches": [
                {"query": q.tolist(), "gt": g.tolist()} for q, g in matches
            ],
            "contract": contract,
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False, indent=2))
        seen += len(targets)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
