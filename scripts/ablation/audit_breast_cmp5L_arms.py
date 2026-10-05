#!/usr/bin/env python3
"""Read-only audit of the five-arm breast comparison (cmp5L family).

For every arm of the cmp5L family this script:

1. resolves the YAML inheritance chain and checks the shared contract
   (classes, epochs, patience, batch, accumulation, augmentation schedule,
   ECDet-L neck/decoder hyper-parameters, unified optimizer rule);
2. checks that the declared initialisation weights exist on disk;
3. builds the model on CPU and prints a *detector-side structural signature*
   (sha256 over the sorted (name, shape) pairs of every parameter that does
   NOT start with ``backbone.``). Identical signatures across the five arms
   are the evidence that the neck/decoder/head contract really is shared;
4. reproduces the optimiser grouping of
   ``engine/core/yaml_config.py::get_optim_params`` and reports how many
   parameters land in each group, which proves no parameter is left out and
   no parameter is claimed twice;
5. if an arm declares a ``tuning_checkpoint`` (whole-model init through ``-t``),
   audits that checkpoint against the target state dict (matched /
   shape-mismatch / source-missing) so the expected ``-t`` loading outcome is
   known before training starts. All five cmp5L arms are ``backbone_only`` as
   of the arm-4 rebuild, so this section is dormant for the family; it is kept
   because it is what diagnosed the original arm-4 NaN (11 class-head tensors
   under the 80->4 remap never load).

Nothing is written unless ``--out`` is given, and nothing is ever modified in
the repository. Run it with the training interpreter, e.g.

    /cobot/miniforge3/envs/lw-detr/bin/python \
        scripts/ablation/audit_breast_cmp5L_arms.py --arm all
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ARM_ORDER = ["dinov2s", "dinov2b", "lw_xlarge", "mae_vitb", "ecvits_official"]

ARMS: dict[str, dict] = {
    "dinov2s": {
        "index": 0,
        "config": "ecdetseg/configs/ecdet/ecdet_l_dinov2s_breast_cmp5L_363_100e_es20.yml",
        "backbone": "DinoV2Adapter",
        "init_scope": "backbone_only",
        "init_weights": [
            "/cobot/Code/CODE/eomt/checkpoints/dinov2/vit_small_patch14_reg4_dinov2.pth",
        ],
        "expected_backbone": {
            "backbone_name": "vit_small_patch14_reg4_dinov2",
            "patch_size": 16,
            "interaction_indexes": [10, 11],
            "skip_load_backbone": False,
        },
        "tuning_checkpoint": None,
    },
    "dinov2b": {
        "index": 1,
        "config": "ecdetseg/configs/ecdet/ecdet_l_dinov2b_breast_cmp5L_363_100e_es20.yml",
        "backbone": "DinoV2Adapter",
        "init_scope": "backbone_only",
        "init_weights": [
            "/cobot/Code/CODE/eomt/checkpoints/dinov2/vit_base_patch14_reg4_dinov2.pth",
        ],
        "expected_backbone": {
            "backbone_name": "vit_base_patch14_reg4_dinov2",
            "patch_size": 16,
            "interaction_indexes": [10, 11],
            "skip_load_backbone": False,
        },
        "tuning_checkpoint": None,
    },
    "lw_xlarge": {
        "index": 2,
        "config": "ecdetseg/configs/ecdet/ecdet_l_lw_xlarge_breast_cmp5L_363_100e_es20.yml",
        "backbone": "LWDetrBackbone",
        "init_scope": "backbone_only",
        "init_weights": [
            "/cobot/Code/CODE/LW-DETR-main/pretrain_weights/LWDETR_xlarge_30e_objects365.pth",
        ],
        "expected_backbone": {
            "out_feature_indexes": [8, 9],
            "window_block_indexes": [],
            "projector_type": "ec",
        },
        "tuning_checkpoint": None,
    },
    "mae_vitb": {
        "index": 3,
        "config": "ecdetseg/configs/ecdet/ecdet_l_mae_dino_vitb_breast_cmp5L_363_100e_es20.yml",
        "backbone": "MAEDinoViTBackbone",
        "init_scope": "backbone_only",
        "init_weights": [
            "/cobot/Code/CODE/MAE_DINO/output/dinov2_visible_cls_croped1m_iter500000_base/mae_dino-checkpoint-500000iter.pth",
        ],
        "expected_backbone": {
            "out_feature_indexes": [10, 11],
            "proj_dim": 256,
        },
        "tuning_checkpoint": None,
    },
    "ecvits_official": {
        "index": 4,
        "config": "ecdetseg/configs/ecdet/ecdet_l_ecvits_official_breast_cmp5L_363_100e_es20.yml",
        "backbone": "ViTAdapter",
        "init_scope": "backbone_only",
        "init_weights": [
            "/cobot/Code/wanrui/EdgeCrafter/ecdetseg/ecvits/ecvits_o365_ecdet_l_backbone.pth",
        ],
        "expected_backbone": {
            "name": "ecvits",
            "embed_dim": 384,
            "num_heads": 6,
            "interaction_indexes": [10, 11],
            "skip_load_backbone": False,
        },
        "tuning_checkpoint": None,
    },
}

BREAST_ROOT = "/cobot/Data/Lesion_det/det_breast"
BREAST_ANN = f"{BREAST_ROOT}/annotations/Lesion/Ignore_Delete-Image"

CONTRACT = {
    "num_classes": 4,
    "remap_mscoco_category": False,
    "epochs": 100,
    "early_stop_patience": 20,
    "gradient_accumulation_steps": 2,
    "eval_spatial_size": [640, 640],
}

CONTRACT_DATALOADER = {
    "train_total_batch_size": 16,
    "val_total_batch_size": 16,
    "mosaic_epoch": 24,
    "mosaic_prob": 1,
    "stop_epoch": 98,
    "mixup_prob": 1,
    "mixup_epoch": 24,
    "train_ann_suffix": "Lesion/Ignore_Delete-Image/train.json",
    "val_ann_suffix": "Lesion/Ignore_Delete-Image/valid.json",
    "img_folder": f"{BREAST_ROOT}/img",
}

CONTRACT_NECK = {
    "in_channels": [256, 256, 256],
    "feat_strides": [8, 16, 32],
    "hidden_dim": 256,
    "depth_mult": 1,
    "expansion": 0.75,
    "dim_feedforward": 1024,
}

CONTRACT_DECODER = {
    "feat_channels": [256, 256, 256],
    "feat_strides": [8, 16, 32],
    "hidden_dim": 256,
    "num_levels": 3,
    "num_layers": 4,
    "eval_idx": -1,
    "num_queries": 300,
    "num_points": [3, 6, 3],
    "dim_feedforward": 1024,
}


def _signature(named: list[tuple[str, tuple[int, ...]]]) -> str:
    payload = "\n".join(f"{name}:{shape}" for name, shape in sorted(named))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_facts(path: str) -> dict:
    target = Path(path)
    if not target.is_file():
        return {"path": path, "exists": False}
    stat = target.stat()
    return {
        "path": path,
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    }


def _load_state(path: str) -> dict:
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "ema" in checkpoint and isinstance(checkpoint["ema"], dict):
        return checkpoint["ema"]["module"], sorted(checkpoint.keys())
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"], sorted(checkpoint.keys())
    if isinstance(checkpoint, dict):
        return checkpoint, sorted(checkpoint.keys())
    raise TypeError(f"No usable state dict in {path} ({type(checkpoint).__name__})")


def audit_arm(project_root: Path, arm: str, build_model: bool) -> dict:
    spec = ARMS[arm]
    sys.path.insert(0, str(project_root))
    from ecdetseg.engine.core.yaml_utils import load_config

    result: dict = {
        "arm": arm,
        "index": spec["index"],
        "config": spec["config"],
        "declared_backbone": spec["backbone"],
        "init_scope": spec["init_scope"],
        "contract_failures": [],
    }

    config_path = project_root / spec["config"]
    cfg = load_config(str(config_path), {})

    def check(label: str, expected, actual) -> None:
        if expected != actual:
            result["contract_failures"].append(
                {"field": label, "expected": expected, "actual": actual}
            )

    for key, expected in CONTRACT.items():
        check(key, expected, cfg.get(key))

    train_ds = cfg["train_dataloader"]["dataset"]
    train_tf = train_ds.get("transforms", {})
    check("train_batch", CONTRACT_DATALOADER["train_total_batch_size"], cfg["train_dataloader"]["total_batch_size"])
    check("val_batch", CONTRACT_DATALOADER["val_total_batch_size"], cfg["val_dataloader"]["total_batch_size"])
    check("mosaic_epoch", CONTRACT_DATALOADER["mosaic_epoch"], train_tf.get("mosaic_epoch"))
    check("mosaic_prob", CONTRACT_DATALOADER["mosaic_prob"], train_tf.get("mosaic_prob"))
    check("stop_epoch", CONTRACT_DATALOADER["stop_epoch"], train_tf.get("stop_epoch"))
    check("mixup_prob", CONTRACT_DATALOADER["mixup_prob"], cfg["train_dataloader"]["collate_fn"].get("mixup_prob"))
    check("mixup_epoch", CONTRACT_DATALOADER["mixup_epoch"], cfg["train_dataloader"]["collate_fn"].get("mixup_epoch"))
    check("train_img_folder", CONTRACT_DATALOADER["img_folder"], train_ds.get("img_folder"))
    check("val_img_folder", CONTRACT_DATALOADER["img_folder"], cfg["val_dataloader"]["dataset"].get("img_folder"))
    check("train_ann", CONTRACT_DATALOADER["train_ann_suffix"], str(train_ds.get("ann_file", "")).split("/annotations/")[-1])
    check("val_ann", CONTRACT_DATALOADER["val_ann_suffix"], str(cfg["val_dataloader"]["dataset"].get("ann_file", "")).split("/annotations/")[-1])

    for key, expected in CONTRACT_NECK.items():
        check(f"HybridEncoder.{key}", expected, cfg["HybridEncoder"].get(key))
    for key, expected in CONTRACT_DECODER.items():
        check(f"ECTransformer.{key}", expected, cfg["ECTransformer"].get(key))

    check("ECDet.encoder", "HybridEncoder", cfg["ECDet"]["encoder"])
    check("ECDet.decoder", "ECTransformer", cfg["ECDet"]["decoder"])
    check("ECDet.backbone", spec["backbone"], cfg["ECDet"]["backbone"])
    check("optimizer.lr", 0.0005, cfg["optimizer"]["lr"])
    check("optimizer.type", "AdamW", cfg["optimizer"]["type"])

    backbone_section = cfg.get(spec["backbone"], {})
    for key, expected in spec["expected_backbone"].items():
        check(f"{spec['backbone']}.{key}", expected, backbone_section.get(key))

    # --- initialisation weights ------------------------------------------------
    result["init_files"] = [_file_facts(path) for path in spec["init_weights"]]
    result["init_files_ok"] = all(item["exists"] for item in result["init_files"])

    # --- model build -----------------------------------------------------------
    if not build_model:
        return result

    import torch
    from ecdetseg.engine.core import YAMLConfig
    from ecdetseg.engine.core.yaml_config import YAMLConfig as _YC  # noqa: F401  (grouping helper)

    yaml_cfg = YAMLConfig(str(config_path))
    model = yaml_cfg.model.cpu().eval()
    state = model.state_dict()

    detector_named = [(k, tuple(v.shape)) for k, v in state.items() if not k.startswith("backbone.")]
    backbone_named = [(k, tuple(v.shape)) for k, v in state.items() if k.startswith("backbone.")]
    result["detector_signature"] = _signature(detector_named)
    result["detector_tensor_count"] = len(detector_named)
    result["detector_param_count"] = int(
        sum(v.numel() for k, v in state.items() if not k.startswith("backbone."))
    )
    result["backbone_tensor_count"] = len(backbone_named)
    result["backbone_param_count"] = int(
        sum(v.numel() for k, v in state.items() if k.startswith("backbone."))
    )

    neck_named = [(k, tuple(v.shape)) for k, v in state.items() if k.startswith("encoder.")]
    dec_named = [(k, tuple(v.shape)) for k, v in state.items() if k.startswith("decoder.")]
    result["neck_signature"] = _signature(neck_named)
    result["decoder_signature"] = _signature(dec_named)
    result["neck_param_count"] = int(sum(v.numel() for k, v in state.items() if k.startswith("encoder.")))
    result["decoder_param_count"] = int(sum(v.numel() for k, v in state.items() if k.startswith("decoder.")))
    result["num_queries"] = int(state["decoder.enc_output.0.weight"].shape[0]) if "decoder.enc_output.0.weight" in state else "see CONTRACT_DECODER"
    result["num_classes_from_head"] = int(state["decoder.enc_score_head.weight"].shape[0]) if "decoder.enc_score_head.weight" in state else None

    # --- optimiser grouping (reproduces engine/core/yaml_config.py) ------------
    groups = _YC.get_optim_params(cfg["optimizer"], model)
    group_sizes = []
    for index, group in enumerate(groups):
        group_sizes.append(
            {
                "index": index,
                "params": len(group["params"]),
                "lr": group.get("lr", cfg["optimizer"]["lr"]),
                "weight_decay": group.get("weight_decay", cfg["optimizer"]["weight_decay"]),
                "pattern": group.get("params_regex", None) or cfg["optimizer"]["params"][index]["params"] if index < len(cfg["optimizer"]["params"]) else "<default-unmatched-group>",
            }
        )
    total_named = len([k for k, v in model.named_parameters() if v.requires_grad])
    covered = sum(item["params"] for item in group_sizes)
    result["optimizer_groups"] = group_sizes
    result["optimizer_covered_params"] = covered
    result["optimizer_total_params"] = total_named
    result["optimizer_grouping_ok"] = covered == total_named

    # --- tuning checkpoint audit (official-EC arm only) ------------------------
    tuning = spec["tuning_checkpoint"]
    if tuning:
        try:
            source_state, top_keys = _load_state(tuning)
            source_state = {
                k: v for k, v in source_state.items() if isinstance(v, torch.Tensor)
            }
            matched, shape_mismatch, source_missing = [], [], []
            for key, value in state.items():
                if key not in source_state:
                    source_missing.append(key)
                elif tuple(source_state[key].shape) != tuple(value.shape):
                    shape_mismatch.append(key)
                else:
                    matched.append(key)
            result["tuning_checkpoint"] = {
                "path": tuning,
                "top_level_keys": top_keys,
                "source_tensor_count": len(source_state),
                "matched": len(matched),
                "shape_mismatch": shape_mismatch,
                "source_missing": source_missing,
                "note": (
                    "engine/solver/_solver.py::load_tuning_state first tries "
                    "_adjust_head_parameters; the 80->4 class mapping raises and the "
                    "shape-based fallback then skips every entry listed under "
                    "shape_mismatch and loads everything under source_missing that is "
                    "present in the source (none here)."
                ),
            }
        except Exception as error:  # pragma: no cover - reporting path
            result["tuning_checkpoint"] = {"path": tuning, "error": repr(error)}

    del model, state
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/cobot/Code/wanrui/EdgeCrafter")
    parser.add_argument("--arm", default="all", choices=["all", *ARM_ORDER])
    parser.add_argument("--out", default=None, help="optional JSON output path")
    parser.add_argument("--no-model", action="store_true", help="config-only checks (no model build)")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    selected = ARM_ORDER if args.arm == "all" else [args.arm]

    reports = []
    for arm in selected:
        reports.append(audit_arm(project_root, arm, build_model=not args.no_model))

    signatures = {report["detector_signature"] for report in reports if "detector_signature" in report}
    summary = {
        "arms": reports,
        "detector_signature_unique_count": len(signatures),
        "detector_structure_aligned": len(signatures) == 1 and len(reports) == len(ARM_ORDER),
        "any_contract_failure": any(report["contract_failures"] for report in reports),
        "any_missing_init_file": any(not report.get("init_files_ok", False) for report in reports),
    }

    header = f"{'arm':<17}{'idx':>4}{'backbone':<24}{'det.tensors':>12}{'det.params':>12}{'bkb.params':>12}{'opt.groups.ok':>14}{'init.ok':>9}{'contract':>10}"
    print(header)
    print("-" * len(header))
    for report in reports:
        print(
            f"{report['arm']:<17}{report['index']:>4}{report['declared_backbone']:<24}"
            f"{report.get('detector_tensor_count', -1):>12}{report.get('detector_param_count', -1):>12}"
            f"{report.get('backbone_param_count', -1):>12}{str(report.get('optimizer_grouping_ok', '-')):>14}"
            f"{str(report.get('init_files_ok', '-')):>9}{str(not report['contract_failures']):>10}"
        )
    print()
    print(f"detector structural signature(s) seen: {sorted(signatures)}")
    if len(reports) == len(ARM_ORDER):
        print(f"detector_structure_aligned = {summary['detector_structure_aligned']}")
    else:
        print("detector_structure_aligned = n/a (single-arm preflight; run --arm all for the alignment proof)")
    for report in reports:
        for failure in report["contract_failures"]:
            print(f"  [contract] {report['arm']}: {failure['field']} expected={failure['expected']!r} actual={failure['actual']!r}")
        tuning = report.get("tuning_checkpoint")
        if tuning:
            print(
                f"  [tuning] {report['arm']}: top_keys={tuning.get('top_level_keys')} "
                f"matched={tuning.get('matched')}/{tuning.get('source_tensor_count')} "
                f"shape_mismatch={len(tuning.get('shape_mismatch', []))} "
                f"source_missing={len(tuning.get('source_missing', []))}"
            )
            for key in tuning.get("shape_mismatch", []):
                print(f"      shape-mismatch (will NOT load, stays random): {key}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nfull JSON report: {out_path}")

    if summary["any_contract_failure"] or summary["any_missing_init_file"]:
        print("\nAUDIT RESULT: FAIL", file=sys.stderr)
        return 1
    print("\nAUDIT RESULT: OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
