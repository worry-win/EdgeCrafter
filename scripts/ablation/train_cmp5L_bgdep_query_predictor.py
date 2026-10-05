"""Experiment 1: fit small GT-label-only BG-dependency predictors."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ablation.cmp5L_bgdep_proxy import binary_metrics, regression_metrics


class Predictor(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 0):
        super().__init__()
        self.net = (
            nn.Linear(input_dim, 1)
            if hidden <= 0
            else nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def feature_matrix(payload, variant: str) -> np.ndarray:
    emb = payload["emb"]
    stats = payload["stats"]
    q0, q1, q2 = emb[:, :256], emb[:, 256:512], emb[:, 512:768]
    query = {
        "q_l0": q0,
        "q_l1": q1,
        "q_l2": q2,
        "q_concat": emb,
        "q_mean": (q0 + q1 + q2) / 3.0,
    }
    if variant in query:
        return query[variant]
    # 4 global stats + 6 non-value stats per layer. Last two per layer are sampled-value stats.
    internal_idx = list(range(4))
    for li in range(3):
        base = 4 + li * 8
        internal_idx.extend(range(base, base + 6))
    if variant == "q_internal":
        return np.concatenate([emb, stats[:, internal_idx]], axis=1)
    if variant == "q_sampled_stats":
        return np.concatenate([emb, stats], axis=1)
    raise ValueError(f"unknown feature variant: {variant}")


def predict(model, x, mean, std, device, batch_size=32768):
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(device=device, dtype=torch.float32)
            xb = (xb - mean) / std
            outputs.append(model(xb).cpu())
    return torch.cat(outputs).numpy()


def train_one(x, y, task, hidden, device, epochs, seed):
    generator = torch.Generator().manual_seed(seed)
    mean = torch.from_numpy(x.mean(0)).to(device=device, dtype=torch.float32)
    std = torch.from_numpy(x.std(0)).to(device=device, dtype=torch.float32).clamp_min(1e-5)
    model = Predictor(x.shape[1], hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y.astype(np.float32))
    batch_size = 8192
    if task == "classification":
        n_pos = float(y.sum())
        pos_weight = torch.tensor([(len(y) - n_pos) / max(n_pos, 1.0)], device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(y_tensor), generator=generator)
        for start in range(0, len(y_tensor), batch_size):
            idx = order[start:start + batch_size]
            xb = x_tensor[idx].to(device=device, dtype=torch.float32)
            yb = y_tensor[idx].to(device=device)
            prediction = model((xb - mean) / std)
            loss = loss_fn(prediction, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model, mean, std


def best_f1_threshold(target, score):
    candidates = np.linspace(0.05, 0.95, 91)
    scored = [(binary_metrics(target, score, float(t))["f1"], float(t)) for t in candidates]
    return max(scored)[1]


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train = np.load(args.train)
    valid = np.load(args.valid)
    variants = ["q_l0", "q_l1", "q_l2", "q_concat", "q_mean", "q_internal", "q_sampled_stats"]
    results = {}
    best = None
    for variant in variants:
        x_train = feature_matrix(train, variant).astype(np.float32, copy=False)
        x_valid = feature_matrix(valid, variant).astype(np.float32, copy=False)
        records = {}

        regression, rmean, rstd = train_one(
            x_train, train["d_bg"], "regression", 0, device, args.linear_epochs, args.seed
        )
        rpred = predict(regression, x_valid, rmean, rstd, device)
        records["linear_regression"] = regression_metrics(valid["d_bg"], rpred)

        for name, hidden in (("logistic_regression", 0), ("mlp_2layer", args.hidden)):
            model, mean, std = train_one(
                x_train, train["g_bg"], "classification", hidden, device,
                args.mlp_epochs if hidden else args.linear_epochs, args.seed,
            )
            probability = 1.0 / (1.0 + np.exp(-predict(model, x_valid, mean, std, device)))
            threshold = best_f1_threshold(valid["g_bg"], probability)
            metrics = binary_metrics(valid["g_bg"], probability, threshold=threshold)
            records[name] = metrics
            candidate = (metrics["pr_auc"], metrics["roc_auc"], variant, name)
            if best is None or candidate > best[0]:
                best = (candidate, model, mean, std, threshold, metrics, x_train.shape[1], hidden)
        results[variant] = records
        print(json.dumps({variant: records}, indent=2), flush=True)

    candidate, model, mean, std, threshold, metrics, input_dim, hidden = best
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "variant": candidate[2],
        "model_type": candidate[3],
        "input_dim": input_dim,
        "hidden": hidden,
        "mean": mean.cpu(),
        "std": std.cpu(),
        "threshold": threshold,
        "metrics": metrics,
        "bgdep_tau": 0.20,
        "seed": args.seed,
    }
    torch.save(checkpoint, out_dir / "best_query_predictor.pth")
    payload = {
        "experiment": 1,
        "split_policy": "train predictor fit; validation model/threshold selection; test untouched",
        "train_file": args.train,
        "valid_file": args.valid,
        "n_train_queries": int(len(train["g_bg"])),
        "n_valid_queries": int(len(valid["g_bg"])),
        "best": {"variant": candidate[2], "model_type": candidate[3], **metrics},
        "results": results,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "COMPLETED.txt").write_text("status=passed\n", encoding="utf-8")
    print(json.dumps(payload["best"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--linear-epochs", type=int, default=5)
    parser.add_argument("--mlp-epochs", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
