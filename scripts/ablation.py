"""Ablation: learnable τ vs several fixed-τ values on the regime-switching task.

Tests whether the learnable per-head time-decay matters. Fixed-τ configurations
pin the window to one of the regimes; if the learnable variant matches or
beats all of them, the extra parameter is pulling its weight.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src import RegimeSwitchLOBDataset, Config, LOBTransformer


def train_val(model, train_loader, val_loader, epochs=5, lr=1e-3):
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_acc = 0.0
    for _ in range(epochs):
        model.train()
        for feats, t, y in train_loader:
            optim.zero_grad(set_to_none=True)
            loss = loss_fn(model(feats, t), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        model.eval()
        correct = n = 0
        with torch.no_grad():
            for feats, t, y in val_loader:
                correct += int((model(feats, t).argmax(-1) == y).sum())
                n += y.size(0)
        best_acc = max(best_acc, correct / n)
    return best_acc


def freeze_tau(model: LOBTransformer, tau_seconds: float) -> None:
    """Set every head's log_tau to log(τ) and freeze it."""
    log_tau = torch.tensor(np.log(tau_seconds)).float()
    for blk in model.blocks:
        with torch.no_grad():
            blk.attn.log_tau.fill_(log_tau.item())
        blk.attn.log_tau.requires_grad_(False)


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    cfg = Config(seq_len=48)
    seeds = 3
    epochs = 5

    configs = [
        ("fixed τ = 0.08 s", 0.08, False),
        ("fixed τ = 0.40 s", 0.40, False),
        ("fixed τ = 1.80 s", 1.80, False),
        ("learnable τ (multiscale init)", None, True),
    ]

    results: dict[str, list[float]] = {name: [] for name, *_ in configs}
    start = time.time()

    for seed in range(seeds):
        print(f"\n=== seed {seed} ===")
        train_ds = RegimeSwitchLOBDataset(n=256, cfg=cfg, seed=1000 + 10 * seed)
        val_ds = RegimeSwitchLOBDataset(n=128, cfg=cfg, seed=9000 + 10 * seed)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        for name, tau, learnable in configs:
            torch.manual_seed(seed)
            model = LOBTransformer(num_features=cfg.num_features, dim=64,
                                   num_heads=4, num_layers=2, max_distance_s=5.0)
            if not learnable:
                freeze_tau(model, tau)
            acc = train_val(model, train_loader, val_loader, epochs=epochs)
            results[name].append(acc)
            print(f"  {name:<32} acc = {acc:.3f}")

    print(f"\n=== Summary over {seeds} seeds ===")
    print(f"  {'variant':<32} {'mean':>8} {'95% CI':>20}")
    for name in results:
        arr = np.array(results[name])
        m = arr.mean()
        h = 2.0 * arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        print(f"  {name:<32} {m:>8.3f} [{m-h:.3f}, {m+h:.3f}]")

    # Plot.
    out = ROOT / "plots"
    out.mkdir(exist_ok=True)
    names = list(results.keys())
    means = [np.mean(results[n]) for n in names]
    errs = [2.0 * np.std(results[n], ddof=1) / np.sqrt(len(results[n])) if len(results[n]) > 1 else 0.0
            for n in names]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#ff8888", "#ffaa88", "#aaaaaa", "#64ffda"]
    bars = ax.bar(np.arange(len(names)), means, yerr=errs, color=colors,
                  edgecolor="black", capsize=6)
    for i, n in enumerate(names):
        for v in results[n]:
            ax.scatter([i], [v], c="black", s=14, alpha=0.6, zorder=3)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("Val accuracy")
    ax.set_title(f"Ablation: fixed τ vs learnable τ ({seeds} seeds)")
    ax.axhline(1/3, color="red", linestyle="--", linewidth=1, label="random baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "tau_ablation.png", dpi=140)
    plt.close(fig)
    print(f"\nPlot → {out}/tau_ablation.png")
    print(f"Total wall-time: {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
