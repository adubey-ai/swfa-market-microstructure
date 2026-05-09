"""Train SWFA vs DeepLOB vs Vanilla Transformer over multiple seeds.

Reports mean ± 95% CI for each metric, per-regime breakdown, and writes plots.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import (
    RegimeSwitchLOBDataset, Config,
    LOBTransformer, DeepLOBLike, VanillaTransformerLOB,
)


def train_one_model(name, build_fn, train_loader, val_loader, device, epochs: int, lr: float):
    torch.manual_seed(0)
    model = build_fn().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_acc = 0.0
    for _ in range(epochs):
        model.train()
        for feats, t, y in train_loader:
            feats, t, y = feats.to(device), t.to(device), y.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(feats, t)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        model.eval()
        correct = n = 0
        with torch.no_grad():
            for feats, t, y in val_loader:
                feats, t, y = feats.to(device), t.to(device), y.to(device)
                pred = model(feats, t).argmax(-1)
                correct += int((pred == y).sum())
                n += y.size(0)
        best_acc = max(best_acc, correct / n)
    return model, best_acc


def per_regime_accuracy(model, dataset, device) -> dict[float, tuple[int, int]]:
    """Return {tau_regime: (correct, total)}."""
    model.eval()
    result: dict[float, list[int]] = {}
    with torch.no_grad():
        for i in range(len(dataset)):
            feats, t, label, tau = dataset.sample_with_regime(i)
            feats_t = torch.from_numpy(feats).unsqueeze(0).to(device)
            ts_t = torch.from_numpy(t).unsqueeze(0).to(device)
            pred = int(model(feats_t, ts_t).argmax(-1).item())
            bucket = result.setdefault(tau, [0, 0])
            bucket[0] += int(pred == label)
            bucket[1] += 1
    return {k: (v[0], v[1]) for k, v in result.items()}


def ci95(x: np.ndarray) -> tuple[float, float, float]:
    """Return (mean, lo, hi) for 95% t-CI."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= 1:
        return float(x.mean()), float(x.min()), float(x.max())
    m = x.mean()
    s = x.std(ddof=1) / np.sqrt(n)
    # t_{0.975, df} for small n; use 2.0 as a safe approximation here.
    h = 2.0 * s
    return m, m - h, m + h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n-train", type=int, default=512)
    parser.add_argument("--n-val", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", type=str, default=str(ROOT / "plots"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    cfg = Config(seq_len=48)
    # Build train + val once per seed so data is the same for all three models in that seed.
    all_accs: dict[str, list[float]] = {"DeepLOB-like": [], "Vanilla Transformer": [], "SWFA LOB": []}
    all_regime: dict[str, dict[float, list[float]]] = {n: {} for n in all_accs}
    final_taus: list[np.ndarray] = []

    builders = {
        "DeepLOB-like": lambda: DeepLOBLike(num_features=cfg.num_features, hidden=48),
        "Vanilla Transformer": lambda: VanillaTransformerLOB(num_features=cfg.num_features, dim=64, num_heads=4, num_layers=2),
        "SWFA LOB": lambda: LOBTransformer(num_features=cfg.num_features, dim=64, num_heads=4, num_layers=2, max_distance_s=5.0),
    }

    for seed in range(args.seeds):
        print(f"\n=== seed {seed} ===")
        train_ds = RegimeSwitchLOBDataset(n=args.n_train, cfg=cfg, seed=1_000 + seed * 10)
        val_ds = RegimeSwitchLOBDataset(n=args.n_val, cfg=cfg, seed=9_000 + seed * 10)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        for name, build in builders.items():
            t0 = time.time()
            model, best = train_one_model(name, build, train_loader, val_loader, device, args.epochs, lr=1e-3)
            all_accs[name].append(best)
            regimes = per_regime_accuracy(model, val_ds, device)
            for tau, (c, n) in regimes.items():
                all_regime[name].setdefault(tau, []).append(c / n)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  {name:<22} best_val={best:.3f}  ({n_params:,} params)  [{time.time()-t0:.1f}s]")

            if name == "SWFA LOB":
                taus_here = []
                for blk in model.blocks:
                    taus_here.append(blk.attn.effective_memory_s().cpu().numpy())
                final_taus.append(np.stack(taus_here))

    print("\n=== Summary (mean ± 95% CI over seeds) ===")
    summary_rows = []
    for name, accs in all_accs.items():
        m, lo, hi = ci95(np.array(accs))
        summary_rows.append((name, m, lo, hi, accs))
        print(f"  {name:<22}  val_acc = {m:.3f}  [95% CI: {lo:.3f}, {hi:.3f}]  seeds={accs}")

    print("\n=== Per-regime val accuracy (mean over seeds) ===")
    regimes_all = sorted({tau for d in all_regime.values() for tau in d.keys()})
    print(f"  {'regime τ (s)':<14}  " + "  ".join(f"{n:>22}" for n in all_accs))
    for tau in regimes_all:
        row = [f"{tau:>14.3f}"]
        for name in all_accs:
            arr = all_regime[name].get(tau, [])
            if arr:
                row.append(f"{np.mean(arr):>22.3f}")
            else:
                row.append(" " * 22)
        print("  " + "  ".join(row))

    # ---- Plot: accuracy with CI ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xs = np.arange(len(all_accs))
    means = [np.mean(all_accs[n]) for n in all_accs]
    err = [2.0 * np.std(all_accs[n], ddof=1) / np.sqrt(len(all_accs[n])) if len(all_accs[n]) > 1 else 0.0
           for n in all_accs]
    colors = ["#88aaff", "#aaaaaa", "#64ffda"]
    ax.bar(xs, means, yerr=err, color=colors, capsize=6, edgecolor="black")
    for i, name in enumerate(all_accs):
        for v in all_accs[name]:
            ax.scatter([i], [v], c="black", s=14, alpha=0.6, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(all_accs.keys()))
    ax.set_ylabel("Validation accuracy")
    ax.set_title(f"3-class LOB direction — mean ± 95% CI over {args.seeds} seeds")
    ax.axhline(1/3, color="red", linestyle="--", linewidth=1, label="random baseline")
    ax.set_ylim(0.2, min(1.0, max(means) + max(err) + 0.1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_ci.png", dpi=140)
    plt.close(fig)

    # ---- Plot: per-regime accuracy ----
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    width = 0.25
    for j, name in enumerate(all_accs):
        ys = [np.mean(all_regime[name].get(t, [0.0])) for t in regimes_all]
        ax.bar(np.arange(len(regimes_all)) + j * width, ys, width, label=name, color=colors[j], edgecolor="black")
    ax.set_xticks(np.arange(len(regimes_all)) + width)
    ax.set_xticklabels([f"τ={t:.2f}s" for t in regimes_all])
    ax.set_ylabel("Val accuracy")
    ax.set_title("Per-regime accuracy (fast / medium / slow momentum decay)")
    ax.axhline(1/3, color="red", linestyle="--", linewidth=1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "per_regime_accuracy.png", dpi=140)
    plt.close(fig)

    # ---- Plot: learned effective memory horizons ----
    if final_taus:
        arr = np.stack(final_taus)            # (seeds, layers, heads)
        avg = arr.mean(axis=0)
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        for layer in range(avg.shape[0]):
            ax.bar(np.arange(avg.shape[1]) + layer * 0.4, avg[layer], 0.35,
                   label=f"layer {layer}")
        for t in RegimeSwitchLOBDataset.TAU_CHOICES:
            ax.axhline(3 * t, color="red", linestyle="--", linewidth=1, alpha=0.5,
                       label=f"true 3τ={3*t:.2f}s" if t == RegimeSwitchLOBDataset.TAU_CHOICES[0] else None)
        ax.set_xlabel("head")
        ax.set_ylabel("effective memory 3τ (seconds)")
        ax.set_title("SWFA learned per-head effective memory horizon")
        ax.legend(loc="upper left", fontsize=9)
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(out_dir / "effective_memory.png", dpi=140)
        plt.close(fig)

    print(f"\nPlots written to {out_dir}/")


if __name__ == "__main__":
    main()
