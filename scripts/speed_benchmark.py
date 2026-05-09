"""Measure wall-clock latency + theoretical FLOPs for SWFA vs Vanilla attention.

Sweeps sequence length N ∈ {64, 128, 256, 512, 1024}. The vanilla transformer
is O(N^2), SWFA with window τ_max is effectively O(N·K_eff). Expected: SWFA
overtakes vanilla's wall-clock once N·K_eff < N^2 with a meaningful gap.
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

from src import TimeDecayedSWFA
from src.flops import vanilla_attention_flops, swfa_attention_flops, window_fraction


def bench_forward(module_fn, N: int, D: int, dt_mean: float, n_iter: int = 50) -> float:
    """Median forward latency in milliseconds."""
    module = module_fn()
    x = torch.randn(2, N, D)
    t = torch.cumsum(torch.full((2, N), dt_mean), dim=1)

    # Warmup.
    for _ in range(5):
        module(x, t) if "t" in module.forward.__code__.co_varnames else module(x)

    timings = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        module(x, t) if "t" in module.forward.__code__.co_varnames else module(x)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(timings))


class VanillaCausalAttention(torch.nn.Module):
    """Reference vanilla causal attention — no sparsity."""
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.dim = dim

    def forward(self, x, t=None):
        N = x.size(1)
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        return out


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    D, H = 64, 4
    # 2000 events/s (realistic high-activity LOB) with a 0.2s window:
    # at N=1024 only the last ~400 tokens are in-window — SWFA should dominate.
    dt_mean = 0.0005
    max_window_s = 0.2
    seq_lens = [64, 128, 256, 512, 1024, 2048]

    results = []
    for N in seq_lens:
        t_vanilla = bench_forward(lambda: VanillaCausalAttention(D, H), N, D, dt_mean, n_iter=30)
        t_swfa = bench_forward(lambda: TimeDecayedSWFA(D, H, max_distance_s=max_window_s),
                               N, D, dt_mean, n_iter=30)

        t_sample = np.cumsum(np.full(N, dt_mean))
        frac = window_fraction(t_sample, max_window_s)
        fv = vanilla_attention_flops(N, D, H).flops_total
        fs = swfa_attention_flops(N, D, H, in_window_frac=frac).flops_total

        results.append({"N": N, "t_vanilla_ms": t_vanilla, "t_swfa_ms": t_swfa,
                        "flops_vanilla_M": fv / 1e6, "flops_swfa_M": fs / 1e6,
                        "window_frac": frac})

    print(f"{'N':>6} {'vanilla (ms)':>14} {'SWFA (ms)':>12} {'speedup':>9} "
          f"{'vanilla FLOPs (M)':>18} {'SWFA FLOPs (M)':>16} {'win frac':>10}")
    print("-" * 95)
    for r in results:
        speedup = r["t_vanilla_ms"] / r["t_swfa_ms"] if r["t_swfa_ms"] > 0 else float("inf")
        print(f"{r['N']:>6} {r['t_vanilla_ms']:>14.2f} {r['t_swfa_ms']:>12.2f} "
              f"{speedup:>8.2f}x {r['flops_vanilla_M']:>18.2f} {r['flops_swfa_M']:>16.2f} "
              f"{r['window_frac']:>10.3f}")

    # Plot.
    out = ROOT / "plots"
    out.mkdir(exist_ok=True)
    xs = [r["N"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(xs, [r["t_vanilla_ms"] for r in results], "o-", color="#aaaaaa",
             label="Vanilla causal (PyTorch MHA)", linewidth=2)
    ax1.plot(xs, [r["t_swfa_ms"] for r in results], "o-", color="#64ffda",
             label="SWFA (unfused Python)", linewidth=2)
    ax1.set_xlabel("Sequence length N")
    ax1.set_ylabel("Forward-pass median latency (ms)")
    ax1.set_title("Wall-clock (unfused kernel — no Flash-Attention fusion)")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(xs, [r["flops_vanilla_M"] for r in results], "o-", color="#aaaaaa",
             label="Vanilla causal", linewidth=2)
    ax2.plot(xs, [r["flops_swfa_M"] for r in results], "o-", color="#64ffda",
             label="SWFA (window=2s)", linewidth=2)
    ax2.set_xlabel("Sequence length N")
    ax2.set_ylabel("Theoretical FLOPs (millions)")
    ax2.set_title("Compute: Vanilla O(N²) vs SWFA O(N·K_eff)")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "speed_benchmark.png", dpi=140)
    plt.close(fig)
    print(f"\nPlot → {out}/speed_benchmark.png")


if __name__ == "__main__":
    main()
