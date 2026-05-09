"""Visualize SWFA attention patterns per head for a trained model."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src import RegimeSwitchLOBDataset, Config, LOBTransformer


def main():
    torch.manual_seed(0)
    cfg = Config(seq_len=48)

    # Quick train to get a meaningful pattern.
    from torch.utils.data import DataLoader
    model = LOBTransformer(num_features=cfg.num_features, dim=64, num_heads=4,
                           num_layers=2, max_distance_s=5.0)
    train_ds = RegimeSwitchLOBDataset(n=256, cfg=cfg, seed=42)
    loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lf = torch.nn.CrossEntropyLoss()
    for _ in range(5):
        for feats, t, y in loader:
            opt.zero_grad(set_to_none=True)
            lf(model(feats, t), y).backward()
            opt.step()
    model.eval()

    # Pick a sample from each regime.
    slow = None; medium = None; fast = None
    for idx in range(len(train_ds)):
        feats, t, label, tau = train_ds.sample_with_regime(idx)
        if slow is None and abs(tau - 1.8) < 0.01:
            slow = (feats, t, tau)
        elif medium is None and abs(tau - 0.4) < 0.01:
            medium = (feats, t, tau)
        elif fast is None and abs(tau - 0.08) < 0.01:
            fast = (feats, t, tau)
        if slow and medium and fast:
            break

    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    rows = [("fast τ=0.08", fast), ("medium τ=0.40", medium), ("slow τ=1.80", slow)]
    with torch.no_grad():
        for i, (regime_name, sample) in enumerate(rows):
            feats, t, _ = sample
            feats_t = torch.from_numpy(feats).unsqueeze(0)
            t_t = torch.from_numpy(t).unsqueeze(0)
            # Layer 0 attention.
            x = model.input_proj(feats_t)
            x_norm = model.blocks[0].norm1(x)
            _, attn = model.blocks[0].attn(x_norm, t_t, return_attn=True)
            attn = attn[0].cpu().numpy()  # (H, N, N)

            for h in range(4):
                ax = axes[i, h]
                ax.imshow(attn[h], cmap="magma", aspect="equal", vmin=0,
                          vmax=np.percentile(attn[h][attn[h] > 0], 95) if (attn[h] > 0).any() else 1)
                tau_h = float(model.blocks[0].attn.log_tau[h].exp().detach())
                ax.set_title(f"{regime_name}\nhead {h} (τ={tau_h:.2f}s)", fontsize=9)
                ax.set_xlabel("key index"); ax.set_ylabel("query")
                ax.tick_params(labelsize=7)

    fig.suptitle("SWFA attention patterns — layer 0", fontsize=12)
    fig.tight_layout()
    out = ROOT / "plots" / "attention_patterns.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
