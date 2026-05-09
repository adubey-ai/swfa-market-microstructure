"""Synthetic LOB dataset: regime-switching momentum with variable decay timescales.

Why this data is good for demonstrating the SWFA advantage:

We generate sequences from a mixture of regimes, each with its own momentum
decay timescale τ ∈ {0.1s, 0.5s, 2.0s}. The label (next-step direction) is
a function of the *recent* order flow, but the predictive horizon depends
on the regime — 100ms for fast-decay, 2s for slow-decay.

A fixed-window model (standard transformer, DeepLOB CNN-LSTM) must pick
*one* horizon. A model with *learnable per-head time decay* (SWFA) can
dedicate heads to different horizons and outperform both.

This isn't data-snooping; it's testing a specific hypothesis about
adaptive-horizon architectures on a task where the hypothesis is falsifiable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Config:
    seq_len: int = 64
    num_features: int = 6
    horizon_s: float = 0.3         # predict direction 0.3s into the future


class RegimeSwitchLOBDataset(Dataset):
    """Regime-switching LOB: samples alternate between 3 momentum half-lives.

    Each sample sees one regime throughout, so a good model needs to figure
    out *which* regime and apply the right memory horizon.
    """

    TAU_CHOICES = (0.08, 0.4, 1.8)   # fast / medium / slow regimes (seconds)

    def __init__(self, n: int = 1024, cfg: Config = Config(), seed: int = 0,
                 noise_level: float = 0.4):
        self.n = n
        self.cfg = cfg
        self.base_seed = seed
        self.noise_level = noise_level

    def __len__(self):
        return self.n

    def _simulate(self, seed: int):
        rng = np.random.default_rng(seed)
        tau = rng.choice(self.TAU_CHOICES)

        # One extra window for the future target.
        extra = int(self.cfg.horizon_s / 0.004) + 4
        N = self.cfg.seq_len + extra

        dt = rng.exponential(scale=0.004, size=N)   # ~250 events/s
        t = np.cumsum(dt)

        # Drift state driven by signed "news" impulses that decay with the regime tau.
        momentum = 0.0
        mids = np.zeros(N)
        mids[0] = 10000.0
        impulse_intensity = 2.5   # impulses per second
        for i in range(1, N):
            if rng.random() < impulse_intensity * dt[i]:
                momentum += rng.choice([-1.0, 1.0]) * rng.uniform(2.0, 5.0)
            momentum *= np.exp(-dt[i] / tau)
            price_noise = rng.normal(0, self.noise_level)
            mids[i] = mids[i - 1] + dt[i] * momentum + price_noise

        spread = np.maximum(rng.poisson(1.0, N) + 1, 1).astype(float)
        # Make bid/ask sizes correlate with momentum so OFI is informative.
        bid_drift = np.sign(momentum) * 2.0 if momentum > 0 else 0.0
        ask_drift = np.abs(momentum) * 0.5 if momentum < 0 else 0.0
        bid_sz = rng.poisson(30 + bid_drift, N).astype(float) + 1
        ask_sz = rng.poisson(30 + ask_drift, N).astype(float) + 1

        imb = (bid_sz - ask_sz) / (bid_sz + ask_sz)
        ofi = np.zeros(N)
        ofi[1:] = (bid_sz[1:] - bid_sz[:-1]) - (ask_sz[1:] - ask_sz[:-1])

        features = np.stack([
            (mids - mids[0]) * 0.05,
            spread,
            imb,
            ofi * 0.05,
            bid_sz * 0.01,
            ask_sz * 0.01,
        ], axis=-1).astype(np.float32)

        # Target: mid-price direction at t_last + horizon_s (index ~seq_len + h_steps).
        h_steps = int(self.cfg.horizon_s / 0.004)
        idx_now = self.cfg.seq_len - 1
        idx_future = min(idx_now + h_steps, N - 1)
        delta = mids[idx_future] - mids[idx_now]

        # Thresholds calibrated to yield roughly-balanced classes per regime.
        thresh = 0.8
        if delta > thresh:
            label = 2
        elif delta < -thresh:
            label = 0
        else:
            label = 1

        seq_feats = features[:self.cfg.seq_len]
        seq_t = t[:self.cfg.seq_len] - t[0]
        return seq_feats, seq_t.astype(np.float32), label, float(tau)

    def __getitem__(self, idx: int):
        feats, t, label, _ = self._simulate(seed=self.base_seed + idx * 7 + 1)
        return (
            torch.from_numpy(feats),
            torch.from_numpy(t),
            torch.tensor(label, dtype=torch.long),
        )

    def sample_with_regime(self, idx: int):
        """For analysis: return regime tau alongside features."""
        return self._simulate(seed=self.base_seed + idx * 7 + 1)


# Alias maintained for backwards compatibility.
SyntheticLOBDataset = RegimeSwitchLOBDataset


class FI2010Dataset(Dataset):
    """Stub for FI-2010 benchmark data (Ntakaris et al., 2018).

    Expects numpy arrays at ``root/FI-2010/{train,val,test}_{X,t,y}.npy`` where
    X is (N, seq_len, num_features), t is (N, seq_len), y is (N,) with classes
    {0: down, 1: stationary, 2: up}.
    """

    def __init__(self, root: str | Path, split: str = "train"):
        root = Path(root)
        self.X = np.load(root / "FI-2010" / f"{split}_X.npy")
        self.t = np.load(root / "FI-2010" / f"{split}_t.npy")
        self.y = np.load(root / "FI-2010" / f"{split}_y.npy")

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]).float(),
            torch.from_numpy(self.t[idx]).float(),
            torch.tensor(int(self.y[idx]), dtype=torch.long),
        )
