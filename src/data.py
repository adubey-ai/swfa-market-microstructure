"""Synthetic LOB dataset + FI-2010 loader stub.

Synthetic data is mean-reverting with momentum bursts — short-horizon price
direction is predictable from recent order-flow features, and the predictive
horizon varies across regimes (which is what the learnable time-decay should
discover).

FI-2010 loader (below) works when the dataset is downloaded at ``data/FI-2010/``.
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
    horizon_s: float = 0.5   # predict direction 0.5s into the future


class SyntheticLOBDataset(Dataset):
    """Ornstein-Uhlenbeck mid with momentum bursts; OFI and book imbalance features."""

    def __init__(self, n: int = 1024, cfg: Config = Config(), seed: int = 0,
                 burst_rate: float = 0.15):
        self.n = n
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.burst_rate = burst_rate

    def __len__(self):
        return self.n

    def _simulate(self, seed: int):
        rng = np.random.default_rng(seed)
        N = self.cfg.seq_len + 32
        dt = rng.exponential(scale=0.002, size=N)   # event gaps ~2ms avg
        t = np.cumsum(dt)
        mid = np.zeros(N)
        mid[0] = 10000.0

        momentum_mode = 0.0  # drift state
        for i in range(1, N):
            if rng.random() < self.burst_rate * dt[i]:  # enter burst
                momentum_mode = rng.choice([-1.0, 1.0]) * rng.uniform(5.0, 15.0)
            momentum_mode *= np.exp(-dt[i] / 0.3)       # burst decays with ~0.3s half-life
            mean_rev = -0.5 * (mid[i - 1] - 10000.0)
            shock = rng.normal(0, 1.0)
            mid[i] = mid[i - 1] + dt[i] * (mean_rev + momentum_mode) + shock

        spread = np.maximum(rng.poisson(1.0, N) + 1, 1).astype(float)
        bid = mid - spread / 2.0
        ask = mid + spread / 2.0
        bid_sz = rng.poisson(30, N).astype(float) + 1
        ask_sz = rng.poisson(30, N).astype(float) + 1

        # Book imbalance.
        imb = (bid_sz - ask_sz) / (bid_sz + ask_sz)
        # Order-flow imbalance (diff of sizes).
        ofi = np.zeros(N)
        ofi[1:] = (bid_sz[1:] - bid_sz[:-1]) - (ask_sz[1:] - ask_sz[:-1])

        features = np.stack([
            (mid - mid[0]) * 0.1,
            spread,
            imb,
            ofi * 0.1,
            bid_sz * 0.01,
            ask_sz * 0.01,
        ], axis=-1).astype(np.float32)

        seq_feats = features[-self.cfg.seq_len:]
        seq_t = t[-self.cfg.seq_len:] - t[-self.cfg.seq_len]

        # Target: mid-price direction at t_last + horizon_s.
        t_last = t[-1]
        horizon_t = t_last + self.cfg.horizon_s
        # Extend the walk to the horizon.
        steps = max(1, int(self.cfg.horizon_s / 0.002))
        future_mid = mid[-1]
        mom = momentum_mode
        for _ in range(steps):
            dt_ = rng.exponential(scale=0.002)
            mom *= np.exp(-dt_ / 0.3)
            future_mid += dt_ * (-0.5 * (future_mid - 10000.0) + mom) + rng.normal(0, 1.0)

        delta = future_mid - mid[-1]
        if delta > 1.0:
            label = 2
        elif delta < -1.0:
            label = 0
        else:
            label = 1

        return seq_feats, seq_t.astype(np.float32), label

    def __getitem__(self, idx: int):
        feats, t, label = self._simulate(seed=idx * 7 + 1)
        return (
            torch.from_numpy(feats),
            torch.from_numpy(t),
            torch.tensor(label, dtype=torch.long),
        )


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
