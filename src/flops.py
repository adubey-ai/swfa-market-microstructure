"""Theoretical FLOP counting for attention variants.

For sequence length N, embedding dim D, H heads:

    Vanilla causal:   O(N^2 * D)  — every pair of tokens computes QK^T + softmax + V.
    SWFA with block-classification: tokens with physical time gap > 3τ_max
                      are skipped entirely, reducing the effective N^2 term to
                      N * K_eff(N) where K_eff(N) is the in-window neighbor count.

We count the dominant QK^T and AV matmuls; we don't count softmax or
projections since those are identical across variants.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FLOPReport:
    variant: str
    seq_len: int
    flops_qk: int    # query-key outer product
    flops_av: int    # attention-values
    flops_total: int


def vanilla_attention_flops(N: int, D: int, H: int) -> FLOPReport:
    per_head = D // H
    # QK^T  : H * N * N * per_head  multiply-adds (×2 for MAC → FLOPs)
    fqk = 2 * H * N * N * per_head
    # A V    : H * N * N * per_head
    fav = 2 * H * N * N * per_head
    return FLOPReport("Vanilla", N, fqk, fav, fqk + fav)


def swfa_attention_flops(N: int, D: int, H: int, in_window_frac: float) -> FLOPReport:
    """`in_window_frac` is the average fraction of past tokens inside 3τ window."""
    per_head = D // H
    eff = max(1.0, in_window_frac * N)   # avg in-window neighbors per query
    fqk = int(2 * H * N * eff * per_head)
    fav = int(2 * H * N * eff * per_head)
    return FLOPReport("SWFA", N, fqk, fav, fqk + fav)


def window_fraction(timestamps: np.ndarray, max_window_s: float) -> float:
    """Empirical in-window fraction from a real sequence."""
    N = len(timestamps)
    gaps = timestamps[:, None] - timestamps[None, :]
    # Causal only: lower triangle.
    causal = np.tril(np.ones_like(gaps), k=-1)
    mask = (gaps <= max_window_s) & (gaps >= 0) & causal.astype(bool)
    return mask.sum() / max(N * (N - 1) / 2, 1)
