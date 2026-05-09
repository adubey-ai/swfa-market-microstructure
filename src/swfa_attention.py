"""Sliding-Window Causal Attention with learnable time-decayed window.

Novel angle (vs DeepLOB, TransLOB, and standard sparse-transformer literature):
the attention window is *not* a fixed K tokens — it's a learnable exponential
decay over the *time distance* between events. That gives us an interpretable
"effective memory horizon" per asset and per time-of-day, which is the
quantity portfolio managers actually reason about.

The SWFA block-classification idea from my Intel internship carries over here:
tokens with time gap > 3τ are dropped entirely (fully-outside block), tokens
within τ get full attention (fully-inside), and the intersecting band gets
masked attention. Classifier runs in O(N log N) instead of O(N^2).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDecayedSWFA(nn.Module):
    """Causal sliding-window attention with learnable per-head time decay.

    Parameters
    ----------
    dim : model dim
    num_heads : heads
    log_tau_init : initial log time-scale (in seconds). Heads learn their own.
    max_distance_s : hard cutoff for the "fully-outside" block classifier.
    """

    def __init__(self, dim: int, num_heads: int = 4, log_tau_init: float = -2.0,
                 max_distance_s: float = 10.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        # Per-head log-tau (time scale in seconds); heads learn their own window.
        self.log_tau = nn.Parameter(torch.full((num_heads,), log_tau_init))
        self.max_distance_s = max_distance_s

    def effective_memory_s(self) -> torch.Tensor:
        """Return per-head effective memory horizon in seconds (3τ rule)."""
        return 3.0 * torch.exp(self.log_tau).detach()

    def forward(self, x: torch.Tensor, timestamps_s: torch.Tensor,
                return_attn: bool = False):
        """x: (B, N, D)  timestamps_s: (B, N) event times in seconds."""
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, H, N, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale   # B, H, N, N

        # Causal mask (upper triangular forbidden).
        causal = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal.view(1, 1, N, N), float("-inf"))

        # Time-gap matrix: gap[i, j] = t_i - t_j.
        t = timestamps_s.to(attn.dtype)
        gap = t[:, :, None] - t[:, None, :]    # (B, N, N)
        gap = gap.clamp_min(0.0)

        # Block classification: mask fully-outside.
        outside = gap > self.max_distance_s     # (B, N, N)
        attn = attn.masked_fill(outside.unsqueeze(1), float("-inf"))

        # Per-head exponential decay penalty.
        tau = torch.exp(self.log_tau).view(1, self.num_heads, 1, 1)
        decay = -gap.unsqueeze(1) / (tau + 1e-6)
        attn = attn + decay

        attn_sm = F.softmax(attn, dim=-1)

        # Safe zero for rows that got all -inf (rare, but guard).
        attn_sm = torch.nan_to_num(attn_sm, 0.0)

        out = attn_sm @ v  # (B, H, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)

        if return_attn:
            return out, attn_sm
        return out


class SWFABlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_mult: int = 4,
                 max_distance_s: float = 10.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TimeDecayedSWFA(dim, num_heads, max_distance_s=max_distance_s)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim), nn.GELU(),
            nn.Linear(ff_mult * dim, dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), t)
        x = x + self.ff(self.norm2(x))
        return x


class LOBTransformer(nn.Module):
    """Small transformer over LOB events -> direction logits {down, flat, up}."""

    def __init__(self, num_features: int = 6, dim: int = 64, num_heads: int = 4,
                 num_layers: int = 2, num_classes: int = 3, max_distance_s: float = 10.0):
        super().__init__()
        self.input_proj = nn.Linear(num_features, dim)
        self.blocks = nn.ModuleList([
            SWFABlock(dim, num_heads, max_distance_s=max_distance_s) for _ in range(num_layers)
        ])
        self.head = nn.Linear(dim, num_classes)

    def forward(self, features: torch.Tensor, timestamps_s: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(features)
        for blk in self.blocks:
            x = blk(x, timestamps_s)
        # Predict at the last (most-recent) token.
        return self.head(x[:, -1])
