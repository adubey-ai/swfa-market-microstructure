"""Sanity tests for SWFA attention."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src import TimeDecayedSWFA, LOBTransformer, SyntheticLOBDataset, Config


def test_causal_mask():
    attn = TimeDecayedSWFA(dim=16, num_heads=2)
    B, N, D = 2, 8, 16
    x = torch.randn(B, N, D)
    t = torch.cumsum(torch.rand(B, N) * 0.01, dim=1)
    _, a = attn(x, t, return_attn=True)
    # Upper triangle of each attention map should be zero (causal).
    upper = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
    assert (a[..., upper].abs().max() < 1e-6), "causal mask violated"
    print("  causal mask enforced ✓")


def test_time_decay_clamps_far_tokens():
    attn = TimeDecayedSWFA(dim=16, num_heads=2, max_distance_s=0.05)
    x = torch.randn(1, 4, 16)
    t = torch.tensor([[0.0, 0.01, 0.1, 1.0]])   # last token is far from first two
    _, a = attn(x, t, return_attn=True)
    # Attention from token 3 back to tokens 0 and 1 (gap > 0.05) should be ~0.
    assert a[0, :, 3, 0].max() < 1e-5, "fully-outside token not masked"
    assert a[0, :, 3, 1].max() < 1e-5, "fully-outside token not masked"
    # Attention back to self must be > 0 (it's always in-window).
    assert a[0, :, 3, 3].min() > 0.0
    print("  fully-outside blocks masked; in-window attention retained ✓")


def test_shapes_and_effective_memory():
    model = LOBTransformer(num_features=6, dim=32, num_heads=4, num_layers=2)
    cfg = Config()
    ds = SyntheticLOBDataset(n=4, cfg=cfg, seed=0)
    feats, t, _ = ds[0]
    logits = model(feats.unsqueeze(0), t.unsqueeze(0))
    assert logits.shape == (1, 3)
    mem = model.blocks[0].attn.effective_memory_s()
    assert mem.shape == (4,)
    assert (mem > 0).all()
    print(f"  forward pass shape OK; initial effective memory = {mem.tolist()} ✓")


if __name__ == "__main__":
    torch.manual_seed(0)
    print("Running SWFA tests…")
    test_causal_mask()
    test_time_decay_clamps_far_tokens()
    test_shapes_and_effective_memory()
    print("All tests passed.")
