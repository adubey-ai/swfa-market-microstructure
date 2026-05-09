"""Baseline models for fair comparison: DeepLOB-style CNN-LSTM and vanilla transformer."""
from __future__ import annotations

import torch
import torch.nn as nn


class DeepLOBLike(nn.Module):
    """Compact CNN-LSTM baseline in the spirit of Zhang et al. (2019).

    Not a faithful reproduction (we operate on feature sequences, not raw
    order-book snapshots), but matches the spirit and parameter count.
    """

    def __init__(self, num_features: int = 6, hidden: int = 32, num_classes: int = 3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(num_features, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, features: torch.Tensor, timestamps_s: torch.Tensor) -> torch.Tensor:
        x = features.transpose(1, 2)        # (B, F, N)
        x = self.cnn(x).transpose(1, 2)     # (B, N, H)
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


class VanillaTransformerLOB(nn.Module):
    """Standard TransformerEncoder for the same task — no time decay, no windowing."""

    def __init__(self, num_features: int = 6, dim: int = 64, num_heads: int = 4,
                 num_layers: int = 2, num_classes: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(num_features, dim)
        enc_layer = nn.TransformerEncoderLayer(dim, num_heads, dim_feedforward=4 * dim,
                                               batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, features: torch.Tensor, timestamps_s: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(features)
        # Causal mask for fair comparison with SWFA.
        N = x.size(1)
        causal = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
        x = self.encoder(x, mask=causal)
        return self.head(x[:, -1])
