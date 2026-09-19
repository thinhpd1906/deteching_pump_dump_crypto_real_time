"""Deep baselines for Family 3 of the comparison table.

  B8  CNNBiLSTM  -- supervised deep model (Chadalapaka et al. 2022 style).
                    Also the FALLBACK HEADLINE MODEL if Gate 3 fails.
                    Its attention weights double as a free XAI figure.
  B7  LSTMAE     -- unsupervised autoencoder; score = reconstruction MSE.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CNNBiLSTM(nn.Module):
    """Conv1d x2 -> BiLSTM -> attention pooling -> logit.

    Returns (logit, attention_weights) so the same forward pass gives both the
    prediction and the explainability signal.
    """

    def __init__(self, d_in: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_in, hidden, kernel_size=5, padding=2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True,
                            bidirectional=True)
        self.attn = nn.Linear(2 * hidden, 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, 1))

    def forward(self, x):                       # x: (B, T, F)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)   # (B, T, H)
        h, _ = self.lstm(h)                                # (B, T, 2H)
        w = torch.softmax(self.attn(h).squeeze(-1), dim=-1)  # (B, T)
        pooled = (w.unsqueeze(-1) * h).sum(1)              # (B, 2H)
        return self.head(pooled).squeeze(-1), w


class LSTMAE(nn.Module):
    """Seq2seq LSTM autoencoder. Anomaly score = reconstruction MSE."""

    def __init__(self, d_in: int, hidden: int = 64, latent: int = 32):
        super().__init__()
        self.enc = nn.LSTM(d_in, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, d_in)

    def forward(self, x):                       # x: (B, T, F)
        _, (h, _) = self.enc(x)
        z = self.to_latent(h[-1])               # (B, latent)
        h0 = self.from_latent(z)                # (B, hidden)
        dec_in = h0.unsqueeze(1).repeat(1, x.size(1), 1)
        d, _ = self.dec(dec_in)
        return self.out(d)

    @torch.no_grad()
    def score(self, x):
        self.eval()
        return ((self(x) - x) ** 2).mean(dim=(1, 2))
