"""Anomaly Transformer (Xu et al., ICLR 2022) -- the main thesis model.

Core idea
---------
Each layer computes TWO attention distributions per time step:
  series S : standard softmax self-attention        (data-driven, can be global)
  prior  P : learnable Gaussian kernel over |i-j|   (local by construction)

Normal points can build associations that match a local prior; anomalies
cannot -- they must attend broadly to reconstruct themselves. The KL
divergence between P and S ("association discrepancy", AssDis) therefore
separates them WITHOUT labels. Training is minimax: the prior chases the
series (+AssDis) while the series is pushed away (-AssDis), amplifying the
gap for anomalies.

WARNING: the official Anomaly Transformer repo evaluates with
point-adjustment (PA). Kim et al. (AAAI 2022) proved PA inflates even random
scores to SOTA. We reuse only the ARCHITECTURE, never their eval scripts.

Exports
-------
  AnomalyTransformer        encoder + reconstruction head (self-supervised)
  ATClassifier              encoder + pooled classifier head (fine-tuning)
  minimax_losses            the two-phase training objective
  association_discrepancy   per-timestep AssDis (also the XAI signal)
  anomaly_score             unsupervised window score
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------- utils
def _kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Row-wise KL(p || q) over the last dim.  (B,H,L,L) -> (B,H,L)."""
    p = p.clamp_min(1e-8)
    q = q.clamp_min(1e-8)
    return (p * (p.log() - q.log())).sum(-1)


def sym_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Symmetric KL, averaged over heads.  (B,H,L,L) x2 -> (B,L)."""
    return (_kl(p, q) + _kl(q, p)).mean(1)


# --------------------------------------------------------- anomaly attention
class AnomalyAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, win_size: int,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.h = n_heads
        self.dk = d_model // n_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.sigma = nn.Linear(d_model, n_heads)     # learnable prior width
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        idx = torch.arange(win_size, dtype=torch.float32)
        self.register_buffer("dist", (idx[None, :] - idx[:, None]).abs())

    def forward(self, x: torch.Tensor):
        B, L, _ = x.shape

        q = self.q(x).view(B, L, self.h, self.dk).transpose(1, 2)
        k = self.k(x).view(B, L, self.h, self.dk).transpose(1, 2)
        v = self.v(x).view(B, L, self.h, self.dk).transpose(1, 2)

        # series association: plain self-attention
        series = torch.softmax(
            q @ k.transpose(-1, -2) / math.sqrt(self.dk), dim=-1)   # (B,H,L,L)

        # prior association: Gaussian over temporal distance, learnable sigma
        sigma = F.softplus(self.sigma(x)) + 1e-4                    # (B,L,H)
        sigma = sigma.transpose(1, 2).unsqueeze(-1)                 # (B,H,L,1)
        d = self.dist[:L, :L][None, None]                           # (1,1,L,L)
        prior = torch.exp(-(d ** 2) / (2 * sigma ** 2))
        prior = prior / prior.sum(-1, keepdim=True)                 # row-normalize

        ctx = (self.drop(series) @ v).transpose(1, 2).reshape(B, L, -1)
        return self.out(ctx), series, prior


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, win_size, dropout=0.1):
        super().__init__()
        self.attn = AnomalyAttention(d_model, n_heads, win_size, dropout)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, series, prior = self.attn(self.n1(x))
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.n2(x)))
        return x, series, prior


class AnomalyTransformer(nn.Module):
    """Encoder + reconstruction head (used for self-supervised pretraining)."""

    def __init__(self, win_size: int, d_in: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.win = win_size
        self.embed = nn.Linear(d_in, d_model)

        pe = torch.zeros(win_size, d_model)
        pos = torch.arange(win_size).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, win_size, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.recon = nn.Linear(d_model, d_in)

    def encode(self, x):
        h = self.embed(x) + self.pe[:, : x.size(1)]
        series_l, prior_l = [], []
        for layer in self.layers:
            h, s, p = layer(h)
            series_l.append(s)
            prior_l.append(p)
        return self.norm(h), series_l, prior_l

    def forward(self, x):
        h, series_l, prior_l = self.encode(x)
        return self.recon(h), series_l, prior_l


class ATClassifier(nn.Module):
    """Fine-tuning head: mean-pool encoder states -> P(pump)."""

    def __init__(self, encoder: AnomalyTransformer, d_model: int,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        h, series_l, prior_l = self.encoder.encode(x)
        return self.head(h.mean(1)).squeeze(-1), series_l, prior_l


# ----------------------------------------------------------------- scoring
def association_discrepancy(series_l, prior_l) -> torch.Tensor:
    """Mean over layers of symmetric KL -> (B, L) per-timestep AssDis."""
    return torch.stack([sym_kl(p, s)
                        for s, p in zip(series_l, prior_l)]).mean(0)


@torch.no_grad()
def anomaly_score(model: AnomalyTransformer, x: torch.Tensor,
                  temperature: float = 50.0) -> torch.Tensor:
    """Unsupervised window score: softmax(-AssDis) * reconstruction error."""
    model.eval()
    x_hat, series_l, prior_l = model(x)
    rec = ((x - x_hat) ** 2).mean(-1)                  # (B, L)
    ad = association_discrepancy(series_l, prior_l)    # (B, L)
    w = torch.softmax(-ad * temperature, dim=-1)
    return (w * rec).sum(-1)                           # (B,)


def minimax_losses(x, x_hat, series_l, prior_l, k: float = 3.0):
    """Two-phase minimax objective (official scheme).

    loss_series : rec - k*AssDis(S, P.detach())   -> push S AWAY from prior
    loss_prior  : rec + k*AssDis(P, S.detach())   -> pull P TOWARD series
    Call .backward() on each in turn (retain_graph on the first).
    """
    rec = F.mse_loss(x_hat, x)
    s_loss = torch.stack([sym_kl(s, p.detach()).mean()
                          for s, p in zip(series_l, prior_l)]).mean()
    p_loss = torch.stack([sym_kl(p, s.detach()).mean()
                          for s, p in zip(series_l, prior_l)]).mean()
    return rec - k * s_loss, rec + k * p_loss, rec
