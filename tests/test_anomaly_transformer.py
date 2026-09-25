"""Unit tests for src/models/anomaly_transformer.py -- architecture verification only.

NO training loop is invoked here (no pretrain/finetune scripts). These are pure
forward/backward smoke tests on random tensors, run with:

    python -m pytest tests/test_anomaly_transformer.py -v

Spec under test (locked design, config/config.yaml -> model:):
    d_model=64, n_heads=4, n_layers=3, dropout=0.1, k=3.0 (lambda_k)
Window spec (config/config.yaml -> windows:):
    T = window_bars = 120, F = 10 microstructure features
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.anomaly_transformer import (  # noqa: E402
    AnomalyTransformer,
    ATClassifier,
    association_discrepancy,
    anomaly_score,
    minimax_losses,
    sym_kl,
)

torch.manual_seed(0)

B, T, FEAT = 4, 120, 10          # locked spec: T=120 bars, F=10 features
D_MODEL, N_HEADS, N_LAYERS, K = 64, 4, 3, 3.0   # locked spec: config/config.yaml


def _make_model():
    return AnomalyTransformer(
        win_size=T, d_in=FEAT, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, dropout=0.1,
    )


def test_series_association_rows_sum_to_one():
    model = _make_model()
    x = torch.randn(B, T, FEAT)
    _, series_l, _ = model(x)
    for s in series_l:
        row_sums = s.sum(-1)                       # (B, H, L)
        assert row_sums.shape == (B, N_HEADS, T)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


def test_prior_association_rows_sum_to_one():
    model = _make_model()
    x = torch.randn(B, T, FEAT)
    _, _, prior_l = model(x)
    for p in prior_l:
        row_sums = p.sum(-1)                       # (B, H, L)
        assert row_sums.shape == (B, N_HEADS, T)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


def test_association_discrepancy_shape():
    model = _make_model()
    x = torch.randn(B, T, FEAT)
    _, series_l, prior_l = model(x)
    ad = association_discrepancy(series_l, prior_l)
    assert ad.shape == (B, T)
    assert torch.isfinite(ad).all()


def test_training_step_decreases_reconstruction_loss():
    """Fixed batch, a handful of minimax steps: rec (MSE) should trend down.

    A single step is noisy (two backward passes with opposing signs on the
    same encoder), so we run 10 steps on ONE fixed batch and compare the
    first-step rec loss to the last-step rec loss.
    """
    torch.manual_seed(0)
    model = _make_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(B, T, FEAT)

    rec_losses = []
    for _ in range(10):
        x_hat, series_l, prior_l = model(x)
        loss_s, loss_p, rec = minimax_losses(x, x_hat, series_l, prior_l, k=K)
        rec_losses.append(rec.item())

        opt.zero_grad()
        loss_s.backward(retain_graph=True)
        loss_p.backward()
        opt.step()

    assert rec_losses[-1] < rec_losses[0], (
        f"rec loss did not decrease over 10 steps: {rec_losses[0]:.6f} -> "
        f"{rec_losses[-1]:.6f} (full trace: {rec_losses})"
    )
    assert all(torch.isfinite(torch.tensor(rec_losses)))


def test_anomaly_score_shape_and_finite():
    model = _make_model()
    x = torch.randn(B, T, FEAT)
    score = anomaly_score(model, x)
    assert score.shape == (B,)
    assert torch.isfinite(score).all()
    assert not torch.isnan(score).any()


# ------------------------------------------------------- extra coverage
def test_at_classifier_forward_shape():
    encoder = _make_model()
    clf = ATClassifier(encoder, d_model=D_MODEL)
    x = torch.randn(B, T, FEAT)
    logit, series_l, prior_l = clf(x)
    assert logit.shape == (B,)
    assert torch.isfinite(logit).all()
    assert len(series_l) == N_LAYERS
    assert len(prior_l) == N_LAYERS


def test_sym_kl_symmetric_and_zero_for_identical_dists():
    p = torch.softmax(torch.randn(B, N_HEADS, T, T), dim=-1)
    assert torch.allclose(sym_kl(p, p), torch.zeros(B, T), atol=1e-4)
    q = torch.softmax(torch.randn(B, N_HEADS, T, T), dim=-1)
    assert torch.allclose(sym_kl(p, q), sym_kl(q, p), atol=1e-5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
