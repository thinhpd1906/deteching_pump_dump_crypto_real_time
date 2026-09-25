"""Smoke tests for src/models/deep_baselines.py (B8 CNN-BiLSTM, B7 LSTM-AE).

Pure forward-pass shape/NaN checks -- no training loop invoked.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.deep_baselines import CNNBiLSTM, LSTMAE  # noqa: E402

torch.manual_seed(0)

B, T, FEAT = 4, 120, 10   # locked window spec: T=120 bars, F=10 features


def test_cnn_bilstm_forward_shape():
    model = CNNBiLSTM(d_in=FEAT, hidden=64, dropout=0.2)
    x = torch.randn(B, T, FEAT)
    logit, attn = model(x)

    assert logit.shape == (B,)
    assert torch.isfinite(logit).all()

    assert attn.shape == (B, T)
    assert torch.isfinite(attn).all()
    # attention pooling weights should be a valid distribution over time
    assert torch.allclose(attn.sum(-1), torch.ones(B), atol=1e-4)


def test_lstm_ae_forward_and_score_shape():
    model = LSTMAE(d_in=FEAT, hidden=64, latent=32)
    x = torch.randn(B, T, FEAT)
    x_hat = model(x)
    assert x_hat.shape == (B, T, FEAT)
    assert torch.isfinite(x_hat).all()

    score = model.score(x)
    assert score.shape == (B,)
    assert torch.isfinite(score).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
