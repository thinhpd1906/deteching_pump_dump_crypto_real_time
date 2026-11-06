"""A5: serving path (StreamingCleaner -> BarAggregator -> OnlineFeaturizer)
reproduces the training path (clean_ticks -> trades_to_bars ->
bars_to_features -> scaler) on the same trades. Synthetic, no Kafka."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from verify_parity import compare, verdict  # noqa: E402

from src.cleaning.filters import StreamingCleaner, clean_ticks  # noqa: E402
from src.pipeline.python_bars import BarAggregator  # noqa: E402


def _trades(n_bars=1100, seed=0, spikes=30):
    """~1.5 h of trades, irregular arrival, some quiet (empty) bars."""
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.exponential(700, size=int(n_bars * 5000 / 700))).astype(np.int64)
    t = t[t < n_bars * 5000] + 1_700_000_000_000
    quiet = (t // 60_000) % 17 == 3                    # one empty minute in 17
    t = t[~quiet]
    n = len(t)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 2e-4, n)))
    idx = rng.choice(np.arange(min(3000, n // 3), n - 1), spikes, replace=False)
    p[idx] *= np.where(rng.random(spikes) < 0.5, 1.03, 0.97)
    return pd.DataFrame({"agg_id": np.arange(n), "price": p,
                         "qty": rng.exponential(1.0, n), "time": t,
                         "is_buyer_maker": rng.random(n) < 0.5})


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(open(os.path.join(ROOT, "config", "config.yaml")))


def test_bar_aggregator_matches_trades_to_bars(cfg):
    from src.data.features import trades_to_bars
    df = _trades(300)
    agg = BarAggregator(5, "none", cfg["cleaning"], 1000)
    out = []
    for r in df.itertuples(index=False):
        out += agg.due(int(r.time))
        out += agg.add({"s": "X", "a": r.agg_id, "p": r.price, "q": r.qty,
                        "T": int(r.time), "m": bool(r.is_buyer_maker)})
    out += agg.flush_all()
    on = pd.DataFrame(out).set_index(pd.DatetimeIndex(
        [pd.Timestamp(b["bar_time"]) for b in out]))
    off = trades_to_bars(df, 5)
    off = off[off["n_trades"] > 0]
    assert len(on) == len(off) and agg.n_late == 0
    for c in ("open", "high", "low", "close", "volume", "n_trades",
              "taker_buy_vol", "quote_vol"):
        np.testing.assert_allclose(on[c].to_numpy(float), off[c].to_numpy(float),
                                   rtol=1e-12, err_msg=c)


@pytest.mark.parametrize("mode", ["none", "adaptive"])
def test_online_windows_match_offline(cfg, mode):
    F = 10
    mean, std = np.zeros(F, np.float32), np.ones(F, np.float32)
    r = compare(_trades(), cfg, mode, mean, std, 120)
    assert r["windows_compared"] > 100, r
    assert verdict(r), r                     # serving code path is exact
    assert r["path_max_abs_diff"] < 1e-4
    if mode == "none":
        assert r["windows_over_tol"] == 0


def test_streaming_cleaner_agrees_at_realistic_spike_rate():
    """Spikes at ~0.05% of ticks (training removal rate is 0.06%), after a
    warm-up: the causal filter makes the same decisions as the batch one.
    (With dense spikes the batch filter's whole-file regime median, which sees
    the future, diverges; verify_parity.py reports that residual skew.)"""
    rng = np.random.default_rng(1)
    n = 40_000
    p = 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, n)))
    idx = np.sort(rng.choice(np.arange(5000, n - 1), 20, replace=False))
    p[idx] *= np.where(rng.random(20) < 0.5, 1.02, 0.98)
    df = pd.DataFrame({"agg_id": np.arange(n), "price": p, "qty": 1.0,
                       "time": np.arange(n) * 50, "is_buyer_maker": False})
    batch = set(df.agg_id) - set(clean_ticks(df, "adaptive").agg_id)
    sc = StreamingCleaner("adaptive")
    kept = []
    for r in df.itertuples(index=False):
        kept += [t["a"] for t in sc.push({"a": r.agg_id, "p": r.price, "T": r.time})]
    kept += [t["a"] for t in sc.flush()]
    online = set(df.agg_id) - set(kept)
    assert len(batch) >= 10
    assert len(batch & online) >= 0.9 * len(batch)
    assert len(online - batch) <= 1
