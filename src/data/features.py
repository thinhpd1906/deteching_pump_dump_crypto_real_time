"""M4a — Shared feature engineering: ticks -> 5s bars -> features -> windows.

CRITICAL DESIGN RULE
--------------------
This module is imported by BOTH the offline dataset builder AND the real-time
inference service. Online and offline features are therefore identical BY
CONSTRUCTION — there is no training/serving skew. Never re-implement these
functions anywhere else.

Feature set (10 per bar), grounded in the P&D literature:
  ret              log return of close                    (price move)
  hl_range         (high - low) / close                   (intra-bar violence)
  vwap_dev         (close - vwap) / vwap                  (price vs volume centre)
  taker_buy_ratio  taker-buy volume / volume              (buy pressure; 0.5=balanced)
  log_vol          log1p(volume)                          (size)
  log_ntr          log1p(n_trades)                        (trade velocity)
  z_vol            z-score of log_vol vs 1h rolling       ("abnormal for THIS coin")
  z_ntr            z-score of log_ntr vs 1h rolling
  z_ret            ret / rolling sigma                    (move size in sigma units)
  roll_sigma       5-min return volatility                (regime context)

Note on terminology: Binance aggTrades are lightly aggregated, so we say
"per-trade granularity", never "raw ticks" (microstructure reviewers flag it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "ret", "hl_range", "vwap_dev", "taker_buy_ratio",
    "log_vol", "log_ntr", "z_vol", "z_ntr", "z_ret", "roll_sigma",
]


# --------------------------------------------------------------------- bars
def trades_to_bars(trades: pd.DataFrame, bar_seconds: int = 5) -> pd.DataFrame:
    """Aggregate per-trade records into fixed bars on a continuous time grid.

    Empty bars ARE imputed (close forward-filled, OHLC collapsed to close,
    volume/n_trades = 0). This is the imputation step of the pipeline — say so
    explicitly in the paper.

    Args:
        trades: columns (agg_id, price, qty, time[ms], is_buyer_maker)
    Returns:
        DataFrame indexed by UTC bar start, columns
        [open, high, low, close, volume, n_trades, taker_buy_vol, quote_vol, vwap]
    """
    df = trades.copy()
    df["ts"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()

    df["quote"] = df["price"] * df["qty"]
    # is_buyer_maker == False  =>  the taker was the BUYER (aggressive buy)
    df["tb_qty"] = np.where(~df["is_buyer_maker"].astype(bool), df["qty"], 0.0)

    g = df.resample(f"{bar_seconds}s")
    bars = pd.DataFrame({
        "open":          g["price"].first(),
        "high":          g["price"].max(),
        "low":           g["price"].min(),
        "close":         g["price"].last(),
        "volume":        g["qty"].sum(),
        "n_trades":      g["price"].count(),
        "taker_buy_vol": g["tb_qty"].sum(),
        "quote_vol":     g["quote"].sum(),
    })

    bars["close"] = bars["close"].ffill()
    bars = bars.dropna(subset=["close"])              # drop leading empties
    for c in ("open", "high", "low"):
        bars[c] = bars[c].fillna(bars["close"])
    for c in ("volume", "n_trades", "taker_buy_vol", "quote_vol"):
        bars[c] = bars[c].fillna(0.0)

    bars["vwap"] = np.where(
        bars["volume"] > 0,
        bars["quote_vol"] / bars["volume"].replace(0, np.nan),
        bars["close"],
    )
    bars["vwap"] = bars["vwap"].fillna(bars["close"])
    return bars


def normalize_bars(bars: pd.DataFrame, bar_seconds: int = 5) -> pd.DataFrame:
    """Re-grid streamed bars onto a full continuous grid (serving path).

    The streaming job may skip empty windows; training never does. This makes
    the two paths agree.
    """
    idx = pd.date_range(bars.index.min(), bars.index.max(),
                        freq=f"{bar_seconds}s", tz="UTC")
    bars = bars.reindex(idx)
    bars["close"] = bars["close"].ffill()
    for c in ("open", "high", "low", "vwap"):
        if c in bars:
            bars[c] = bars[c].fillna(bars["close"])
    for c in ("volume", "n_trades", "taker_buy_vol", "quote_vol"):
        if c in bars:
            bars[c] = bars[c].fillna(0.0)
    return bars.dropna(subset=["close"])


# ----------------------------------------------------------------- features
def bars_to_features(bars: pd.DataFrame, z_window: int = 720,
                     vol_window: int = 60) -> pd.DataFrame:
    """Bars -> the 10 model features. Returns float32, no NaN."""
    f = pd.DataFrame(index=bars.index)
    close = bars["close"].astype(float)
    ret = np.log(close).diff()

    f["ret"] = ret
    f["hl_range"] = (bars["high"] - bars["low"]) / close
    f["vwap_dev"] = (close - bars["vwap"]) / bars["vwap"]
    f["taker_buy_ratio"] = np.where(
        bars["volume"] > 0, bars["taker_buy_vol"] / bars["volume"], 0.5)
    f["log_vol"] = np.log1p(bars["volume"])
    f["log_ntr"] = np.log1p(bars["n_trades"])

    def _z(s: pd.Series) -> pd.Series:
        minp = max(z_window // 4, 20)
        mu = s.rolling(z_window, min_periods=minp).mean()
        sd = s.rolling(z_window, min_periods=minp).std()
        return (s - mu) / (sd + 1e-9)

    f["z_vol"] = _z(f["log_vol"])
    f["z_ntr"] = _z(f["log_ntr"])

    sigma = ret.rolling(vol_window, min_periods=10).std()
    f["z_ret"] = ret / (sigma + 1e-9)
    f["roll_sigma"] = sigma

    f = f.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return f[FEATURES].astype(np.float32)


# ------------------------------------------------------------------ windows
def make_windows(feats: pd.DataFrame, window_bars: int,
                 stride_bars: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows -> X [N, T, F] and window END times (ns since epoch)."""
    vals = feats.to_numpy(dtype=np.float32)
    # pandas>=2 may hold ms-resolution index; force ns before .asi8
    times = feats.index.as_unit("ns").asi8

    T = window_bars
    if len(vals) < T:
        return (np.empty((0, T, vals.shape[1]), np.float32),
                np.empty((0,), np.int64))

    ends = np.arange(T - 1, len(vals), stride_bars)
    X = np.stack([vals[e - T + 1: e + 1] for e in ends])
    return X, times[ends]


def label_windows(end_times_ns: np.ndarray, pump_times: pd.Series, *,
                  pre_minutes: int, during_minutes: int,
                  buffer_hours: int) -> tuple[np.ndarray, np.ndarray]:
    """Label windows.

    y = 1        if window END is inside [t_pump - pre, t_pump + during]
    mask_out     True for windows within +-buffer_hours of a pump but NOT
                 labelled positive -> these are ambiguous and are DROPPED
                 (never used as negatives).
    """
    ends = pd.to_datetime(end_times_ns, utc=True)
    y = np.zeros(len(ends), dtype=np.int64)
    near = np.zeros(len(ends), dtype=bool)

    for t in pump_times:
        pos = (ends >= t - pd.Timedelta(minutes=pre_minutes)) & \
              (ends <= t + pd.Timedelta(minutes=during_minutes))
        buf = (ends >= t - pd.Timedelta(hours=buffer_hours)) & \
              (ends <= t + pd.Timedelta(hours=buffer_hours))
        y |= np.asarray(pos)
        near |= np.asarray(buf)

    mask_out = near & (y == 0)
    return y, mask_out
