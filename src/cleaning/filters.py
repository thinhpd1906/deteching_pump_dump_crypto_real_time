"""M3 — Tick-stream cleaning (Contribution 1 ablation axis).

Three modes, identical interface:

  none     : dedup + time-sort only (baseline "raw")
  static   : + drop isolated reverting moves > FIXED 2% threshold
  adaptive : + drop isolated reverting moves > ADAPTIVE threshold
             thr_i = k0 × rolling_tick_σ_i × regime_i
             regime_i = clip(σ_i / median(σ), lo, hi)

Design rationale (defend in the paper):
  * Pumps are large moves that DO NOT immediately revert → only single-tick
    spikes that revert (exchange glitches, fat fingers) are removed.
  * Adaptive: calm markets → tight filter; volatile markets → wide filter
    (don't accidentally filter real pump spikes). Static can't do both.

Usage:
    from src.cleaning.filters import clean_ticks, cleaning_stats
    cleaned = clean_ticks(df, mode="adaptive")
"""
import numpy as np
import pandas as pd


def _dedup_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate agg_id, stable-sort by (time, agg_id)."""
    df = df.drop_duplicates(subset=["agg_id"]).sort_values(
        ["time", "agg_id"], kind="mergesort"
    )
    return df.reset_index(drop=True)


def _reverting_spikes(price: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """Boolean mask: True for ticks that spike beyond `thr` AND revert ≥70%."""
    r = np.zeros_like(price)
    r[1:] = np.diff(np.log(np.maximum(price, 1e-12)))

    nxt = np.zeros_like(r)
    nxt[:-1] = r[1:]

    spike = np.abs(r) > thr
    reverts = np.abs(r + nxt) < 0.3 * np.abs(r)  # next tick undoes ≥70%
    return spike & reverts


def clean_ticks(
    df: pd.DataFrame,
    mode: str = "adaptive",
    *,
    static_pct: float = 0.02,
    k0: float = 6.0,
    tick_sigma_window: int = 500,
    regime_clip: tuple[float, float] = (0.5, 3.0),
) -> pd.DataFrame:
    """Clean one symbol's tick dataframe.

    Args:
        df: must have columns (agg_id, price, qty, time, is_buyer_maker)
        mode: "none", "static", or "adaptive"

    Returns:
        Cleaned dataframe (same schema, fewer rows).
    """
    df = _dedup_sort(df)
    if mode == "none" or len(df) < 10:
        return df

    price = df["price"].to_numpy(dtype=float)

    if mode == "static":
        thr = np.full(len(df), float(static_pct))

    elif mode == "adaptive":
        log_ret = pd.Series(np.concatenate([[0.0], np.diff(np.log(
            np.maximum(price, 1e-12)))]))
        sigma = (
            log_ret.rolling(tick_sigma_window, min_periods=50)
            .std()
            .bfill()
            .to_numpy()
        )
        sigma = np.maximum(sigma, 1e-8)
        med_sigma = np.median(sigma)
        regime = np.clip(sigma / max(med_sigma, 1e-9), *regime_clip)
        thr = k0 * sigma * regime

    else:
        raise ValueError(f"unknown cleaning mode: {mode}")

    bad = _reverting_spikes(price, thr)
    return df.loc[~bad].reset_index(drop=True)


def cleaning_stats(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    """Stats for reporting: how much cleaning changed the data."""
    n0, n1 = len(before), len(after)
    return {
        "ticks_in": n0,
        "ticks_out": n1,
        "removed": n0 - n1,
        "removed_pct": round(100.0 * (n0 - n1) / max(n0, 1), 4),
    }
