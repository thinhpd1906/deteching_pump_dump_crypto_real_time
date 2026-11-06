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


# --------------------------------------------------------------- streaming
class StreamingCleaner:
    """Causal, tick-by-tick version of `clean_ticks` for the serving path.

    Same rule as the batch filter: a tick is dropped if its log return exceeds
    the threshold AND the next tick undoes >= 70% of it. Deciding needs the
    next tick, so every tick is released with a ONE-TICK delay (`push`
    returns the previous tick once its fate is known; `flush` releases the
    held tick when no successor arrives in time, e.g. at a bar close, which is
    what the batch filter does for the last tick of a file).

    Differences from the batch filter, all bounded and reported by the parity
    check (scripts/verify_parity.py):
      * rolling tick sigma: identical (window `tick_sigma_window`, ddof=1,
        min_periods=50), except the first 49 ticks of a stream use the
        expanding std instead of the batch filter's backward fill (future);
      * regime median: batch uses the median sigma of the whole file
        (non-causal); online uses the median of the last `median_ticks`
        sigma values, refreshed every `median_every` ticks. regime is clipped
        to `regime_clip`, which bounds the effect.
    Input ticks must arrive in exchange order (Binance aggTrade streams do).
    """

    def __init__(self, mode: str = "adaptive", *, static_pct: float = 0.02,
                 k0: float = 6.0, tick_sigma_window: int = 500,
                 regime_clip: tuple[float, float] = (0.5, 3.0),
                 median_ticks: int = 200_000, median_every: int = 1000,
                 dedup_ids: int = 50_000):
        from collections import deque
        if mode not in ("none", "static", "adaptive"):
            raise ValueError(f"unknown cleaning mode: {mode}")
        self.mode, self.static_pct, self.k0 = mode, float(static_pct), float(k0)
        self.lo, self.hi = regime_clip
        self.min_periods = 50
        self._r = deque(maxlen=tick_sigma_window)
        self._sig_hist = deque(maxlen=median_ticks)
        self._median_every = median_every
        self._med = None
        self._since_med = 0
        self._ids: set = set()
        self._id_order = deque(maxlen=dedup_ids)
        self._prev_price = None
        self._pending = None            # (tick, r, thr)
        self.n_in = self.n_out = self.n_removed = self.n_dup = 0
        self.removed_ids: list = []      # rare (~0.06% of ticks); for audits

    # -- threshold for the newest return -----------------------------------
    def _thr(self) -> float:
        if self.mode == "static":
            return self.static_pct
        n = len(self._r)
        if n < 2:
            return float("inf")
        r = np.fromiter(self._r, float, n)
        sigma = max(float(r.std(ddof=1)), 1e-8)
        self._sig_hist.append(sigma)
        self._since_med += 1
        if self._med is None or self._since_med >= self._median_every \
                or len(self._sig_hist) < self._median_every:
            self._med = float(np.median(np.fromiter(self._sig_hist, float)))
            self._since_med = 0
        regime = min(max(sigma / max(self._med, 1e-9), self.lo), self.hi)
        return self.k0 * sigma * regime

    def push(self, tick: dict) -> list:
        """tick needs 'a' (agg id) and 'p' (price). Returns released ticks."""
        aid = tick["a"]
        if aid in self._ids:
            self.n_dup += 1
            return []
        if len(self._id_order) == self._id_order.maxlen:
            self._ids.discard(self._id_order[0])
        self._id_order.append(aid)
        self._ids.add(aid)
        self.n_in += 1
        if self.mode == "none":
            self.n_out += 1
            return [tick]

        p = max(float(tick["p"]), 1e-12)
        r = 0.0 if self._prev_price is None else float(np.log(p / self._prev_price))
        self._prev_price = p
        self._r.append(r)
        thr = self._thr()

        out = []
        if self._pending is not None:
            prev, r_prev, thr_prev = self._pending
            spike = abs(r_prev) > thr_prev
            reverts = abs(r_prev + r) < 0.3 * abs(r_prev)
            if spike and reverts:
                self.n_removed += 1
                self.removed_ids.append(prev["a"])
            else:
                out.append(prev)
                self.n_out += 1
        self._pending = (tick, r, thr)
        return out

    def flush(self) -> list:
        """Release the held tick (no successor seen: kept, like batch)."""
        if self._pending is None:
            return []
        t = self._pending[0]
        self._pending = None
        self.n_out += 1
        return [t]

    @property
    def pending_time(self):
        return None if self._pending is None else int(self._pending[0]["T"])
