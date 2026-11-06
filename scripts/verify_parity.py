"""Train <-> serve parity check (verify-scaler-parity skill, task A5).

Feeds the SAME trades through both paths and compares the model input:

  offline (training): clean_ticks (batch) -> trades_to_bars -> bars_to_features
                      -> scaler.json -> window of T bars
  online  (serving) : StreamingCleaner -> BarAggregator (python_bars) ->
                      OnlineFeaturizer (inference_service) -> window of T bars

Every online window produced once the feature buffer is warm (full z_window
history) is compared with the offline window ending at the same bar.

Source of trades:
  --kafka   : replay the live raw-trades topic captured during the latency run
              (Kafka keeps it 24 h), per symbol
  (pytest)  : synthetic trades via compare()

Two separate questions:
  1. PATH parity (must be exact): bars + features + scaler online == offline.
     For `none` the offline path uses the same trades; for `adaptive` the
     offline path uses the trades minus the ticks the ONLINE filter removed,
     so only the bar/feature/scaler code is compared. Any window beyond
     tolerance is a FAIL.
  2. CLEANING-decision agreement (reported): online StreamingCleaner vs the
     batch clean_ticks used to build the training data. The batch filter's
     regime median is taken over the whole file (non-causal), so a causal
     filter cannot match it exactly; we report removed-tick precision/recall
     and the share of online windows whose offline (batch-cleaned) window
     differs, i.e. the residual train/serve skew that comes only from
     cleaning.

Usage:
    python scripts/verify_parity.py --kafka [--symbols BTCUSDT ETHUSDT]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.cleaning.filters import clean_ticks  # noqa: E402
from src.data.features import bars_to_features, trades_to_bars  # noqa: E402
from src.pipeline.inference_service import OnlineFeaturizer  # noqa: E402
from src.pipeline.python_bars import BarAggregator  # noqa: E402
from src.training.protocol import atomic_write_json, git_sha  # noqa: E402

TOL = 1e-4      # on SCALED features (float32 rolling sums from different starts)


def compare(trades: pd.DataFrame, cfg: dict, mode: str, mean, std,
            T: int, symbol: str = "SYM") -> dict:
    """trades: columns agg_id, price, qty, time (ms), is_buyer_maker."""
    B, C = cfg["bars"], cfg["cleaning"]
    trades = trades.sort_values(["time", "agg_id"], kind="mergesort")

    # offline
    clean = clean_ticks(trades, mode, static_pct=C["static_pct"],
                        k0=C["adaptive_k0"],
                        tick_sigma_window=C["tick_sigma_window"],
                        regime_clip=tuple(C["regime_clip"]))
    bars = trades_to_bars(clean, B["bar_seconds"])
    off = (bars_to_features(bars, B["z_window"], B["vol_window"])
           .to_numpy(np.float32) - mean) / std
    off_idx = {t: i for i, t in enumerate(bars.index)}

    # online: trade-driven closes plus the timer close on exchange time
    agg = BarAggregator(B["bar_seconds"], mode, C, cfg["pipeline"].get("bar_grace_ms", 1000))
    feat = OnlineFeaturizer(B, T, mean, std)
    n_win = n_bad = 0
    max_diff = 0.0
    kept = []
    windows = []

    def consume(bs):
        nonlocal n_win, n_bad, max_diff
        for b in bs:
            kept.extend([b["n_trades"]])
            x = feat.add(b)
            if x is None or not feat.warm(b["symbol"]):
                continue
            t_end = pd.Timestamp(b["bar_time"])
            windows.append((t_end, x))
            i = off_idx.get(t_end)
            if i is None or i + 1 < T:
                continue
            d = float(np.abs(x - off[i + 1 - T:i + 1]).max())
            n_win += 1
            max_diff = max(max_diff, d)
            n_bad += d > TOL

    for r in trades.itertuples(index=False):
        t = {"s": symbol, "a": int(r.agg_id), "p": float(r.price),
             "q": float(r.qty), "T": int(r.time), "m": bool(r.is_buyer_maker)}
        consume(agg.due(int(r.time)))
        consume(agg.add(t))
    consume(agg.flush_all())

    cl = agg.cleaners[symbol]
    removed_off = set(trades["agg_id"]) - set(clean["agg_id"])
    removed_on = set(cl.removed_ids)
    n_rm_on = cl.n_removed
    # path parity: offline features from the trades the online filter kept
    path = {"windows": 0, "over_tol": 0, "max_abs_diff": 0.0}
    if mode == "none" or removed_on == removed_off:
        path = {"windows": n_win, "over_tol": int(n_bad), "max_abs_diff": max_diff}
    else:
        kept_tr = trades[~trades["agg_id"].isin(removed_on)]
        bars2 = trades_to_bars(kept_tr, B["bar_seconds"])
        off2 = (bars_to_features(bars2, B["z_window"], B["vol_window"])
                .to_numpy(np.float32) - mean) / std
        idx2 = {t: i for i, t in enumerate(bars2.index)}
        for t_end, x in windows:
            i = idx2.get(t_end)
            if i is None or i + 1 < T:
                continue
            d = float(np.abs(x - off2[i + 1 - T:i + 1]).max())
            path["windows"] += 1
            path["over_tol"] += int(d > TOL)
            path["max_abs_diff"] = max(path["max_abs_diff"], d)
    inter = len(removed_on & removed_off)
    return {
        "symbol": symbol, "mode": mode, "trades": int(len(trades)),
        "removed_offline": len(removed_off), "removed_online": int(n_rm_on),
        "late_dropped": int(agg.n_late),
        "bars_offline": int(len(bars)), "windows_compared": n_win,
        "windows_over_tol": int(n_bad), "max_abs_diff": max_diff,
        "online_trades_in_bars": int(sum(kept)),
        "offline_trades_in_bars": int(bars["n_trades"].sum()),
        "path_windows": path["windows"], "path_over_tol": path["over_tol"],
        "path_max_abs_diff": path["max_abs_diff"],
        "clean_precision": inter / len(removed_on) if removed_on else 1.0,
        "clean_recall": inter / len(removed_off) if removed_off else 1.0,
        "skew_windows_share": n_bad / n_win if n_win else 0.0,
    }


def verdict(r: dict) -> bool:
    """PASS = the serving code path is exact. Cleaning agreement is reported."""
    return bool(r["path_windows"] > 0 and r["path_over_tol"] == 0)


def kafka_trades(cfg: dict, symbols: list[str]) -> dict[str, pd.DataFrame]:
    from kafka import KafkaConsumer
    P = cfg["pipeline"]
    c = KafkaConsumer(P["topic_raw"], bootstrap_servers=P["kafka_bootstrap"],
                      auto_offset_reset="earliest", enable_auto_commit=False,
                      group_id=None, consumer_timeout_ms=15000,
                      value_deserializer=lambda b: json.loads(b))
    rows = {s: [] for s in symbols}
    for m in c:
        v = m.value
        if v["s"] in rows:
            rows[v["s"]].append((v["a"], v["p"], v["q"], v["T"], v["m"]))
    return {s: pd.DataFrame(r, columns=["agg_id", "price", "qty", "time",
                                        "is_buyer_maker"])
            for s, r in rows.items() if r}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--clean", default="adaptive")
    ap.add_argument("--kafka", action="store_true", required=True)
    ap.add_argument("--symbols", nargs="*", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(ROOT, args.config)))
    sc = json.load(open(os.path.join(cfg["paths"]["processed_dir"], args.clean,
                                     "scaler.json")))
    mean = np.asarray(sc["mean"], np.float32)
    std = np.asarray(sc["std"], np.float32)
    T = cfg["windows"]["window_bars"]
    syms = args.symbols or cfg["pipeline"]["latency"]["symbols"]

    data = kafka_trades(cfg, syms)
    results = []
    print("=== SCALER / FEATURE PARITY CHECK (live trades replayed from Kafka) ===")
    for s, df in data.items():
        for mode in ("none", args.clean):
            r = compare(df, cfg, mode, mean, std, T, s)
            r["pass"] = verdict(r)
            results.append(r)
            print(f"{s:10s} {mode:8s} trades={r['trades']:7d} "
                  f"removed off/on={r['removed_offline']}/{r['removed_online']} "
                  f"path windows={r['path_windows']:5d} max|diff|={r['path_max_abs_diff']:.1e} "
                  f"| clean P/R={r['clean_precision']:.2f}/{r['clean_recall']:.2f} "
                  f"skew windows={r['skew_windows_share']:.1%}  "
                  f"{'PASS' if r['pass'] else 'FAIL'}")
    tested = [r for r in results if r["path_windows"] > 0]
    ok = bool(tested) and all(r["pass"] for r in tested)
    out = {"features": sc["features"], "scaler_mean_len": len(mean), "window": T,
           "tol": TOL, "results": results, "pass": ok, "git_sha": git_sha(),
           "note": "symbols with 0 windows (shorter than warm-up) are not counted"}
    atomic_write_json(out, os.path.join(ROOT, "artifacts", "latency", "parity.json"))
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'} "
          f"({len(tested)} symbol/mode runs with windows)")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
