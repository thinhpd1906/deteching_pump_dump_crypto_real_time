"""GATE 1 — Label alignment verification.

Plot 1-minute close price ±2 hours around each pump event.
Visual check: does a price spike start at/after the labeled pump time?

PASS: ≥ 7/10 events show visible spike aligned with label.
FAIL: debug timezone; try shifting labels by ±1..±12 hours.

Usage:
    python -m scripts.gate1_alignment
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yaml


def load_ticks_for_pair(raw_dir: str, pair: str) -> pd.DataFrame | None:
    """Load all cached parquet files for a pair, concatenated."""
    files = sorted(glob.glob(os.path.join(raw_dir, pair, "*.parquet")))
    if not files:
        return None
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["agg_id"]).sort_values("time")
    df["ts"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df


def make_1min_close(ticks: pd.DataFrame) -> pd.Series:
    """Resample ticks to 1-minute close prices."""
    return ticks.set_index("ts")["price"].resample("1min").last().dropna()


def plot_gate1(labels: pd.DataFrame, raw_dir: str, out_dir: str,
               hours_around: float = 2.0):
    """Generate one plot per event, plus a summary grid."""
    os.makedirs(out_dir, exist_ok=True)
    results = []

    for idx, row in labels.iterrows():
        pair = row["pair"]
        pump_time = pd.Timestamp(row["pump_time"])

        ticks = load_ticks_for_pair(raw_dir, pair)
        if ticks is None or len(ticks) < 100:
            results.append({"pair": pair, "pump_time": pump_time,
                            "status": "NO_DATA"})
            print(f"  [{idx}] {pair} {pump_time} — NO DATA")
            continue

        close = make_1min_close(ticks)
        # Zoom to ±2 hours around pump
        t0 = pump_time - pd.Timedelta(hours=hours_around)
        t1 = pump_time + pd.Timedelta(hours=hours_around)
        window = close.loc[t0:t1]

        if len(window) < 10:
            results.append({"pair": pair, "pump_time": pump_time,
                            "status": "INSUFFICIENT_DATA"})
            print(f"  [{idx}] {pair} {pump_time} — insufficient data in window")
            continue

        # Compute metrics
        pre_pump = close.loc[t0:pump_time]
        post_pump = close.loc[pump_time:t1]
        if len(pre_pump) > 0 and len(post_pump) > 0:
            pre_mean = pre_pump.mean()
            post_max = post_pump.max()
            spike_pct = (post_max - pre_mean) / pre_mean * 100
        else:
            spike_pct = 0.0

        aligned = spike_pct > 2.0  # > 2% spike after label = aligned
        status = f"ALIGNED (spike={spike_pct:.1f}%)" if aligned else f"WEAK ({spike_pct:.1f}%)"
        results.append({"pair": pair, "pump_time": pump_time,
                        "status": status, "spike_pct": spike_pct,
                        "aligned": aligned})

        # Individual plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(window.index, window.values, linewidth=0.8, color="#2563EB")
        ax.axvline(pump_time, color="red", linewidth=1.5, linestyle="--",
                   label=f"Pump label ({pump_time.strftime('%H:%M')} UTC)")
        ax.set_title(f"{pair} — {pump_time.date()} — {status}",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Price")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"gate1_{idx:02d}_{pair}.png"), dpi=120)
        plt.close(fig)
        print(f"  [{idx}] {pair} {pump_time} — {status}")

    # Summary
    df_res = pd.DataFrame(results)
    if "aligned" in df_res.columns:
        aligned_bool = df_res["aligned"].fillna(False).astype(bool)
    else:
        aligned_bool = pd.Series(False, index=df_res.index)
    n_aligned = int(aligned_bool.sum())
    usable_mask = ~df_res["status"].isin(["NO_DATA", "INSUFFICIENT_DATA"])
    n_total = int(usable_mask.sum())
    print(f"\n{'='*60}")
    print(f"GATE 1 RESULT: {n_aligned}/{n_total} events aligned")
    if n_total > 0 and n_aligned / n_total >= 0.7:
        print("GATE 1: PASS (>=70% aligned)")
    else:
        print("GATE 1: FAIL -- debug timezone or label source")
        print("  Try: shift labels by +/-1h increments and re-run")
    print(f"{'='*60}")

    df_res.to_csv(os.path.join(out_dir, "gate1_results.csv"), index=False)
    return df_res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-events", type=int, default=0,
                    help="Only check the first N labeled events (matches "
                         "download_binance.py --limit-events for Gate 1)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open("config/config.yaml"))
    labels = pd.read_csv(cfg["paths"]["labels_csv"], parse_dates=["pump_time"])

    if args.limit_events:
        labels = labels.head(args.limit_events)
        print(f"Gate 1 mode: checking first {len(labels)} events only")

    # Use only events that have downloaded data
    available = set()
    for pair_dir in glob.glob(os.path.join(cfg["paths"]["raw_dir"], "*")):
        if os.path.isdir(pair_dir):
            available.add(os.path.basename(pair_dir))

    labels = labels[labels["pair"].isin(available)]
    print(f"Gate 1: checking {len(labels)} events with available data\n")

    if len(labels) == 0:
        print("ERROR: No data downloaded yet. Run download_binance first:")
        print("  python -m src.data.download_binance --limit-events 10")
        return

    plot_gate1(labels, cfg["paths"]["raw_dir"], "artifacts/gate1")


if __name__ == "__main__":
    main()
