"""M2 — Download Binance historical aggTrades (free bulk data, no API key).

Source: https://data.binance.vision/data/spot/daily/aggTrades/{PAIR}/
        {PAIR}-aggTrades-YYYY-MM-DD.zip

Each day cached as parquet: data/raw/{PAIR}/{YYYY-MM-DD}.parquet
Columns: agg_id, price, qty, time_ms, is_buyer_maker

Usage:
    # Download for specific events (Gate 1 check — 10 events only):
    python -m src.data.download_binance --limit-events 10

    # Full download (after Gate 1 passes):
    python -m src.data.download_binance
"""
import io
import os
import zipfile
from datetime import timedelta

import pandas as pd
import requests
import yaml
from tqdm import tqdm

BASE = "https://data.binance.vision/data/spot/daily/aggTrades"
AGG_COLS = [
    "agg_id", "price", "qty", "first_id", "last_id",
    "time", "is_buyer_maker", "is_best_match",
]
KEEP = ["agg_id", "price", "qty", "time", "is_buyer_maker"]


def _parse_csv(raw: bytes) -> pd.DataFrame:
    """Parse aggTrades CSV — handles both headerless (old) and headered (new)."""
    first_line = raw.split(b"\n", 1)[0].lower()
    has_header = b"agg" in first_line or b"price" in first_line
    skip = 1 if has_header else 0

    df = pd.read_csv(
        io.BytesIO(raw), header=None, skiprows=skip,
        names=AGG_COLS, usecols=KEEP,
    )
    # is_buyer_maker comes as string "True"/"False" or bool
    df["is_buyer_maker"] = (
        df["is_buyer_maker"].astype(str).str.lower().isin(["true", "1"])
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce").astype("int64")

    # 2025+ files use microseconds — normalize to ms
    big = df["time"] > 10_000_000_000_000
    if big.any():
        df.loc[big, "time"] = df.loc[big, "time"] // 1000

    return df.dropna().reset_index(drop=True)


def download_day(pair: str, day: str, raw_dir: str) -> str | None:
    """Download one pair-day. Returns parquet path or None if unavailable."""
    out = os.path.join(raw_dir, pair, f"{day}.parquet")
    if os.path.exists(out):
        return out  # already cached

    url = f"{BASE}/{pair}/{pair}-aggTrades-{day}.zip"
    try:
        r = requests.get(url, timeout=60)
    except requests.exceptions.RequestException as e:
        print(f"    network error {pair} {day}: {e}")
        return None

    if r.status_code in (403, 404, 451):
        return None  # 404=not traded, 403/451=blocked/geo
    if r.status_code >= 400:
        print(f"    HTTP {r.status_code} for {pair} {day}")
        return None

    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        raw = zf.read(zf.namelist()[0])

    df = _parse_csv(raw)
    if df.empty:
        return None

    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def plan_from_labels(labels: pd.DataFrame, before: int, after: int,
                     normal_days: int) -> dict[str, list[str]]:
    """Build download plan: {pair: [YYYY-MM-DD, ...]}."""
    plan: dict[str, set[str]] = {}

    for pair, grp in labels.groupby("pair"):
        days: set[str] = set()
        pump_days: set[str] = set()

        for t in grp["pump_time"]:
            for d in range(-before, after + 1):
                day = (t + timedelta(days=d)).strftime("%Y-%m-%d")
                days.add(day)
            pump_days.add(t.strftime("%Y-%m-%d"))

        # Add pump-free normal days (for negatives + pretraining)
        t0 = grp["pump_time"].min()
        added, k = 0, 7
        while added < normal_days and k < 180:
            day = (t0 - timedelta(days=k)).strftime("%Y-%m-%d")
            if day not in pump_days:
                days.add(day)
                added += 1
            k += 7

        plan[pair] = sorted(days)

    return plan


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--limit-events", type=int, default=0,
                    help="Only download for first N events (for Gate 1)")
    ap.add_argument("--limit-pairs", type=int, default=0,
                    help="Only download first N pairs")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    raw_dir = cfg["paths"]["raw_dir"]
    labels = pd.read_csv(cfg["paths"]["labels_csv"], parse_dates=["pump_time"])

    if args.limit_events:
        labels = labels.head(args.limit_events)
        print(f"Gate 1 mode: downloading for {len(labels)} events only")
    elif args.limit_pairs:
        keep = labels["pair"].unique()[:args.limit_pairs]
        labels = labels[labels["pair"].isin(keep)]
        print(f"Limited to {len(keep)} pairs, {len(labels)} events")

    plan = plan_from_labels(
        labels,
        cfg["data"]["days_before_event"],
        cfg["data"]["days_after_event"],
        cfg["data"]["normal_days_per_symbol"],
    )

    total_days = sum(len(v) for v in plan.values())
    print(f"Download plan: {len(plan)} pairs, {total_days} pair-days")

    ok = miss = 0
    for pair in plan:
        for day in tqdm(plan[pair], desc=pair, leave=False):
            path = download_day(pair, day, raw_dir)
            if path:
                ok += 1
            else:
                miss += 1

    print(f"\nDone: downloaded/cached={ok}, unavailable={miss}")
    print(f"  Available rate: {100 * ok / max(ok + miss, 1):.1f}%")


if __name__ == "__main__":
    main()
