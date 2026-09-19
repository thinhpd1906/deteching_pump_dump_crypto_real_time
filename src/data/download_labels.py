"""M1 — Download pump-and-dump ground-truth labels.

Source: github.com/SystemsLab-Sapienza/pump-and-dump-dataset
CSV columns (verified): symbol, group, date, hour, exchange

Processing:
  1. Keep exchange == binance only
  2. Build trading pair: symbol + quote_asset (e.g. LRCBTC)
  3. Parse timestamp as UTC
  4. Dedupe on exact (pair, pump_time) -> 338 rows, matches CLAUDE.md's
     documented expected size. This is the ONLY dedupe applied for this
     single-source (La Morgia) load.
  5. Save normalized CSV

Note: `fuzzy_dedupe()` below (pair, |delta t| < 30 min chain-clustering) is
NOT called in this pipeline. It is intentionally kept for later, when
merging in ArdiaD / Fantazzini sources per CLAUDE.md's documented pitfall
("labels overlap across sources ... dedupe on (pair, |dt|<30min)"). It must
NOT be used to dedupe La Morgia against itself — same-pair events called by
different Telegram groups within 30 min are kept as distinct rows here by
user decision.

Usage:
    python -m src.data.download_labels
"""
import io
import os

import pandas as pd
import requests
import yaml

FUZZY_DEDUPE_MINUTES = 30

RAW_URL = (
    "https://raw.githubusercontent.com/SystemsLab-Sapienza/"
    "pump-and-dump-dataset/master/pump_telegram.csv"
)


def download_labels(quote_asset: str = "BTC") -> pd.DataFrame:
    """Fetch La Morgia CSV, filter to Binance, normalize."""
    print(f"Fetching labels from {RAW_URL} ...")
    resp = requests.get(RAW_URL, timeout=30)
    resp.raise_for_status()
    print(f"  HTTP {resp.status_code}, {len(resp.content):,} bytes")

    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Exchanges: {df['exchange'].value_counts().to_dict()}")

    # Filter to Binance
    df = df[df["exchange"].str.lower().eq("binance")].copy()
    print(f"  After binance filter: {len(df)} rows")

    # Build pair and timestamp
    df["pair"] = df["symbol"].str.upper() + quote_asset.upper()
    df["pump_time"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["hour"].astype(str),
        utc=True,
        format="mixed",
    )

    # Add source column (for future merging with ArdiaD + Fantazzini)
    df["source"] = "lamorgia"

    # Exact dedupe
    before = len(df)
    df = (
        df.sort_values("pump_time")
        .drop_duplicates(subset=["pair", "pump_time"])
        .reset_index(drop=True)
    )
    print(f"  Exact dedupe: {before} -> {len(df)} rows")

    # NOTE: fuzzy_dedupe() intentionally NOT applied here — see module
    # docstring. Reserved for future cross-source (ArdiaD/Fantazzini) merge.

    return df[["pair", "symbol", "group", "pump_time", "source"]]


def fuzzy_dedupe(df: pd.DataFrame, minutes: int = 30) -> pd.DataFrame:
    """Chain-cluster same-pair rows whose consecutive time gap < `minutes`.

    Rationale: the same real-world pump event is sometimes announced by
    multiple Telegram groups within a minute or two of each other (e.g.
    BLZBTC pumped by group "MPG" at 16:59 and "CPI" at 17:00 on the same
    day). These are the same market event, not independent pumps, and
    would otherwise double-count positives / bleed into each other's
    windows. Keeps the earliest timestamp and its `group` per cluster,
    and records all merged groups in `group` as a "|"-joined list when
    more than one is merged.
    """
    out_rows = []
    for pair, g in df.sort_values("pump_time").groupby("pair", sort=False):
        g = g.reset_index(drop=True)
        cluster_id = 0
        cluster_ids = [0]
        for i in range(1, len(g)):
            gap_min = (g.loc[i, "pump_time"] - g.loc[i - 1, "pump_time"]).total_seconds() / 60
            if gap_min >= minutes:
                cluster_id += 1
            cluster_ids.append(cluster_id)
        g["_cluster"] = cluster_ids
        for _, cg in g.groupby("_cluster"):
            cg = cg.sort_values("pump_time")
            canonical = cg.iloc[0].copy()
            if len(cg) > 1:
                canonical["group"] = "|".join(cg["group"].astype(str).tolist())
            out_rows.append(canonical)
    return pd.DataFrame(out_rows).sort_values("pump_time").reset_index(drop=True)


def main():
    cfg = yaml.safe_load(open("config/config.yaml"))
    out_path = cfg["paths"]["labels_csv"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df = download_labels(cfg["data"]["quote_asset"])
    df.to_csv(out_path, index=False)

    print(f"\n=== SAVED: {out_path} ===")
    print(f"  Events:     {len(df)}")
    print(f"  Symbols:    {df['pair'].nunique()}")
    print(f"  Time range: {df['pump_time'].min()} .. {df['pump_time'].max()}")
    print(f"\nFirst 10 events:")
    print(df.head(10).to_string(index=False))
    print(f"\nTop 10 most-pumped pairs:")
    print(df["pair"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
