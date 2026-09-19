"""Pipeline Tier 2 (Fallback): pure-Python bar aggregation from Kafka.

If Spark is too heavy to run locally, use this instead.
Claims in the paper become "lightweight stream processor" instead of
"Spark Structured Streaming" -- Kafka claim and real-time claim both intact.

Usage:
    python -m src.pipeline.python_bars --config config/config.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import pandas as pd
import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    P, B = cfg["pipeline"], cfg["bars"]

    try:
        from kafka import KafkaConsumer, KafkaProducer
    except ImportError:
        print("ERROR: pip install kafka-python")
        return

    consumer = KafkaConsumer(
        P["topic_raw"],
        bootstrap_servers=P["kafka_bootstrap"],
        value_deserializer=lambda b: json.loads(b),
        auto_offset_reset="latest",
        group_id="pnd-python-bars",
    )
    producer = KafkaProducer(
        bootstrap_servers=P["kafka_bootstrap"],
        value_serializer=lambda v: json.dumps(v, default=str).encode(),
    )

    # per-symbol buffers: {symbol: [trades in current bar]}
    bar_buf: dict[str, list] = defaultdict(list)
    bar_start: dict[str, int] = {}              # bar open time ms
    seen_ids: dict[str, set] = defaultdict(set) # dedup within last 30s
    bar_s_ms = B["bar_seconds"] * 1000

    print(f"Python bars aggregator running on {P['topic_raw']} -> {P['topic_bars']}")
    n_bars = 0

    for msg in consumer:
        r = msg.value
        sym, agg_id, T = r["s"], r["a"], int(r["T"])

        # streaming dedup (within symbol)
        if agg_id in seen_ids[sym]:
            continue
        seen_ids[sym].add(agg_id)
        # evict old ids (rough 60s window)
        if len(seen_ids[sym]) > 50000:
            seen_ids[sym] = set(list(seen_ids[sym])[-25000:])

        if sym not in bar_start:
            bar_start[sym] = (T // bar_s_ms) * bar_s_ms

        # flush when trade crosses into a new bar
        bar_end = bar_start[sym] + bar_s_ms
        if T >= bar_end and bar_buf[sym]:
            trades = bar_buf[sym]
            prices = [t["p"] for t in trades]
            qtys   = [t["q"] for t in trades]
            tb     = [t["q"] for t in trades if not t["m"]]

            bar = {
                "symbol":        sym,
                "bar_time":      pd.Timestamp(bar_start[sym], unit="ms",
                                              tz="UTC").isoformat(),
                "open":          prices[0],
                "high":          max(prices),
                "low":           min(prices),
                "close":         prices[-1],
                "volume":        sum(qtys),
                "n_trades":      len(trades),
                "taker_buy_vol": sum(tb),
                "quote_vol":     sum(p * q for p, q in zip(prices, qtys)),
            }
            producer.send(P["topic_bars"], key=sym.encode(),
                          value=bar)
            n_bars += 1
            if n_bars % 50 == 0:
                print(f"  {n_bars} bars emitted, last: {sym} {bar['bar_time']}")

            bar_buf[sym] = []
            bar_start[sym] = (T // bar_s_ms) * bar_s_ms

        bar_buf[sym].append(r)


if __name__ == "__main__":
    main()
