"""Pipeline Tier 2 (Fallback): pure-Python bar aggregation from Kafka.

raw-trades (Kafka) -> StreamingCleaner (same rule as training's clean_ticks,
src/cleaning/filters.py) -> 5 s bars -> bars (Kafka)

If Spark is too heavy to run locally, use this instead.
Claims in the paper become "lightweight stream processor" instead of
"Spark Structured Streaming" -- Kafka claim and real-time claim both intact.

A bar [t, t+5s) is closed by whichever comes first:
  * a cleaned trade with time >= t+5s arrives (busy symbols), or
  * the local clock passes t+5s+grace (pipeline.bar_grace_ms), so a quiet
    symbol's bar is not held until its next trade.
Empty bars are not emitted; the inference service re-grids them exactly like
training (normalize_bars). A trade arriving for an already closed bar is
dropped and counted as late.

Every bar carries timing fields for the latency study (A5):
  first_T / last_T : exchange time (ms) of the first / last trade in the bar
  emit_ms          : local wall clock when the bar was handed to Kafka

Usage:
    python -m src.pipeline.python_bars --config config/config.yaml [--clean adaptive]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import pandas as pd
import yaml

from src.cleaning.filters import StreamingCleaner


def make_bar(sym: str, start_ms: int, trades: list[dict]) -> dict:
    prices = [t["p"] for t in trades]
    qtys = [t["q"] for t in trades]
    return {
        "symbol": sym,
        "bar_time": pd.Timestamp(start_ms, unit="ms", tz="UTC").isoformat(),
        "open": prices[0],
        "high": max(prices),
        "low": min(prices),
        "close": prices[-1],
        "volume": sum(qtys),
        "n_trades": len(trades),
        "taker_buy_vol": sum(t["q"] for t in trades if not t["m"]),
        "quote_vol": sum(p * q for p, q in zip(prices, qtys)),
        "first_T": int(trades[0]["T"]),
        "last_T": int(trades[-1]["T"]),
    }


class BarAggregator:
    """Per-symbol cleaning + 5 s bar building. Pure logic, no Kafka (testable)."""

    def __init__(self, bar_seconds: int, clean_mode: str, clean_cfg: dict,
                 grace_ms: int = 1000):
        self.bar_ms = int(bar_seconds) * 1000
        self.grace_ms = int(grace_ms)
        self.cleaners: dict[str, StreamingCleaner] = defaultdict(
            lambda: StreamingCleaner(
                clean_mode, static_pct=clean_cfg["static_pct"],
                k0=clean_cfg["adaptive_k0"],
                tick_sigma_window=clean_cfg["tick_sigma_window"],
                regime_clip=tuple(clean_cfg["regime_clip"])))
        self.buf: dict[str, list] = defaultdict(list)
        self.start: dict[str, int] = {}
        self.closed_until: dict[str, int] = {}     # end of last closed bar
        self.n_late = 0

    def _close(self, sym: str) -> list[dict]:
        if not self.buf[sym]:
            return []
        bar = make_bar(sym, self.start[sym], self.buf[sym])
        self.closed_until[sym] = self.start[sym] + self.bar_ms
        self.buf[sym] = []
        return [bar]

    def _add_clean(self, sym: str, t: dict) -> list[dict]:
        T = int(t["T"])
        if T < self.closed_until.get(sym, -1):
            self.n_late += 1
            return []
        out = []
        b0 = (T // self.bar_ms) * self.bar_ms
        if self.buf[sym] and b0 != self.start[sym]:
            out += self._close(sym)
        if not self.buf[sym]:
            self.start[sym] = b0
        self.buf[sym].append(t)
        return out

    def add(self, t: dict) -> list[dict]:
        sym = t["s"]
        out = []
        for c in self.cleaners[sym].push(t):
            out += self._add_clean(sym, c)
        return out

    def due(self, now_exchange_ms: int) -> list[dict]:
        """Close bars whose end + grace has passed (exchange-clock estimate)."""
        out = []
        for sym in list(self.cleaners):
            cl = self.cleaners[sym]
            pt = cl.pending_time
            # the held tick of a bar that is due can no longer get a successor
            # inside the bar in time: release it (kept, as batch keeps a
            # file's last tick)
            if pt is not None and now_exchange_ms >= \
                    (pt // self.bar_ms + 1) * self.bar_ms + self.grace_ms:
                for c in cl.flush():
                    out += self._add_clean(sym, c)
            if self.buf[sym] and now_exchange_ms >= \
                    self.start[sym] + self.bar_ms + self.grace_ms:
                out += self._close(sym)
        return out

    def flush_all(self) -> list[dict]:
        out = []
        for sym in list(self.cleaners):
            for c in self.cleaners[sym].flush():
                out += self._add_clean(sym, c)
            out += self._close(sym)
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--clean", default="adaptive",
                    help="tick cleaning mode, must match the model's dataset")
    ap.add_argument("--clock-offset-ms", type=float, default=0.0,
                    help="exchange clock minus local clock (ms), for the "
                         "timer close; measure_latency.py passes it")
    ap.add_argument("--group-id", default="pnd-python-bars")
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
        group_id=args.group_id,
    )
    producer = KafkaProducer(
        bootstrap_servers=P["kafka_bootstrap"],
        value_serializer=lambda v: json.dumps(v, default=str).encode(),
        linger_ms=0,
    )
    agg = BarAggregator(B["bar_seconds"], args.clean, cfg["cleaning"],
                        P.get("bar_grace_ms", 1000))
    print(f"Python bars aggregator ({args.clean} cleaning) running on "
          f"{P['topic_raw']} -> {P['topic_bars']}", flush=True)
    n_bars = 0

    def emit(bars: list[dict]) -> None:
        nonlocal n_bars
        for bar in bars:
            bar["emit_ms"] = time.time() * 1000.0
            producer.send(P["topic_bars"], key=bar["symbol"].encode(), value=bar)
            n_bars += 1
            if n_bars % 500 == 0:
                rm = sum(c.n_removed for c in agg.cleaners.values())
                print(f"  {n_bars} bars emitted, last: {bar['symbol']} "
                      f"{bar['bar_time']}  removed_ticks={rm} late={agg.n_late}",
                      flush=True)
        if bars:
            producer.flush()

    while True:
        polled = consumer.poll(timeout_ms=100)
        for records in polled.values():
            for msg in records:
                emit(agg.add(msg.value))
        emit(agg.due(int(time.time() * 1000 + args.clock_offset_ms)))


if __name__ == "__main__":
    main()
