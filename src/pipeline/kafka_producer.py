"""Pipeline Tier 1: Binance WebSocket -> Kafka topic `raw-trades`.

Free public stream, no API key needed. One connection subscribes to multiple
symbols via the combined stream endpoint. Auto-reconnects on drops.

Message written to Kafka:
  key   : symbol (bytes)
  value : JSON {s, a, p, q, T, m}
    s = symbol, a = aggId, p = price (float), q = qty (float),
    T = trade time ms (int), m = is_buyer_maker (bool)

Usage:
    python -m src.pipeline.kafka_producer --config config/config.yaml
    python -m src.pipeline.kafka_producer --symbols BTCUSDT ETHUSDT
"""
from __future__ import annotations

import argparse
import json
import time

import websocket
import yaml


WS_BASE = "wss://stream.binance.com:9443/stream?streams="


def run(symbols: list[str], bootstrap: str, topic_raw: str) -> None:
    try:
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers=bootstrap,
            value_serializer=lambda v: json.dumps(v).encode(),
            key_serializer=lambda k: k.encode(),
            linger_ms=20,
            compression_type="gzip",
        )
    except ImportError:
        print("ERROR: kafka-python not installed. Run: pip install kafka-python")
        return

    streams = "/".join(f"{s.lower()}@aggTrade" for s in symbols)
    url = WS_BASE + streams
    counter = {"n": 0, "t0": time.time()}

    def on_message(_ws, raw_msg: str) -> None:
        msg = json.loads(raw_msg).get("data", {})
        if msg.get("e") != "aggTrade":
            return
        rec = {
            "s": msg["s"],
            "a": int(msg["a"]),
            "p": float(msg["p"]),
            "q": float(msg["q"]),
            "T": int(msg["T"]),
            "m": bool(msg["m"]),
        }
        producer.send(topic_raw, key=rec["s"], value=rec)
        counter["n"] += 1
        if counter["n"] % 1000 == 0:
            elapsed = max(time.time() - counter["t0"], 1)
            print(f"  {counter['n']:,} trades ({counter['n']/elapsed:.0f}/s)")

    def on_error(_ws, err):
        print(f"WS error: {err}")

    def on_close(_ws, *_):
        print("WS closed — will reconnect")

    while True:
        print(f"Connecting: {url}")
        ws = websocket.WebSocketApp(
            url, on_message=on_message,
            on_error=on_error, on_close=on_close)
        ws.run_forever(ping_interval=180, ping_timeout=15)
        print("Reconnecting in 5s ...")
        time.sleep(5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--symbols", nargs="*", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    P = cfg["pipeline"]
    syms = args.symbols or P.get("live_symbols", ["BTCUSDT", "ETHUSDT"])
    run(syms, P["kafka_bootstrap"], P["topic_raw"])


if __name__ == "__main__":
    main()
