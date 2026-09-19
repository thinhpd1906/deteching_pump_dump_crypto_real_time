"""Pipeline Tier 3: Real-time inference service.

bars (Kafka) -> rolling feature buffer -> SAME bars_to_features() as training
-> scaler -> model -> anomaly score -> alerts (Kafka) + latency measurement

FEATURE PARITY RULE: this file IMPORTS src.data.features.bars_to_features.
Never re-implement feature computation in the serving path.
This eliminates training-serving skew by construction.

Usage:
    python -m src.pipeline.inference_service --clean adaptive --model at_pre
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import torch
import yaml


def load_model(art: str, tag: str, dev: str):
    path = os.path.join(art, f"{tag}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            f"Run: python -m src.training.finetune --arch at --pretrained "
            f"--clean <mode>")
    ck = torch.load(path, map_location=dev)
    M = ck["model_cfg"]

    from src.models.anomaly_transformer import AnomalyTransformer, ATClassifier
    from src.models.deep_baselines import CNNBiLSTM

    if ck["arch"] == "at":
        enc = AnomalyTransformer(ck["win"], ck["d_in"],
                                 M["d_model"], M["n_heads"], M["n_layers"])
        model = ATClassifier(enc, M["d_model"]).to(dev).eval()
    else:
        model = CNNBiLSTM(ck["d_in"]).to(dev).eval()

    model.load_state_dict(ck["state_dict"])
    return model, ck


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--clean", default="adaptive")
    ap.add_argument("--model", default="at_pre", help="artifact tag name")
    ap.add_argument("--symbols", nargs="*", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    P, B, W = cfg["pipeline"], cfg["bars"], cfg["windows"]
    dd = os.path.join(cfg["paths"]["processed_dir"], args.clean)
    art = os.path.join(cfg["paths"]["artifacts_dir"], args.clean)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model, ck = load_model(art, args.model, dev)
    thr = cfg["pipeline"].get("alert_threshold") or ck.get("thr", 0.5)
    sc = json.load(open(os.path.join(dd, "scaler.json")))
    mean = np.asarray(sc["mean"], np.float32)
    std_ = np.asarray(sc["std"], np.float32)
    T = ck["win"]
    keep_bars = B["z_window"] + T + 32   # rolling context buffer

    try:
        from kafka import KafkaConsumer, KafkaProducer
    except ImportError:
        print("ERROR: pip install kafka-python"); return

    # Importing THE SAME feature function used during training
    from src.data.features import bars_to_features, normalize_bars

    consumer = KafkaConsumer(
        P["topic_bars"],
        bootstrap_servers=P["kafka_bootstrap"],
        value_deserializer=lambda b: json.loads(b),
        auto_offset_reset="latest",
        group_id="pnd-inference",
    )
    producer = KafkaProducer(
        bootstrap_servers=P["kafka_bootstrap"],
        value_serializer=lambda v: json.dumps(v).encode(),
    )

    BAR_COLS = ["open", "high", "low", "close", "volume",
                "n_trades", "taker_buy_vol", "quote_vol"]
    buf: dict[str, pd.DataFrame] = defaultdict(pd.DataFrame)
    latencies: deque = deque(maxlen=1000)

    syms = args.symbols or P.get("live_symbols", [])
    print(f"Inference service: model={args.model} thr={thr:.4f} "
          f"win={T} dev={dev}")
    print(f"  Watching: {syms or 'all symbols'}")
    print(f"  Consuming topic: {P['topic_bars']}")

    for msg in consumer:
        r = msg.value
        sym = r["symbol"]
        if syms and sym not in syms:
            continue

        # Record trade timestamp for latency measurement
        trade_T = r.get("bar_time")  # ISO string

        row = pd.DataFrame(
            [{c: float(r[c]) for c in BAR_COLS}],
            index=[pd.Timestamp(r["bar_time"], tz="UTC")],
        )
        buf[sym] = pd.concat([buf[sym], row]).sort_index()
        buf[sym] = buf[sym][~buf[sym].index.duplicated(keep="last")]
        buf[sym] = buf[sym].tail(keep_bars)

        bars = normalize_bars(buf[sym], B["bar_seconds"])
        bars["vwap"] = np.where(
            bars["volume"] > 0,
            bars["quote_vol"] / bars["volume"].replace(0, np.nan),
            bars["close"])
        bars["vwap"] = bars["vwap"].fillna(bars["close"])

        if len(bars) < T:
            continue

        feats = bars_to_features(bars, B["z_window"], B["vol_window"])
        x = (feats.tail(T).to_numpy(np.float32) - mean) / std_

        t_infer_start = time.time()
        with torch.no_grad():
            xb = torch.from_numpy(x[None]).to(dev)
            out = model(xb)
            logit = out[0] if isinstance(out, tuple) else out
            score = float(torch.sigmoid(logit)[0])
        infer_ms = (time.time() - t_infer_start) * 1000

        # Latency: bar_time -> now (includes Kafka lag, not just inference)
        now_ts = pd.Timestamp.now(tz="UTC")
        bar_ts = pd.Timestamp(r["bar_time"], tz="UTC")
        e2e_ms = (now_ts - bar_ts).total_seconds() * 1000
        latencies.append(e2e_ms)

        flag = ""
        if score >= thr:
            alert = {
                "symbol": sym,
                "bar_time": r["bar_time"],
                "score": round(score, 4),
                "threshold": round(thr, 4),
                "model": args.model,
                "infer_ms": round(infer_ms, 1),
            }
            producer.send(P["topic_alerts"], key=sym.encode(), value=alert)
            flag = "  *** ALERT ***"

        p50 = np.percentile(list(latencies), 50) if len(latencies) > 10 else 0
        p95 = np.percentile(list(latencies), 95) if len(latencies) > 10 else 0
        print(f"{bar_ts.strftime('%H:%M:%S')} {sym:12s} "
              f"score={score:.4f} e2e={e2e_ms:.0f}ms "
              f"(p50={p50:.0f}ms p95={p95:.0f}ms){flag}")


if __name__ == "__main__":
    main()
