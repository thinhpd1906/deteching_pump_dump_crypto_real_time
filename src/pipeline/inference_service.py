"""Pipeline Tier 3: Real-time inference service.

bars (Kafka) -> rolling feature buffer -> SAME bars_to_features() as training
-> scaler -> model -> anomaly score -> alerts (Kafka) + latency measurement

FEATURE PARITY RULE: this file IMPORTS src.data.features.bars_to_features.
Never re-implement feature computation in the serving path.
This eliminates training-serving skew by construction.

Usage:
    python -m src.pipeline.inference_service --clean adaptive --model at_pre \
        [--device cpu] [--latency-csv artifacts/latency/samples.csv]

With --latency-csv every scored bar appends one row of timestamps (A5); see
scripts/measure_latency.py for the definitions.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import torch
import yaml

from src.training.common import get_device


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


BAR_COLS = ["open", "high", "low", "close", "volume",
            "n_trades", "taker_buy_vol", "quote_vol"]


class OnlineFeaturizer:
    """Rolling per-symbol bar buffer -> the model's (T, F) scaled window.

    Uses THE SAME normalize_bars / bars_to_features as training. The buffer
    keeps z_window + T + 32 bars, so once `warm` is True every row of the
    returned window has a full z_window history (offline parity holds).
    """

    def __init__(self, B: dict, T: int, mean: np.ndarray, std: np.ndarray):
        self.B, self.T = B, T
        self.mean, self.std = mean, std
        self.keep = B["z_window"] + T + 32
        self.buf: dict[str, pd.DataFrame] = defaultdict(pd.DataFrame)

    def warm(self, sym: str) -> bool:
        b = self.buf.get(sym)
        if b is None or not len(b):
            return False
        span = (b.index[-1] - b.index[0]).total_seconds() / self.B["bar_seconds"] + 1
        return span >= self.B["z_window"] + self.T

    def add(self, r: dict):
        """Add one streamed bar; return the scaled window or None (too short)."""
        from src.data.features import bars_to_features, normalize_bars
        sym = r["symbol"]
        row = pd.DataFrame([{c: float(r[c]) for c in BAR_COLS}],
                           index=[pd.Timestamp(r["bar_time"])])
        b = pd.concat([self.buf[sym], row]).sort_index()
        b = b[~b.index.duplicated(keep="last")]
        # keep the last `keep` bars of the regridded time span
        cutoff = b.index[-1] - pd.Timedelta(seconds=self.B["bar_seconds"] * (self.keep - 1))
        self.buf[sym] = b = b[b.index >= cutoff]

        bars = normalize_bars(b, self.B["bar_seconds"])
        bars["vwap"] = np.where(
            bars["volume"] > 0,
            bars["quote_vol"] / bars["volume"].replace(0, np.nan),
            bars["close"])
        bars["vwap"] = bars["vwap"].fillna(bars["close"])
        if len(bars) < self.T:
            return None
        feats = bars_to_features(bars, self.B["z_window"], self.B["vol_window"])
        return (feats.tail(self.T).to_numpy(np.float32) - self.mean) / self.std


LAT_COLS = ["symbol", "bar_time_ms", "bar_close_ms", "first_T", "last_T",
            "emit_ms", "recv_ms", "feat_done_ms", "done_ms", "infer_ms",
            "score", "alert", "warm"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--clean", default="adaptive")
    ap.add_argument("--model", default="at_pre", help="artifact tag name")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--device", default=None,
                    help="override train.device (auto|cuda|mps|cpu)")
    ap.add_argument("--latency-csv", default=None,
                    help="append one timing row per scored bar")
    ap.add_argument("--group-id", default="pnd-inference")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    P, B, W = cfg["pipeline"], cfg["bars"], cfg["windows"]
    dd = os.path.join(cfg["paths"]["processed_dir"], args.clean)
    art = os.path.join(cfg["paths"]["artifacts_dir"], args.clean)
    if args.device:
        cfg["train"]["device"] = args.device
    dev = get_device(cfg)

    model, ck = load_model(art, args.model, dev)
    thr = cfg["pipeline"].get("alert_threshold") or ck.get("thr", 0.5)
    sc = json.load(open(os.path.join(dd, "scaler.json")))
    mean = np.asarray(sc["mean"], np.float32)
    std_ = np.asarray(sc["std"], np.float32)
    T = ck["win"]
    feat = OnlineFeaturizer(B, T, mean, std_)
    bar_ms = B["bar_seconds"] * 1000
    lat_fh = lat_w = None
    if args.latency_csv:
        os.makedirs(os.path.dirname(args.latency_csv) or ".", exist_ok=True)
        new = not os.path.exists(args.latency_csv)
        lat_fh = open(args.latency_csv, "a", newline="")
        lat_w = csv.writer(lat_fh)
        if new:
            lat_w.writerow(LAT_COLS)

    try:
        from kafka import KafkaConsumer, KafkaProducer
    except ImportError:
        print("ERROR: pip install kafka-python"); return

    consumer = KafkaConsumer(
        P["topic_bars"],
        bootstrap_servers=P["kafka_bootstrap"],
        value_deserializer=lambda b: json.loads(b),
        auto_offset_reset="latest",
        group_id=args.group_id,
    )
    producer = KafkaProducer(
        bootstrap_servers=P["kafka_bootstrap"],
        value_serializer=lambda v: json.dumps(v).encode(),
        linger_ms=0,
    )
    latencies: deque = deque(maxlen=1000)

    syms = args.symbols or P.get("live_symbols", [])
    print(f"Inference service: model={args.model} thr={thr:.4f} "
          f"win={T} dev={dev}")
    print(f"  Watching: {syms or 'all symbols'}")
    print(f"  Consuming topic: {P['topic_bars']}")

    with torch.no_grad():                       # warm-up (first call is slow)
        model(torch.zeros(1, T, len(mean), device=dev))

    for msg in consumer:
        recv_ms = time.time() * 1000.0
        r = msg.value
        sym = r["symbol"]
        if syms and sym not in syms:
            continue

        x = feat.add(r)
        if x is None:
            continue
        feat_done_ms = time.time() * 1000.0

        t0 = time.perf_counter()
        with torch.no_grad():
            xb = torch.from_numpy(x[None]).to(dev)
            out = model(xb)
            logit = out[0] if isinstance(out, tuple) else out
            score = float(torch.sigmoid(logit)[0])
        infer_ms = (time.perf_counter() - t0) * 1000

        bar_ts = pd.Timestamp(r["bar_time"])
        bar_time_ms = bar_ts.value // 1_000_000
        is_alert = score >= thr
        if is_alert:
            alert = {
                "symbol": sym,
                "bar_time": r["bar_time"],
                "score": round(score, 4),
                "threshold": round(thr, 4),
                "model": args.model,
                "infer_ms": round(infer_ms, 1),
            }
            producer.send(P["topic_alerts"], key=sym.encode(), value=alert)
            producer.flush()
        # decision time: score computed (and alert handed to Kafka if any)
        done_ms = time.time() * 1000.0
        bar_close_ms = bar_time_ms + bar_ms
        e2e_ms = done_ms - bar_close_ms
        latencies.append(e2e_ms)
        if lat_w is not None:
            lat_w.writerow([sym, bar_time_ms, bar_close_ms, r.get("first_T"),
                            r.get("last_T"), r.get("emit_ms"), round(recv_ms, 1),
                            round(feat_done_ms, 1), round(done_ms, 1),
                            round(infer_ms, 3), round(score, 6), int(is_alert),
                            int(feat.warm(sym))])
            lat_fh.flush()

        p50 = np.percentile(list(latencies), 50) if len(latencies) > 10 else 0
        p95 = np.percentile(list(latencies), 95) if len(latencies) > 10 else 0
        print(f"{bar_ts.strftime('%H:%M:%S')} {sym:12s} "
              f"score={score:.4f} close->done={e2e_ms:.0f}ms "
              f"(p50={p50:.0f}ms p95={p95:.0f}ms)"
              f"{'  *** ALERT ***' if is_alert else ''}", flush=True)


if __name__ == "__main__":
    main()
