"""A5 / C3: live end-to-end latency of the streaming pipeline.

Runs the real pipeline for `pipeline.latency.hours` on live Binance data:

    Binance WebSocket -> kafka_producer -> Kafka raw-trades
      -> python_bars (StreamingCleaner + 5 s bars) -> Kafka bars
      -> inference_service (shared features + scaler + model) -> Kafka alerts

and writes
    artifacts/latency/samples.csv   one row per scored bar (inference_service)
    artifacts/latency/summary.json  percentiles + run metadata
    artifacts/latency/logs/         stdout of every component

Definitions (every scored bar counts, alert or not: the decision for a bar
is produced whether or not it crosses the threshold, and alerts are rare):
  (a) bar-close -> decision = done - (bar_start + 5 s)
  (b) trade -> decision     = done - first_T  (oldest trade in the bar, so it
                              includes the full <= 5 s bar wait; conservative)
                              and done - last_T (newest trade)
  `done` is local wall clock after the score is computed and, for an alert,
  handed to Kafka. bar times and trade times are exchange (Binance) clock.
  Local clock is corrected to NTP with the offset measured by `sntp` at start
  and end (mean); the Binance server-time offset is recorded as a cross-check.
  Stage breakdown (local clock only, no offset involved): Kafka transit
  bars->inference, feature computation, model forward.
  `warm` rows are those with a full z_window feature history; the headline
  uses all rows (latency does not depend on feature values) and warm-only is
  reported alongside.

Usage:
    python scripts/measure_latency.py                 # full live run
    python scripts/measure_latency.py --hours 0.05    # smoke test
    python scripts/measure_latency.py --summarize     # recompute summary only
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.training.protocol import atomic_write_json, git_sha  # noqa: E402

OUT = os.path.join(ROOT, "artifacts", "latency")


# ------------------------------------------------------------------ clocks
def ntp_offset_ms(server: str) -> dict:
    """sntp prints '+0.081132 +/- 0.103023 server ip': NTP minus local, s."""
    try:
        out = subprocess.run(["sntp", server], capture_output=True, text=True,
                             timeout=30).stdout
        m = re.search(r"([+-]\d+\.\d+)\s+\+/-\s+(\d+\.\d+)", out)
        if m:
            return {"offset_ms": float(m.group(1)) * 1000,
                    "err_ms": float(m.group(2)) * 1000, "server": server,
                    "time": dt.datetime.now(dt.timezone.utc).isoformat()}
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {"offset_ms": None, "err_ms": None, "server": server}


def binance_offset_ms(n: int = 7) -> dict:
    """Binance server time minus local clock, from the min-RTT sample."""
    best = None
    for _ in range(n):
        try:
            t0 = time.time() * 1000
            with urllib.request.urlopen("https://api.binance.com/api/v3/time",
                                        timeout=10) as r:
                srv = json.load(r)["serverTime"]
            t1 = time.time() * 1000
            rtt = t1 - t0
            if best is None or rtt < best[1]:
                best = (srv - (t0 + t1) / 2, rtt)
        except Exception:
            pass
        time.sleep(0.2)
    return ({"offset_ms": round(best[0], 1), "rtt_ms": round(best[1], 1)}
            if best else {"offset_ms": None, "rtt_ms": None})


# ---------------------------------------------------------------- summary
def _pct(v: np.ndarray) -> dict:
    v = v[np.isfinite(v)]
    if not len(v):
        return {"count": 0}
    return {"count": int(len(v)),
            "p50": round(float(np.percentile(v, 50)), 1),
            "p95": round(float(np.percentile(v, 95)), 1),
            "p99": round(float(np.percentile(v, 99)), 1),
            "mean": round(float(v.mean()), 1),
            "max": round(float(v.max()), 1)}


def summarize(meta: dict) -> dict:
    df = pd.read_csv(os.path.join(OUT, "samples.csv"))
    offs = [o["offset_ms"] for o in (meta.get("ntp_start", {}),
                                    meta.get("ntp_end", {}))
            if o.get("offset_ms") is not None]
    ntp = float(np.mean(offs)) if offs else 0.0
    done = df["done_ms"] + ntp                     # local -> NTP time

    def block(d: pd.DataFrame, done_s: pd.Series) -> dict:
        return {
            "a_bar_close_to_decision_ms": _pct((done_s - d["bar_close_ms"]).to_numpy(float)),
            "b_first_trade_to_decision_ms": _pct((done_s - d["first_T"]).to_numpy(float)),
            "b_last_trade_to_decision_ms": _pct((done_s - d["last_T"]).to_numpy(float)),
            "stages_ms": {
                "bar_close_to_bar_emitted": _pct((d["emit_ms"] + ntp - d["bar_close_ms"]).to_numpy(float)),
                "kafka_bars_to_inference": _pct((d["recv_ms"] - d["emit_ms"]).to_numpy(float)),
                "features": _pct((d["feat_done_ms"] - d["recv_ms"]).to_numpy(float)),
                "model_forward": _pct(d["infer_ms"].to_numpy(float)),
                "features_to_done": _pct((d["done_ms"] - d["feat_done_ms"]).to_numpy(float)),
            },
        }

    warm = df["warm"] == 1
    a95 = _pct((done - df["bar_close_ms"]).to_numpy(float)).get("p95")
    span_h = (df["done_ms"].max() - df["done_ms"].min()) / 3.6e6 if len(df) else 0
    res = {
        **meta,
        "ntp_offset_used_ms": round(ntp, 1),
        "measured_hours": round(float(span_h), 3),
        "n_bars_scored": int(len(df)),
        "n_alerts": int(df["alert"].sum()),
        "symbols": sorted(df["symbol"].unique().tolist()),
        "n_symbols": int(df["symbol"].nunique()),
        "all": block(df, done),
        "warm_only": block(df[warm], done[warm]),
        "target": "p95 (a) < 5000 ms",
        "pass": bool(a95 is not None and a95 < 5000),
    }
    atomic_write_json(res, os.path.join(OUT, "summary.json"))
    return res


def print_summary(res: dict) -> None:
    a = res["all"]
    print(f"\n=== LATENCY [{res.get('model')} on {res.get('device')}] "
          f"{res['measured_hours']} h, {res['n_bars_scored']} bars, "
          f"{res['n_symbols']} symbols ===")
    for k in ("a_bar_close_to_decision_ms", "b_first_trade_to_decision_ms",
              "b_last_trade_to_decision_ms"):
        v = a[k]
        print(f"  {k:32s} p50={v.get('p50')}  p95={v.get('p95')}  p99={v.get('p99')}")
    for k, v in a["stages_ms"].items():
        print(f"    stage {k:26s} p50={v.get('p50')}  p95={v.get('p95')}")
    print(f"  NTP offset used {res['ntp_offset_used_ms']} ms; "
          f"target {res['target']}: {'PASS' if res['pass'] else 'FAIL'}")


# --------------------------------------------------------------------- run
def wait_port(host: str, port: int, timeout: float = 120) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection((host, port), 2):
                return
        except OSError:
            time.sleep(2)
    raise SystemExit(f"Kafka not reachable on {host}:{port}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(ROOT, args.config)))
    P, L = cfg["pipeline"], cfg["pipeline"]["latency"]
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    meta_path = os.path.join(OUT, "run_meta.json")

    if args.summarize:
        print_summary(summarize(json.load(open(meta_path))))
        return

    hours = args.hours if args.hours is not None else L["hours"]
    samples = os.path.join(OUT, "samples.csv")
    if os.path.exists(samples):                   # keep earlier runs
        os.replace(samples, samples.replace(
            ".csv", f"_{dt.datetime.now():%Y%m%d_%H%M%S}.csv"))

    subprocess.run(["docker", "compose", "up", "-d", "kafka"], cwd=ROOT,
                   check=True)
    host, port = P["kafka_bootstrap"].split(":")
    wait_port(host, int(port))

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {
        "run_id": run_id, "model": L["model"], "device": L["device"],
        "clean": "adaptive", "hours_planned": hours,
        "symbols_requested": L["symbols"],
        "git_sha": git_sha(), "python": platform.python_version(),
        "machine": platform.platform(), "chip": platform.processor(),
        "start": dt.datetime.now(dt.timezone.utc).isoformat(),
        "ntp_start": ntp_offset_ms(L["ntp_server"]),
        "binance_offset_start": binance_offset_ms(),
        "concurrent_jobs": subprocess.run(
            ["pgrep", "-fl", "scripts/search.py|src.training"],
            capture_output=True, text=True).stdout.strip().splitlines(),
    }
    atomic_write_json(meta, meta_path)
    exch_off = meta["binance_offset_start"]["offset_ms"] or 0.0

    py = sys.executable
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    cmds = {
        "inference": [py, "-m", "src.pipeline.inference_service",
                      "--config", args.config, "--model", L["model"],
                      "--device", L["device"], "--latency-csv", samples,
                      "--group-id", f"lat-inf-{run_id}",
                      "--symbols", *L["symbols"]],
        "bars": [py, "-m", "src.pipeline.python_bars", "--config", args.config,
                 "--clean", "adaptive", "--clock-offset-ms", str(exch_off),
                 "--group-id", f"lat-bars-{run_id}"],
        "producer": [py, "-m", "src.pipeline.kafka_producer",
                     "--config", args.config, "--symbols", *L["symbols"]],
    }
    procs, restarts = {}, {k: 0 for k in cmds}

    def start(name):
        log = open(os.path.join(OUT, "logs", f"{name}_{run_id}.log"), "a")
        procs[name] = subprocess.Popen(cmds[name], cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    for name in ("inference", "bars"):            # consumers first
        start(name)
    time.sleep(8)
    start("producer")
    t_end = time.time() + hours * 3600
    print(f"[latency] run {run_id}: {hours} h, {len(L['symbols'])} symbols, "
          f"model {L['model']} on {L['device']}; samples -> {samples}", flush=True)
    last_report = 0.0
    while time.time() < t_end and not stop["flag"]:
        time.sleep(5)
        for name, p in procs.items():
            if p.poll() is not None:
                restarts[name] += 1
                print(f"[latency] {name} exited ({p.returncode}); restart "
                      f"#{restarts[name]}", flush=True)
                start(name)
        if time.time() - last_report > 600 and os.path.exists(samples):
            n = max(sum(1 for _ in open(samples)) - 1, 0)
            print(f"[latency] {dt.datetime.now():%H:%M} {n} bars scored, "
                  f"{(t_end - time.time()) / 60:.0f} min left", flush=True)
            last_report = time.time()

    for name in ("producer", "bars", "inference"):
        procs[name].terminate()
    for p in procs.values():
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            p.kill()

    meta.update({
        "end": dt.datetime.now(dt.timezone.utc).isoformat(),
        "ntp_end": ntp_offset_ms(L["ntp_server"]),
        "binance_offset_end": binance_offset_ms(),
        "restarts": restarts,
        "load_avg_end": os.getloadavg(),
        "interrupted": stop["flag"],
    })
    atomic_write_json(meta, meta_path)
    if not os.path.exists(samples):
        raise SystemExit("no samples recorded (see artifacts/latency/logs/)")
    print_summary(summarize(meta))


if __name__ == "__main__":
    main()
