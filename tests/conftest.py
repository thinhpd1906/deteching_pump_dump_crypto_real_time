"""Shared fixtures: a tiny synthetic processed dataset + config (CPU)."""
from __future__ import annotations

import builtins
import json
import os

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS = ["AAABTC", "BBBBTC", "CCCBTC"]
T, F = 120, 10


def _split(rng, day0: str, n_per_pair: int = 40):
    """Windows every 5 min; one pump per pair at window 30."""
    X, y, end, pair, pumps = [], [], [], [], []
    t0 = pd.Timestamp(day0, tz="UTC")
    for k, p in enumerate(PAIRS):
        base = t0 + pd.Timedelta(hours=6 * k)
        pump = base + pd.Timedelta(minutes=5 * 30)
        pumps.append((p, pump))
        for i in range(n_per_pair):
            e = base + pd.Timedelta(minutes=5 * i)
            pos = pump - pd.Timedelta(minutes=60) <= e <= pump + pd.Timedelta(minutes=5)
            x = rng.normal(size=(T, F)).astype(np.float32)
            if pos:
                x[-24:] += 1.5
            X.append(x); y.append(int(pos)); end.append(e.value); pair.append(p)
    return (np.stack(X), np.array(y, np.int64), np.array(end, np.int64),
            np.array(pair, "<U8"), pumps)


@pytest.fixture()
def tiny(tmp_path):
    """Processed dir + labels + config pointing into tmp_path."""
    rng = np.random.default_rng(0)
    dd = tmp_path / "processed" / "adaptive"
    dd.mkdir(parents=True)
    rows = []
    for split, day in [("train", "2019-01-01"), ("val", "2019-02-01"),
                       ("test", "2019-03-01")]:
        X, y, e, p, pumps = _split(rng, day, 60 if split == "train" else 40)
        np.savez(dd / f"{split}.npz", X=X, y=y, end_ns=e, pair=p)
        if split == "train":
            np.savez(dd / "unlabeled_pretrain.npz", X=X[y == 0])
        rows += [{"pair": a, "symbol": a[:-3], "group": "g",
                  "pump_time": str(t), "source": "test"} for a, t in pumps]
    (dd / "scaler.json").write_text(json.dumps(
        {"features": [f"f{i}" for i in range(F)],
         "mean": [0.0] * F, "std": [1.0] * F}))
    labels = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(labels, index=False)

    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "config.yaml")))
    cfg["paths"] = {"raw_dir": str(tmp_path / "raw"),
                    "labels_csv": str(labels),
                    "processed_dir": str(tmp_path / "processed"),
                    "artifacts_dir": str(tmp_path / "artifacts")}
    cfg["train"]["device"] = "cpu"
    cfg["train"]["batch_size"] = 32
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return {"cfg": cfg, "cfg_path": str(cfg_path),
            "art": tmp_path / "artifacts", "dd": dd}


@pytest.fixture()
def spy_test_open(monkeypatch):
    """Record every attempt to open a file named test.npz."""
    opened = []
    real_open, real_load = builtins.open, np.load

    def _open(file, *a, **k):
        if "test.npz" in str(file):
            opened.append(str(file))
        return real_open(file, *a, **k)

    def _load(file, *a, **k):
        if "test.npz" in str(file):
            opened.append(str(file))
        return real_load(file, *a, **k)

    monkeypatch.setattr(builtins, "open", _open)
    monkeypatch.setattr(np, "load", _load)
    return opened


