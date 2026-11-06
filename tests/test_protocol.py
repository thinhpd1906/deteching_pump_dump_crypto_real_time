"""Research-protocol guards (task A2): val-only search mode, --set overrides,
test ledger, atomic writes. Runs on a tiny synthetic dataset, CPU only."""
from __future__ import annotations

import builtins
import json
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from src.training import protocol as P

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


# ------------------------------------------------------- search never opens test
@pytest.mark.parametrize("arch", ["at", "cnnbilstm"])
def test_finetune_search_never_opens_test(tiny, spy_test_open, arch):
    from src.training import finetune
    finetune.main(["--config", tiny["cfg_path"], "--arch", arch, "--seed", "0",
                   "--epochs", "1", "--search", "--set", "train.lr=0.0003"])
    assert spy_test_open == []
    out = list((tiny["art"] / "adaptive" / "search").glob("*_val.json"))
    assert len(out) == 1
    res = json.load(open(out[0]))
    assert res["split"] == "val"
    assert res["overrides"] == {"train.lr": 0.0003}
    assert res["tag"].startswith(f"{arch}_scratch_s0_o")
    assert not (tiny["art"] / "test_ledger.jsonl").exists()
    assert not list((tiny["art"] / "adaptive").glob("*_test.json"))


def test_baselines_search_never_opens_test(tiny, spy_test_open, monkeypatch):
    # xgboost + torch (already imported by other tests) in ONE process
    # segfault on macOS (two libomp runtimes). The real baselines script never
    # imports torch, so here B6 is simply skipped via its ImportError path.
    import sys
    monkeypatch.setitem(sys.modules, "xgboost", None)
    from src.models import baselines
    baselines.main(["--config", tiny["cfg_path"], "--search"])
    assert spy_test_open == []
    out = sorted(p.name for p in (tiny["art"] / "adaptive" / "search").glob("*_val.json"))
    assert "b5_logreg_val.json" in out and "b1_rule_val.json" in out
    assert not (tiny["art"] / "test_ledger.jsonl").exists()
    assert not list((tiny["art"] / "adaptive").glob("*_test.json"))


# ------------------------------------------------- default behaviour + ledger
def test_finetune_default_writes_test_and_ledger_then_refuses(tiny):
    from src.training import finetune
    argv = ["--config", tiny["cfg_path"], "--arch", "cnnbilstm",
            "--seed", "0", "--epochs", "1"]
    finetune.main(argv)
    out = tiny["art"] / "adaptive" / "cnnbilstm_scratch_s0_test.json"
    assert out.exists()                       # tag naming unchanged
    led = P.read_ledger(str(tiny["art"] / "test_ledger.jsonl"))
    assert len(led) == 1
    e = led[0]
    assert e["tag"] == "cnnbilstm_scratch_s0" and e["clean"] == "adaptive"
    assert e["config_hash"] == P.config_hash(tiny["cfg"])
    assert e["checkpoint_sha256"] and len(e["checkpoint_sha256"]) == 64
    assert json.load(open(out))["config_hash"] == e["config_hash"]
    # per-window test scores saved at the single test touch
    from src.training.common import load_scores, load_split
    _, yte, _, pte = load_split(str(tiny["dd"]), "test")
    sc, thr = load_scores(str(tiny["art"]), "adaptive",
                          "cnnbilstm_scratch_s0", yte, pte)
    assert len(sc) == len(yte)
    assert thr == pytest.approx(json.load(open(out))["thr"], abs=1e-6)
    with pytest.raises(ValueError):
        load_scores(str(tiny["art"]), "adaptive", "cnnbilstm_scratch_s0",
                    1 - yte, pte)

    # result removed (e.g. someone wants a "better" number): ledger refuses,
    # before any training happens
    out.unlink()
    with pytest.raises(SystemExit, match="TEST LEDGER"):
        finetune.main(argv)
    assert not out.exists()

    # explicit, logged override
    finetune.main(argv + ["--ledger-override", "unit test"])
    led = P.read_ledger(str(tiny["art"] / "test_ledger.jsonl"))
    assert len(led) == 2 and led[1]["override_reason"] == "unit test"


def test_ledger_unit(tmp_path):
    cfg = {"paths": {"artifacts_dir": str(tmp_path)}}
    P.ledger_append(cfg, clean="adaptive", tag="x", chash="h1", overrides={},
                    script="t")
    with pytest.raises(SystemExit):
        P.ledger_append(cfg, clean="adaptive", tag="x", chash="h1",
                        overrides={}, script="t")
    # different config hash or mode is a different experiment
    P.ledger_append(cfg, clean="adaptive", tag="x", chash="h2", overrides={},
                    script="t")
    P.ledger_append(cfg, clean="none", tag="x", chash="h1", overrides={},
                    script="t")
    assert len(P.read_ledger(P.ledger_path(cfg))) == 3


# ------------------------------------------------------------- --set guards
@pytest.mark.parametrize("key", [
    "windows.window_bars", "windows.pre_minutes", "windows.buffer_hours",
    "bars.bar_seconds", "split.train_frac", "model.d_model", "model.n_heads",
    "model.n_layers", "model.mask_ratio", "model.lambda_k",
    "paths.processed_dir", "windows"])
def test_locked_keys_rejected(key):
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "config.yaml")))
    with pytest.raises(SystemExit, match="LOCKED"):
        P.apply_overrides(cfg, {key: 1})


def test_unknown_key_rejected_and_free_key_applied():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "config.yaml")))
    with pytest.raises(SystemExit, match="not in config"):
        P.apply_overrides(cfg, {"train.lrr": 1e-3})
    new = P.apply_overrides(cfg, P.parse_set(["train.lr=3e-4",
                                              "model.dropout=0.2"]))
    assert new["train"]["lr"] == 3e-4 and new["model"]["dropout"] == 0.2
    assert cfg["train"]["lr"] != 3e-4                     # original untouched
    assert P.config_hash(new) != P.config_hash(cfg)


def test_device_is_not_part_of_the_experiment():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "config.yaml")))
    ov = {"train.device": "cpu"}
    assert P.override_suffix(ov) == ""
    assert P.config_hash(P.apply_overrides(cfg, ov)) == P.config_hash(cfg)
    assert P.override_suffix({"train.lr": 1e-3}).startswith("_o")


def test_atomic_write_json_leaves_no_tmp(tmp_path):
    p = tmp_path / "a" / "r.json"
    P.atomic_write_json({"x": 1}, str(p))
    assert json.load(open(p)) == {"x": 1}
    assert not (tmp_path / "a" / "r.json.tmp").exists()
