"""A4: search knobs, pretrained-variant cache, and the resumable search driver."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import pytest
import torch
import yaml

from src.training import protocol as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cfg_with(tiny, **search):
    cfg = dict(tiny["cfg"])
    cfg["train"] = {**cfg["train"], "finetune_epochs": 1}
    S = {**cfg["automation"]["search"], **search}
    cfg["automation"] = {**cfg["automation"], "search": S}
    with open(tiny["cfg_path"], "w") as fh:
        yaml.safe_dump(cfg, fh)
    return cfg


# ------------------------------------------------------------- param groups
def _at_classifier():
    from src.models.anomaly_transformer import ATClassifier, AnomalyTransformer
    return ATClassifier(AnomalyTransformer(120, 10, 64, 4, 3, 0.1), 64, 0.1)


def test_param_groups_modes():
    from src.training.finetune import param_groups
    M = {"n_layers": 3}
    m = _at_classifier()
    g = param_groups(m, "at", True, {"lr": 1e-3, "finetune_mode": "full"}, M)
    assert len(g) == 1 and len(g[0]["params"]) == len(list(m.parameters()))

    g = param_groups(m, "at", True, {"lr": 1e-3, "finetune_mode": "layerwise",
                                     "layerwise_decay": 0.5}, M)
    lrs = [x["lr"] for x in g]
    assert lrs[0] == 1e-3 and lrs == sorted(lrs, reverse=True) and len(lrs) == 6
    assert sum(len(x["params"]) for x in g) == len(list(m.parameters()))

    m = _at_classifier()
    g = param_groups(m, "at", True, {"lr": 1e-3, "finetune_mode": "frozen"}, M)
    assert all(not p.requires_grad for n, p in m.named_parameters()
               if n.startswith("encoder."))
    assert sum(p.numel() for x in g for p in x["params"]) == \
        sum(p.numel() for p in m.head.parameters())

    with pytest.raises(SystemExit):
        param_groups(_at_classifier(), "at", False,
                     {"lr": 1e-3, "finetune_mode": "frozen"}, M)


def test_default_hash_unchanged_by_new_knobs():
    """New knobs at their legacy default must not change the config hash."""
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "config.yaml")))
    legacy = {k: dict(v) if isinstance(v, dict) else v for k, v in cfg.items()}
    for key in P.LEGACY_DEFAULTS:
        sec, name = key.split(".", 1)
        legacy[sec].pop(name, None)
    legacy.pop("automation", None)
    assert P.config_hash(cfg) == P.config_hash(legacy)
    assert P.config_hash(P.apply_overrides(cfg, {"train.weight_decay": 0.02})) \
        != P.config_hash(cfg)


# --------------------------------------------------- pretrained variant cache
def test_pretrain_variant_cache_and_finetune(tiny):
    from src.training import finetune, pretrain
    cfgp = tiny["cfg_path"]
    _cfg_with(tiny)
    with pytest.raises(SystemExit, match="accepts only"):
        pretrain.main(["--config", cfgp, "--epochs", "1", "--set", "train.lr=0.01"])

    pretrain.main(["--config", cfgp, "--epochs", "1"])            # canonical
    canon = tiny["art"] / "adaptive" / "at_pretrained.pt"
    assert canon.exists()
    pretrain.main(["--config", cfgp, "--set", "train.pretrain_epochs=2"])
    var = P.pretrained_ckpt_path(str(tiny["art"] / "adaptive"),
                                 {"train.pretrain_epochs": 2})
    assert os.path.exists(var) and var != str(canon)
    assert torch.load(var, map_location="cpu")["epoch"] == 2

    # fine-tune from the variant, layer-wise lr, val only
    finetune.main(["--config", cfgp, "--arch", "at", "--pretrained", "--seed", "0",
                   "--search", "--set", "train.pretrain_epochs=2",
                   "--set", "train.finetune_mode=layerwise",
                   "--set", "train.layerwise_decay=0.8"])
    out = list((tiny["art"] / "adaptive" / "search").glob("at_pre_s0_o*_val.json"))
    assert len(out) == 1
    res = json.load(open(out[0]))
    assert res["overrides"]["train.pretrain_epochs"] == 2

    # a variant that was never built is an error, not a silent canonical load
    with pytest.raises(SystemExit, match="not found"):
        finetune.main(["--config", cfgp, "--arch", "at", "--pretrained",
                       "--seed", "0", "--search",
                       "--set", "train.pretrain_epochs=7"])


# ------------------------------------------------------------ search driver
def _search(cfgp, *extra, wait=True):
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "search.py"),
           "--config", cfgp, *extra]
    env = dict(os.environ, PYTHONPATH=ROOT)
    if wait:
        return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                              text=True, timeout=900)
    return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


def _rows(art):
    p = art / "search" / "trials.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def test_search_end_to_end_val_only(tiny):
    from src.training import pretrain
    _cfg_with(tiny, budget_trials_per_arm=2, confirm_top_k=1,
              stop_if_no_improve=0)
    pretrain.main(["--config", tiny["cfg_path"], "--epochs", "1"])
    (tiny["dd"] / "test.npz").unlink()          # any test access would crash
    r = _search(tiny["cfg_path"])
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    rows = _rows(tiny["art"])
    done = [x for x in rows if x["phase"] == "search" and x["state"] == "COMPLETE"]
    counts = {a: sum(x["arm"] == a for x in done)
              for a in ("at_pre", "at_scratch", "cnnbilstm")}
    assert counts == {"at_pre": 2, "at_scratch": 2, "cnnbilstm": 2}   # equal budget
    conf = [x for x in rows if x["phase"] == "confirm"]
    assert {(x["arm"], x["seed"]) for x in conf} >= {("cnnbilstm", 1), ("cnnbilstm", 2)}
    lb = (tiny["art"] / "search" / "leaderboard.md").read_text()
    assert "at_pre" in lb and "cnnbilstm" in lb
    assert not (tiny["art"] / "test_ledger.jsonl").exists()
    assert not list((tiny["art"] / "adaptive").glob("*_test.json"))


def test_search_survives_kill9(tiny):
    _cfg_with(tiny, arms=["cnnbilstm"], budget_trials_per_arm=3,
              confirm_top_k=1, stop_if_no_improve=0)
    art = tiny["art"]
    pid_file = art / "search" / "running.pid"
    p = _search(tiny["cfg_path"], "--phase", "search", wait=False)
    # wait until one trial completed and the next one is running, then kill -9
    t0 = time.time()
    while time.time() - t0 < 600:
        n = sum(x.get("state") == "COMPLETE" for x in _rows(art))
        if n >= 1 and pid_file.exists():
            break
        time.sleep(0.5)
    else:
        p.kill()
        pytest.fail("search did not reach the second trial")
    os.kill(p.pid, signal.SIGKILL)
    p.wait()
    before = [x for x in _rows(art) if x.get("state") == "COMPLETE"]

    r = _search(tiny["cfg_path"], "--phase", "search")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    after = [x for x in _rows(art) if x.get("state") == "COMPLETE"]
    assert len(after) == 3                                   # budget reached
    assert after[:len(before)] == before                     # nothing lost
    assert len({x["trial"] for x in after}) == 3             # no duplicates
    assert "re-queued" in r.stdout
    assert not pid_file.exists()
