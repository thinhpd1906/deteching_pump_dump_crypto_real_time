"""Research-protocol guards (Mac plan, task A2).

  * config overrides  : `--set key=value` with the LOCKED design decisions
                        (CLAUDE.md section 4 / 19.2 rule 2) rejected
  * config hash       : identifies the effective experiment configuration
  * test ledger       : artifacts/test_ledger.jsonl, append-only record of
                        every run that touches the test split; a repeated
                        (mode, tag, config hash) is refused unless an explicit
                        override reason is given
  * atomic writes     : JSON and checkpoints go to *.tmp then os.replace, so
                        a crash or kill -9 never leaves a truncated file

Search runs (val-only) never touch the ledger because they never load test.
"""
from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import os
import subprocess

import yaml

# Locked by Bryan (CLAUDE.md s4, s19.2 rule 2). Whole sections are prefixes.
LOCKED_PREFIXES = (
    "paths.",      # data locations (and the scaler.json they hold)
    "data.",       # data acquisition
    "bars.",       # 5 s bars, feature rolling windows
    "windows.",    # T=120, stride=12, label window, +-3h buffer, neg ratios
    "split.",      # chronological 70/15/15
    "cleaning.",   # fixed per mode; changing it needs a dataset rebuild
    "pipeline.",   # serving, not research
    "automation.", # search settings, not a training knob
)
LOCKED_KEYS = {
    "model.d_model", "model.n_heads", "model.n_layers",   # AT size 64/4/3
    "model.mask_ratio",                                   # 25% masking
    "model.lambda_k",                                     # k = 3
}
# Keys read only by pretrain.py; they select the pretrained checkpoint.
PRETRAIN_KEYS = ("train.pretrain_epochs", "train.pretrain_lr")
# Keys that do not change the experiment (excluded from hash and tag).
NON_SEMANTIC_KEYS = {"train.device"}
HASH_EXCLUDE_SECTIONS = ("paths", "pipeline", "automation")


# ------------------------------------------------------------- overrides
def parse_set(items: list[str] | None) -> dict:
    """['train.lr=3e-4', 'model.dropout=0.2'] -> {'train.lr': 0.0003, ...}."""
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--set expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        k = k.strip()
        if not k:
            raise SystemExit(f"--set has an empty key: {it!r}")
        out[k] = _parse_value(v)
    return out


def _parse_value(v: str):
    """int / float first (YAML 1.1 reads '3e-4' as a STRING), then YAML."""
    s = v.strip()
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return yaml.safe_load(s)


def check_locked(key: str) -> None:
    if key in LOCKED_KEYS or any(key.startswith(p) or key + "." == p
                                 for p in LOCKED_PREFIXES):
        raise SystemExit(
            f"--set {key}: this is a LOCKED design decision (CLAUDE.md s4). "
            f"Changing it needs Bryan's explicit approval.")


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    """Return a copy of cfg with dotted-key overrides applied.

    Only keys that already exist in the config may be overridden, so a typo
    (or a knob the code does not read yet) cannot silently do nothing.
    """
    cfg = copy.deepcopy(cfg)
    for key, val in overrides.items():
        check_locked(key)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if not isinstance(node, dict) or p not in node:
                raise SystemExit(f"--set {key}: unknown config section {p!r}")
            node = node[p]
        if not isinstance(node, dict) or parts[-1] not in node:
            raise SystemExit(
                f"--set {key}: not in config.yaml. Add it there (with its "
                f"default) together with the code that reads it.")
        node[parts[-1]] = val
    return cfg


def semantic_overrides(overrides: dict) -> dict:
    return {k: v for k, v in sorted(overrides.items())
            if k not in NON_SEMANTIC_KEYS}


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str)
                          .encode()).hexdigest()


def override_suffix(overrides: dict) -> str:
    """Tag suffix: '' for the default config (keeps protocol tags stable)."""
    sem = semantic_overrides(overrides)
    return "" if not sem else "_o" + _h(sem)[:8]


def pretrain_overrides(overrides: dict) -> dict:
    return {k: v for k, v in sorted(overrides.items()) if k in PRETRAIN_KEYS}


def pretrained_ckpt_path(art_mode_dir: str, overrides: dict) -> str:
    """Checkpoint a fine-tune with these overrides must start from.

    No pretrain keys overridden -> the canonical at_pretrained.pt; otherwise
    the cached variant written by `pretrain --set <same pretrain keys>`.
    """
    sfx = override_suffix(pretrain_overrides(overrides))
    if not sfx:
        return os.path.join(art_mode_dir, "at_pretrained.pt")
    return os.path.join(art_mode_dir, "pretrain_cache", f"at_pretrained{sfx}.pt")


# Knobs added in A4 whose default reproduces the behaviour before they
# existed. At that value they are left out of the hash, so the default
# config keeps the hash already recorded in the test ledger.
LEGACY_DEFAULTS = {
    "train.pretrain_lr": 1.0e-4, "train.weight_decay": 0.01,
    "train.pos_weight_scale": 1.0, "train.label_smoothing": 0.0,
    "train.finetune_mode": "full", "train.layerwise_decay": 1.0,
    "model.cnn_dropout": 0.2,
}


def config_hash(cfg: dict) -> str:
    """Hash of the effective experiment config (paths/serving/device excluded)."""
    c = {k: v for k, v in cfg.items() if k not in HASH_EXCLUDE_SECTIONS}
    c = copy.deepcopy(c)
    drop = set(NON_SEMANTIC_KEYS)
    for key, dflt in LEGACY_DEFAULTS.items():
        sec, name = key.split(".", 1)
        if isinstance(c.get(sec), dict) and c[sec].get(name, dflt) == dflt:
            drop.add(key)
    for key in drop:
        sec, name = key.split(".", 1)
        if isinstance(c.get(sec), dict):
            c[sec].pop(name, None)
    return _h(c)[:16]


# ----------------------------------------------------------- atomic files
def atomic_write_json(obj, path: str, indent: int = 2) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=indent)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_torch_save(obj, path: str) -> None:
    import torch
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def file_sha256(path: str | None) -> str | None:
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def git_sha() -> str | None:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--",
                                 "src", "config", "scripts"],
                                stderr=subprocess.DEVNULL) != 0
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None          # e.g. Kaggle: no .git


# ------------------------------------------------------------- the ledger
def ledger_path(cfg: dict) -> str:
    return os.path.join(cfg["paths"]["artifacts_dir"], "test_ledger.jsonl")


def read_ledger(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def ledger_check(cfg: dict, clean: str, tag: str, chash: str,
                 override_reason: str | None = None) -> None:
    """Refuse a test evaluation already recorded for (mode, tag, config hash)."""
    prev = [e for e in read_ledger(ledger_path(cfg))
            if e.get("clean") == clean and e.get("tag") == tag
            and e.get("config_hash") == chash]
    if prev and not override_reason:
        when = prev[-1].get("time")
        raise SystemExit(
            f"TEST LEDGER: [{clean}/{tag}] with config {chash} already "
            f"touched the test set ({len(prev)}x, last {when}). The test set "
            f"is evaluated once per frozen config. If this rerun is really "
            f"needed, pass --ledger-override \"<reason>\".")


def ledger_append(cfg: dict, *, clean: str, tag: str, chash: str,
                  overrides: dict, script: str, checkpoint: str | None = None,
                  override_reason: str | None = None, **extra) -> dict:
    """Append one entry. Call immediately BEFORE test data is loaded."""
    ledger_check(cfg, clean, tag, chash, override_reason)
    entry = {
        "time": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "clean": clean, "tag": tag, "script": script,
        "git_sha": git_sha(), "config_hash": chash,
        "overrides": semantic_overrides(overrides),
        "checkpoint": checkpoint, "checkpoint_sha256": file_sha256(checkpoint),
        "override_reason": override_reason, **extra,
    }
    path = ledger_path(cfg)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    line = json.dumps(entry, sort_keys=True) + "\n"
    with open(path, "a") as fh:
        try:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())
    return entry


def add_protocol_args(ap, with_set: bool = True) -> None:
    ap.add_argument("--search", action="store_true",
                    help="val-only mode: never loads test; writes "
                         "artifacts/<mode>/search/<tag>_val.json")
    if with_set:
        ap.add_argument("--set", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="override a config value (repeatable); locked "
                             "design decisions are rejected")
    ap.add_argument("--ledger-override", default=None, metavar="REASON",
                    help="allow a repeated test evaluation; reason is logged")
