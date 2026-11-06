"""Shared utilities for every training / evaluation script.

Contains the EVALUATION PROTOCOL that the whole paper depends on:
  * threshold tuned on VALIDATION only (never 0.5, never on test)
  * NO point-adjustment (Kim et al., AAAI 2022) -- plain window-level scoring
  * event-level metrics (what a practitioner actually cares about)
  * bootstrap CIs resampled over SYMBOLS (not windows)
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)


# ------------------------------------------------------------------ device
def get_device(cfg: dict | None = None) -> str:
    """Pick the torch device from config `train.device` (auto|cuda|mps|cpu).

    auto = cuda if available, else Apple MPS, else cpu. An explicit request
    for an unavailable device raises instead of silently falling back.
    """
    import torch
    want = str(((cfg or {}).get("train", {}) or {}).get("device", "auto")).lower()
    has = {"cuda": torch.cuda.is_available(),
           "mps": getattr(torch.backends, "mps", None) is not None
           and torch.backends.mps.is_available(),
           "cpu": True}
    if want == "auto":
        return next(d for d in ("cuda", "mps", "cpu") if has[d])
    if want not in has:
        raise ValueError(f"train.device must be auto|cuda|mps|cpu, got {want!r}")
    if not has[want]:
        raise RuntimeError(f"train.device={want} requested but not available")
    return want


def make_generator(dev: str, seed: int):
    """Seeded torch.Generator on `dev`; falls back to CPU if `dev` lacks one."""
    import torch
    try:
        g = torch.Generator(device=dev).manual_seed(seed)
        torch.randperm(2, device=dev, generator=g)
        return torch.Generator(device=dev).manual_seed(seed)
    except (RuntimeError, TypeError):
        return torch.Generator(device="cpu").manual_seed(seed)


# ------------------------------------------------------------------ loading
def load_split(data_dir: str, split: str, scale: bool = True):
    """Load one split. Returns (X, y, end_ns, pair). X standardized if scale."""
    z = np.load(os.path.join(data_dir, f"{split}.npz"), allow_pickle=False)
    X = z["X"].astype(np.float32)
    y = z["y"].astype(np.int64)
    end_ns, pair = z["end_ns"], z["pair"]
    if scale:
        sc = json.load(open(os.path.join(data_dir, "scaler.json")))
        mean = np.asarray(sc["mean"], np.float32)
        std = np.asarray(sc["std"], np.float32)
        X = (X - mean) / std
    return X, y, end_ns, pair


def load_pretrain_pool(data_dir: str) -> np.ndarray:
    z = np.load(os.path.join(data_dir, "unlabeled_pretrain.npz"))
    X = z["X"].astype(np.float32)
    sc = json.load(open(os.path.join(data_dir, "scaler.json")))
    mean = np.asarray(sc["mean"], np.float32)
    std = np.asarray(sc["std"], np.float32)
    return (X - mean) / std


def group_events(y, end_ns, pair, gap_ns: int) -> np.ndarray:
    """Event id per positive window (-1 for negatives).

    Positive windows of one pump are contiguous in time on one pair; a new
    event starts when the pair changes or the time gap exceeds `gap_ns`.
    """
    ev = np.full(len(y), -1, dtype=np.int64)
    pos = np.where(y == 1)[0]
    if len(pos) == 0:
        return ev
    pos = pos[np.lexsort((end_ns[pos], pair[pos]))]
    new = np.r_[True, (pair[pos][1:] != pair[pos][:-1])
                | (np.diff(end_ns[pos]) > gap_ns)]
    ev[pos] = np.cumsum(new) - 1
    return ev


def subsample_labels(X, y, frac: float, seed: int, end_ns, pair, gap_ns: int):
    """Keep `frac` of the POSITIVE EVENTS (label-efficiency study).

    Sampling is by EVENT, not by window: each pump yields ~65 overlapping
    positive windows, so window-level sampling at 10% still exposes the model
    to every training event and is not real label scarcity.
    Negatives are kept in proportion so the class ratio stays roughly constant.
    """
    if frac >= 1.0:
        return X, y
    rng = np.random.default_rng(seed)
    ev = group_events(y, end_ns, pair, gap_ns)
    n_ev = int(ev.max()) + 1
    kept_ev = rng.choice(n_ev, max(int(round(n_ev * frac)), 1), replace=False)
    pos = np.where(np.isin(ev, kept_ev))[0]
    neg = np.where(y == 0)[0]
    n_neg = max(int(len(neg) * frac), 1)
    keep = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])
    rng.shuffle(keep)
    return X[keep], y[keep]


# --------------------------------------------------------------- thresholds
def tune_threshold(y_true, scores, grid: int = 300):
    """Pick the decision threshold maximizing F1 on VALIDATION. Never 0.5."""
    # Vectorised (same grid and same F1 as sklearn's f1_score with
    # zero_division=0): count TP/FP per threshold via sorted scores.
    lo, hi = np.percentile(scores, 0.5), np.percentile(scores, 99.9)
    ts = np.linspace(lo, hi, grid)
    y = np.asarray(y_true).astype(bool)
    pos_s = np.sort(scores[y])
    neg_s = np.sort(scores[~y])
    tp = len(pos_s) - np.searchsorted(pos_s, ts, side="left")   # scores >= t
    fp = len(neg_s) - np.searchsorted(neg_s, ts, side="left")
    fn = len(pos_s) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    i = int(np.argmax(f1))                       # first max, like the old loop
    return float(ts[i]), float(f1[i])


# ------------------------------------------------------------------ metrics
def metric_table(y_true, scores, thr) -> dict:
    """Window-level metrics. NO point adjustment (Kim et al. AAAI 2022)."""
    pred = scores >= thr
    out = {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
        out["pr_auc"] = float(average_precision_score(y_true, scores))
    except ValueError:
        out["roc_auc"] = out["pr_auc"] = float("nan")
    return {k: round(v, 4) for k, v in out.items()}


def event_metrics(y_true, scores, thr, end_ns, pair, labels_csv: str,
                  pre_minutes: int = 60, during_minutes: int = 5) -> dict:
    """Event-level metrics -- what an exchange compliance team actually needs.

    event_recall : fraction of pump EVENTS with >=1 alert inside their window
    fa_per_day   : false alarms per symbol-day (the operational cost)
    """
    pred = scores >= thr
    labels = pd.read_csv(labels_csv, parse_dates=["pump_time"])
    ends = pd.to_datetime(end_ns, utc=True)

    detected = total = 0
    for _, row in labels.iterrows():
        t = row["pump_time"]
        m = ((pair == row["pair"]) &
             (ends >= t - pd.Timedelta(minutes=pre_minutes)) &
             (ends <= t + pd.Timedelta(minutes=during_minutes)))
        if not m.any():
            continue                      # this event isn't in this split
        total += 1
        detected += bool(pred[m].any())

    # false alarms: positive predictions on negative windows, per symbol-day
    fp = pred & (y_true == 0)
    if len(ends):
        days = pd.DataFrame({"pair": pair, "day": ends.date})
        n_symbol_days = max(len(days.drop_duplicates()), 1)
    else:
        n_symbol_days = 1

    return {
        "event_recall": round(detected / max(total, 1), 4),
        "events_detected": int(detected),
        "events_total": int(total),
        "fa_per_symbol_day": round(float(fp.sum()) / n_symbol_days, 3),
    }


def bootstrap_ci(y_true, scores, thr, pair, n_boot: int = 1000,
                 seed: int = 0) -> dict:
    """95% CI for F1, resampled over SYMBOLS (proxy for events).

    Resampling windows would massively understate uncertainty because windows
    from the same event are highly correlated. Reviewers check this.
    """
    rng = np.random.default_rng(seed)
    symbols = np.unique(pair)
    f1s = []
    for _ in range(n_boot):
        pick = rng.choice(symbols, size=len(symbols), replace=True)
        idx = np.concatenate([np.where(pair == s)[0] for s in pick])
        yb, sb = y_true[idx], scores[idx]
        if yb.sum() == 0:
            continue
        f1s.append(f1_score(yb, sb >= thr, zero_division=0))
    if not f1s:
        return {"f1_lo": float("nan"), "f1_hi": float("nan")}
    return {
        "f1_lo": round(float(np.percentile(f1s, 2.5)), 4),
        "f1_hi": round(float(np.percentile(f1s, 97.5)), 4),
    }


def full_eval(y_true, scores, thr, end_ns, pair, labels_csv, cfg) -> dict:
    """Everything a result row needs: window + event + CI."""
    W = cfg["windows"]
    res = metric_table(y_true, scores, thr)
    res.update(event_metrics(y_true, scores, thr, end_ns, pair, labels_csv,
                             W["pre_minutes"], W["during_minutes"]))
    res.update(bootstrap_ci(y_true, scores, thr, pair))
    res["thr"] = round(float(thr), 6)
    return res


def save_result(artifacts_dir: str, mode: str, tag: str, res: dict) -> None:
    out = os.path.join(artifacts_dir, mode)
    os.makedirs(out, exist_ok=True)
    res = {"tag": tag, **res}
    with open(os.path.join(out, f"{tag}_test.json"), "w") as fh:
        json.dump(res, fh, indent=2)


# -------------------------------------------------------------- early stop
class EarlyStop:
    def __init__(self, patience: int = 6):
        self.patience = patience
        self.best = -np.inf
        self.count = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Returns True when `value` is a new best."""
        if value > self.best:
            self.best, self.count = value, 0
            return True
        self.count += 1
        self.should_stop = self.count >= self.patience
        return False
