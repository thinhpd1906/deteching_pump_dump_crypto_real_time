"""M4b — Build labeled window datasets for every cleaning mode.

Pipeline per symbol:
    raw parquet -> clean_ticks(mode) -> 5s bars -> 10 features
    -> sliding windows -> labels -> chronological split -> negative subsampling

Output per mode (data/processed/{mode}/):
    train.npz / val.npz / test.npz   X f32[N,T,F], y i64[N], end_ns i64[N], pair
    scaler.json                      mean/std from TRAIN ONLY (shared by serving)
    unlabeled_pretrain.npz           train negatives -> self-supervised pool
    cleaning_stats.json              GATE C evidence (how much was removed)

Chronological split by GLOBAL pump-time quantiles -> zero look-ahead leakage.

Usage:
    python -m src.data.build_dataset --clean adaptive
    python -m src.data.build_dataset --clean all
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.cleaning.filters import clean_ticks, cleaning_stats
from src.data.features import (FEATURES, bars_to_features, label_windows,
                               make_windows, trades_to_bars)

RNG = np.random.default_rng(42)


def load_symbol_ticks(raw_dir: str, pair: str) -> pd.DataFrame | None:
    files = sorted(glob.glob(os.path.join(raw_dir, pair, "*.parquet")))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df


def _subsample_negatives(y: np.ndarray, keep_ratio: float,
                         split_tag: np.ndarray, split: str) -> np.ndarray:
    """Keep-mask limiting negatives to keep_ratio x positives within one split."""
    keep = np.ones(len(y), dtype=bool)
    m = split_tag == split
    pos = np.where(m & (y == 1))[0]
    neg = np.where(m & (y == 0))[0]
    n_keep = int(keep_ratio * max(len(pos), 1))
    if len(neg) > n_keep:
        drop = RNG.choice(neg, size=len(neg) - n_keep, replace=False)
        keep[drop] = False
    return keep


def build_mode(cfg: dict, mode: str) -> None:
    P, B, W, S, C = (cfg["paths"], cfg["bars"], cfg["windows"],
                     cfg["split"], cfg["cleaning"])

    labels = pd.read_csv(P["labels_csv"], parse_dates=["pump_time"])
    pairs = [p for p in sorted(labels["pair"].unique())
             if os.path.isdir(os.path.join(P["raw_dir"], p))]
    if not pairs:
        raise SystemExit(
            "No downloaded data found. Run:\n"
            "  python -m src.data.download_binance --limit-events 10")

    print(f"\n[{mode}] building from {len(pairs)} pairs with local data")

    # chronological split boundaries from GLOBAL pump-time quantiles
    t1 = labels["pump_time"].quantile(S["train_frac"])
    t2 = labels["pump_time"].quantile(S["train_frac"] + S["val_frac"])
    print(f"[{mode}] split boundaries: train<= {t1}  val<= {t2}  test> {t2}")

    Xs, ys, ends, prs, tags = [], [], [], [], []
    stats = {"ticks_in": 0, "removed": 0, "removed_in_pump": 0,
             "removed_in_normal": 0}

    for pair in tqdm(pairs, desc=f"[{mode}] symbols"):
        raw = load_symbol_ticks(P["raw_dir"], pair)
        if raw is None or len(raw) < 1000:
            continue

        cleaned = clean_ticks(
            raw, mode,
            static_pct=C["static_pct"], k0=C["adaptive_k0"],
            tick_sigma_window=C["tick_sigma_window"],
            regime_clip=tuple(C["regime_clip"]),
        )
        st = cleaning_stats(raw, cleaned)
        stats["ticks_in"] += st["ticks_in"]
        stats["removed"] += st["removed"]

        # GATE C evidence: were removed trades near pumps or in normal periods?
        if st["removed"] > 0:
            kept = set(cleaned["agg_id"].to_numpy())
            removed = raw.loc[~raw["agg_id"].isin(kept)]
            rem_ts = pd.to_datetime(removed["time"], unit="ms", utc=True)
            ptimes = labels.loc[labels["pair"] == pair, "pump_time"]
            near = np.zeros(len(rem_ts), dtype=bool)
            for t in ptimes:
                near |= np.asarray(
                    (rem_ts >= t - pd.Timedelta(minutes=W["pre_minutes"])) &
                    (rem_ts <= t + pd.Timedelta(minutes=W["during_minutes"])))
            stats["removed_in_pump"] += int(near.sum())
            stats["removed_in_normal"] += int((~near).sum())

        bars = trades_to_bars(cleaned, B["bar_seconds"])
        feats = bars_to_features(bars, B["z_window"], B["vol_window"])
        X, end_ns = make_windows(feats, W["window_bars"], W["stride_bars"])
        if len(X) == 0:
            continue

        ptimes = labels.loc[labels["pair"] == pair, "pump_time"]
        y, mask_out = label_windows(
            end_ns, ptimes,
            pre_minutes=W["pre_minutes"],
            during_minutes=W["during_minutes"],
            buffer_hours=W["buffer_hours"],
        )
        X, y, end_ns = X[~mask_out], y[~mask_out], end_ns[~mask_out]
        if len(X) == 0:
            continue

        ends_dt = pd.to_datetime(end_ns, utc=True)
        tag = np.where(ends_dt <= t1, "train",
                       np.where(ends_dt <= t2, "val", "test"))

        Xs.append(X); ys.append(y); ends.append(end_ns)
        prs.append(np.full(len(y), pair)); tags.append(tag)

    if not Xs:
        raise SystemExit(f"[{mode}] no windows produced — check downloads.")

    X = np.concatenate(Xs); y = np.concatenate(ys)
    end_ns = np.concatenate(ends); pair_arr = np.concatenate(prs)
    tag = np.concatenate(tags)

    rm_pct = 100 * stats["removed"] / max(stats["ticks_in"], 1)
    print(f"[{mode}] cleaning removed {stats['removed']:,} / "
          f"{stats['ticks_in']:,} trades ({rm_pct:.4f}%)")

    keep = (_subsample_negatives(y, W["neg_ratio_train"], tag, "train")
            & _subsample_negatives(y, W["neg_cap_eval"], tag, "val")
            & _subsample_negatives(y, W["neg_cap_eval"], tag, "test"))
    X, y, end_ns, pair_arr, tag = (a[keep] for a in
                                   (X, y, end_ns, pair_arr, tag))

    out_dir = os.path.join(P["processed_dir"], mode)
    os.makedirs(out_dir, exist_ok=True)

    for split in ("train", "val", "test"):
        m = tag == split
        np.savez_compressed(
            os.path.join(out_dir, f"{split}.npz"),
            X=X[m], y=y[m], end_ns=end_ns[m], pair=pair_arr[m])
        pos = int(y[m].sum())
        pct = 100 * y[m].mean() if m.sum() else 0.0
        print(f"[{mode}] {split:5s}: N={m.sum():7,}  pos={pos:5,} ({pct:.2f}%)")
        if m.sum() and pos == 0:
            print(f"    WARNING: {split} has zero positives!")

    # scaler from TRAIN only  (serving + all models share this)
    mtr = tag == "train"
    mean = X[mtr].reshape(-1, X.shape[-1]).mean(0)
    std = X[mtr].reshape(-1, X.shape[-1]).std(0) + 1e-8
    json.dump({"features": FEATURES, "mean": mean.tolist(),
               "std": std.tolist()},
              open(os.path.join(out_dir, "scaler.json"), "w"), indent=2)

    # unlabeled pool for self-supervised pretraining = train negatives
    mpool = mtr & (y == 0)
    np.savez_compressed(os.path.join(out_dir, "unlabeled_pretrain.npz"),
                        X=X[mpool])
    print(f"[{mode}] pretrain pool: {int(mpool.sum()):,} unlabeled windows")

    stats["removed_pct"] = round(rm_pct, 4)
    json.dump(stats, open(os.path.join(out_dir, "cleaning_stats.json"), "w"),
              indent=2)
    print(f"[{mode}] -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--clean", default="adaptive",
                    choices=["none", "static", "adaptive", "all"])
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    modes = (["none", "static", "adaptive"] if args.clean == "all"
             else [args.clean])
    for m in modes:
        build_mode(cfg, m)


if __name__ == "__main__":
    main()
