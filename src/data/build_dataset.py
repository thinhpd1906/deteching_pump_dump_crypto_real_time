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


def load_symbol_chunks(raw_dir: str, pair: str) -> list[pd.DataFrame]:
    """Load a pair's raw ticks as a list of contiguous-calendar-day chunks.

    Chunk boundaries come from the DOWNLOADED FILE LIST (daily parquet
    filenames, e.g. 2020-04-07.parquet), not from gaps in trade timestamps.
    A run of files stays in one chunk only while each date is exactly one
    calendar day after the previous file's date; any gap >=2 days (including
    a day that 404'd and was never downloaded) starts a new chunk.

    This matters because a pair's downloaded days are event-centered clusters
    (±days_before/after_event around each of the pair's pump events, plus
    sparse normal-day samples) — NOT a continuous history. Concatenating all
    of them and resampling in one shot (the old behaviour) bridges multi-day/
    multi-month/multi-year gaps between clusters with millions of phantom
    imputed empty bars. Building bars/features/windows independently per
    chunk keeps rolling-window warmup (z_window, vol_window) from ever
    crossing a real gap.
    """
    files = sorted(glob.glob(os.path.join(raw_dir, pair, "*.parquet")))
    if not files:
        return []

    dates = [pd.Timestamp(os.path.splitext(os.path.basename(f))[0])
             for f in files]

    chunks: list[list[str]] = [[files[0]]]
    for prev_d, d, f in zip(dates, dates[1:], files[1:]):
        if (d - prev_d).days == 1:
            chunks[-1].append(f)
        else:
            chunks.append([f])

    return [pd.concat([pd.read_parquet(f) for f in chunk], ignore_index=True)
            for chunk in chunks]


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

    stats = {"ticks_in": 0, "removed": 0, "removed_in_pump": 0,
             "removed_in_normal": 0}
    chunks_per_pair = []  # for reporting: n contiguous-day chunks per pair

    # ------------------------------------------------------------ PASS 1/2
    # Label every window but do NOT retain X (float32 [N,120,10] is the
    # memory-heavy part -- with ~2,000 downloaded pair-days at 5s bars and
    # a 1-min stride, the full unsampled dataset is ~2.7M windows / ~12GB,
    # which does not fit in RAM at once together with negative subsampling
    # still to come). Only y/end_ns/tag survive this pass (cheap, int64).
    # Cleaned per-chunk tick data IS cached for pass 2 -- the whole raw
    # corpus is only ~150MB, trivial to hold in memory.
    cleaned_cache: dict[str, list[pd.DataFrame]] = {}
    pair_offsets: dict[str, tuple[int, int]] = {}
    ys, ends, prs, tags = [], [], [], []
    cursor = 0

    for pair in tqdm(pairs, desc=f"[{mode}] pass 1/2 (label)"):
        pair_chunks = load_symbol_chunks(P["raw_dir"], pair)
        pair_chunks = [c for c in pair_chunks if len(c) >= 1000]
        if not pair_chunks:
            continue
        chunks_per_pair.append(len(pair_chunks))

        ptimes = labels.loc[labels["pair"] == pair, "pump_time"]

        cleaned_chunks, pair_ys, pair_ends = [], [], []
        for raw in pair_chunks:
            # each contiguous-calendar-day chunk gets its own bars/features/
            # windows pipeline so rolling warmup (z_window, vol_window) and
            # the 5s-bar resample grid never bridge a real (multi-day+) gap
            # between a pair's downloaded date clusters.
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
                near = np.zeros(len(rem_ts), dtype=bool)
                for t in ptimes:
                    near |= np.asarray(
                        (rem_ts >= t - pd.Timedelta(minutes=W["pre_minutes"])) &
                        (rem_ts <= t + pd.Timedelta(minutes=W["during_minutes"])))
                stats["removed_in_pump"] += int(near.sum())
                stats["removed_in_normal"] += int((~near).sum())

            cleaned_chunks.append(cleaned)

            bars = trades_to_bars(cleaned, B["bar_seconds"])
            feats = bars_to_features(bars, B["z_window"], B["vol_window"])
            X, end_ns = make_windows(feats, W["window_bars"], W["stride_bars"])
            if len(X) == 0:
                # expected for short chunks (e.g. a single low-activity
                # normal-day) that don't reach window_bars=120 bars
                continue

            y, mask_out = label_windows(
                end_ns, ptimes,
                pre_minutes=W["pre_minutes"],
                during_minutes=W["during_minutes"],
                buffer_hours=W["buffer_hours"],
            )
            y, end_ns = y[~mask_out], end_ns[~mask_out]
            if len(y) == 0:
                continue

            pair_ys.append(y); pair_ends.append(end_ns)
            # X is deliberately dropped here -- pass 2 rebuilds it cheaply
            # (from the cached cleaned ticks) only for rows that survive
            # negative subsampling.

        if not pair_ys:
            continue

        y = np.concatenate(pair_ys); end_ns = np.concatenate(pair_ends)

        cleaned_cache[pair] = cleaned_chunks
        pair_offsets[pair] = (cursor, cursor + len(y))
        cursor += len(y)

        ends_dt = pd.to_datetime(end_ns, utc=True)
        tag = np.where(ends_dt <= t1, "train",
                       np.where(ends_dt <= t2, "val", "test"))

        ys.append(y); ends.append(end_ns)
        prs.append(np.full(len(y), pair)); tags.append(tag)

    if chunks_per_pair:
        print(f"[{mode}] chunks/pair: mean={np.mean(chunks_per_pair):.1f}  "
              f"max={max(chunks_per_pair)}  "
              f"(from {len(chunks_per_pair)} pairs with usable data)")

    if not ys:
        raise SystemExit(f"[{mode}] no windows produced — check downloads.")

    y_all = np.concatenate(ys); end_ns_all = np.concatenate(ends)
    pair_all = np.concatenate(prs); tag_all = np.concatenate(tags)

    rm_pct = 100 * stats["removed"] / max(stats["ticks_in"], 1)
    print(f"[{mode}] cleaning removed {stats['removed']:,} / "
          f"{stats['ticks_in']:,} trades ({rm_pct:.4f}%)")
    print(f"[{mode}] total windows before neg subsampling: {len(y_all):,}")

    keep = (_subsample_negatives(y_all, W["neg_ratio_train"], tag_all, "train")
            & _subsample_negatives(y_all, W["neg_cap_eval"], tag_all, "val")
            & _subsample_negatives(y_all, W["neg_cap_eval"], tag_all, "test"))

    # ------------------------------------------------------------ PASS 2/2
    # Rebuild X per pair from the cached, already-cleaned ticks (cheap:
    # bars/features/windows recompute only, no re-read/re-clean), then
    # immediately keep only the rows selected by `keep` -- the full
    # unsampled X for a pair (let alone the whole dataset) never has to
    # coexist in memory with every other pair's X.
    Xs, ys_k, ends_k, prs_k, tags_k = [], [], [], [], []
    for pair in tqdm(pairs, desc=f"[{mode}] pass 2/2 (features)"):
        if pair not in pair_offsets:
            continue
        start, stop = pair_offsets[pair]
        pair_keep = keep[start:stop]
        if not pair_keep.any():
            continue

        pair_Xs = []
        for cleaned in cleaned_cache[pair]:
            bars = trades_to_bars(cleaned, B["bar_seconds"])
            feats = bars_to_features(bars, B["z_window"], B["vol_window"])
            X, end_ns = make_windows(feats, W["window_bars"], W["stride_bars"])
            if len(X) == 0:
                continue
            ptimes = labels.loc[labels["pair"] == pair, "pump_time"]
            _, mask_out = label_windows(
                end_ns, ptimes,
                pre_minutes=W["pre_minutes"],
                during_minutes=W["during_minutes"],
                buffer_hours=W["buffer_hours"],
            )
            X = X[~mask_out]
            if len(X) == 0:
                continue
            pair_Xs.append(X)

        X = np.concatenate(pair_Xs)
        assert len(X) == stop - start, (
            f"[{mode}] pass1/pass2 row-count mismatch for {pair}: "
            f"{len(X)} vs {stop - start} -- non-determinism in the pipeline?")
        X = X[pair_keep]

        Xs.append(X)
        ys_k.append(y_all[start:stop][pair_keep])
        ends_k.append(end_ns_all[start:stop][pair_keep])
        prs_k.append(pair_all[start:stop][pair_keep])
        tags_k.append(tag_all[start:stop][pair_keep])

    del cleaned_cache

    X = np.concatenate(Xs); y = np.concatenate(ys_k)
    end_ns = np.concatenate(ends_k); pair_arr = np.concatenate(prs_k)
    tag = np.concatenate(tags_k)

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
