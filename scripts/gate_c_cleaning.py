"""GATE C -- Cleaning effect diagnostics.

Answers the critical question: does adaptive cleaning actually change
anything meaningful, or is it noise?

Three measurements:
  1. Removal rate per mode (% of trades removed)
  2. Differential removal: are removed trades concentrated near pump events?
  3. Feature distribution shift: KS statistic on z_ret/z_vol between modes

PASS: removal >= 0.3% OR differential removal (pump < normal, ratio<0.5)
      OR KS > 0.05
FAIL: cleaning is cosmetic -> strengthen the intervention OR demote C1.

Data sources (both are read-only; nothing under data/processed/ or the
src/cleaning, src/data pipeline modules is modified by this script):

  Measurements 1 & 2 read data/processed/{mode}/cleaning_stats.json, which
  is written by `src.data.build_dataset` from a full run over ALL downloaded
  pairs. That pipeline applies `clean_ticks` per CONTIGUOUS-CALENDAR-DAY
  CHUNK (see `load_symbol_chunks` in src/data/build_dataset.py) -- this is
  the authoritative, already-computed number and is preferred over
  recomputing from raw here.

  Earlier versions of this script re-concatenated a pair's ENTIRE raw
  history (all downloaded files, which can span years across disjoint
  pump/normal-day download windows) and called `trades_to_bars` on it
  directly. Because `trades_to_bars` resamples onto a CONTINUOUS 5s grid,
  that bridges multi-year gaps between a pair's disjoint download clusters
  with tens of millions of phantom empty bars -> OOM (observed: a single
  pair, DLTBTC, spans 2018-01-28 to 2020-12-27 = ~1065 days -> ~18M phantom
  bars). `build_dataset.py` avoids this by chunking per contiguous
  calendar-day run before ever calling `trades_to_bars`. This script now
  avoids the problem entirely for measurement 3 by reading the FEATURES
  that `build_dataset.py` already computed correctly (per-chunk) and saved
  into data/processed/{mode}/{train,val,test}.npz, rather than recomputing
  bars/features from raw ticks.

Usage:
    python -m scripts.gate_c_cleaning
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats

from src.data.features import FEATURES

RNG = np.random.default_rng(42)
KS_SAMPLE_CAP = 50000


def _load_cleaning_stats(mode: str, processed_dir: str) -> dict:
    path = os.path.join(processed_dir, mode, "cleaning_stats.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"Missing {path}. Run `python -m src.data.build_dataset "
            f"--clean {mode}` first (this gate reads its output, it does "
            f"not rebuild datasets)."
        )
    return json.load(open(path))


def _load_feature_samples(mode: str, processed_dir: str, feat_name: str,
                          rng: np.random.Generator) -> np.ndarray:
    """Pull one sample per window (the window's END bar) for `feat_name`,
    concatenated across train/val/test, then randomly subsample to
    KS_SAMPLE_CAP. Using the end bar (not every bar in every overlapping
    window) avoids re-counting the same underlying 5s bar dozens of times
    due to the stride-12-of-120 window overlap.
    """
    feat_idx = FEATURES.index(feat_name)
    parts = []
    for split in ("train", "val", "test"):
        path = os.path.join(processed_dir, mode, f"{split}.npz")
        if not os.path.exists(path):
            continue
        d = np.load(path)
        X = d["X"]  # [N, T, F]
        if len(X) == 0:
            continue
        parts.append(X[:, -1, feat_idx])
    if not parts:
        return np.array([])
    vals = np.concatenate(parts)
    if len(vals) > KS_SAMPLE_CAP:
        vals = rng.choice(vals, size=KS_SAMPLE_CAP, replace=False)
    return vals


def main():
    processed_dir = "data/processed"

    print("GATE C: Cleaning diagnostics (reading production build outputs)\n")

    results = {}
    for mode in ("none", "static", "adaptive"):
        results[mode] = _load_cleaning_stats(mode, processed_dir)

    print("=== MEASUREMENT 1: Removal rates (source: cleaning_stats.json, "
          "all downloaded pairs) ===")
    any_pass = False
    triggers = []
    for mode in ("static", "adaptive"):
        r = results[mode]
        pct = r["removed_pct"]
        print(f"  {mode:10s}: {r['removed']:,} removed / {r['ticks_in']:,} "
              f"({pct:.4f}%)")
        if pct >= 0.3:
            print(f"    -> removal >= 0.3%: evidence of meaningful cleaning")
            any_pass = True
            triggers.append(f"removal_rate_{mode}>=0.3%")

    print("\n=== MEASUREMENT 2: Differential removal (pump-window vs normal) ===")
    for mode in ("static", "adaptive"):
        r = results[mode]
        tot = max(r["removed"], 1)
        pump_frac = r["removed_in_pump"] / tot
        norm_frac = r["removed_in_normal"] / tot
        ratio = pump_frac / (norm_frac + 1e-9)
        print(f"  {mode:10s}: pump-window={pump_frac:.4f}  "
              f"normal={norm_frac:.4f}  ratio(pump/normal)={ratio:.3f}x")
        results[mode]["pump_frac"] = pump_frac
        results[mode]["normal_frac"] = norm_frac
        results[mode]["pump_normal_ratio"] = ratio
        if pump_frac < norm_frac * 0.5:
            print(f"    -> cleaning removes MORE from normal periods "
                  f"(good: pump signal preserved)")
            any_pass = True
            triggers.append(f"differential_removal_{mode}")

    print("\n=== MEASUREMENT 3: Feature distribution shift (KS test, "
          "none vs adaptive) ===")
    print("  (source: end-bar feature values from data/processed/{mode}/"
          "{train,val,test}.npz -- one sample per window to avoid "
          "double-counting overlapping windows)")
    ks_results = {}
    for feat in ("z_ret", "z_vol"):
        a = _load_feature_samples("none", processed_dir, feat, RNG)
        b = _load_feature_samples("adaptive", processed_dir, feat, RNG)
        if len(a) > 100 and len(b) > 100:
            ks, pval = stats.ks_2samp(a, b)
            print(f"  KS({feat}): statistic={ks:.5f}  p={pval:.4g}  "
                  f"(n_none={len(a):,}, n_adaptive={len(b):,})", end="")
            ks_results[feat] = {"statistic": float(ks), "pvalue": float(pval),
                                "n_none": int(len(a)), "n_adaptive": int(len(b))}
            if ks > 0.05:
                print("  -> significant shift (cleaning changes feature distribution)")
                any_pass = True
                triggers.append(f"ks_{feat}>0.05")
            else:
                print("  -> no significant shift")
        else:
            print(f"  KS({feat}): insufficient data (n_none={len(a)}, "
                  f"n_adaptive={len(b)})")
            ks_results[feat] = None

    # Save results for the paper
    os.makedirs("artifacts", exist_ok=True)
    out = {
        "cleaning_stats": results,
        "ks_tests": ks_results,
        "gate_pass": any_pass,
        "triggers": triggers,
    }
    json.dump(out, open("artifacts/gate_c_results.json", "w"), indent=2)

    print("\n" + "=" * 62)
    if any_pass:
        print(f"GATE C: PASS -- cleaning has measurable effect on data.")
        print(f"  Triggered by: {', '.join(triggers)}")
        print("  Contribution 1 (cleaning ablation) is defensible.")
    else:
        print("GATE C: FAIL -- cleaning appears cosmetic.")
        print("  Action: strengthen the intervention OR demote C1.")
    print("=" * 62)


if __name__ == "__main__":
    main()
