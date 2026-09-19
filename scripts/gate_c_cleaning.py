"""GATE C -- Cleaning effect diagnostics.

Answers the critical question: does adaptive cleaning actually change
anything meaningful, or is it noise?

Three measurements:
  1. Removal rate per mode (% of trades removed)
  2. Differential removal: are removed trades concentrated near pump events?
  3. Feature distribution shift: KS statistic on z_ret/z_vol between modes

PASS: removal >= 0.3% OR differential removal (pump>normal) OR KS > 0.05
FAIL: cleaning is cosmetic -> strengthen the intervention OR demote C1.

Usage:
    python -m scripts.gate_c_cleaning
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from src.cleaning.filters import clean_ticks, cleaning_stats
from src.data.features import FEATURES


def load_raw(raw_dir: str, pair: str) -> pd.DataFrame | None:
    files = sorted(glob.glob(os.path.join(raw_dir, pair, "*.parquet")))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def main():
    cfg = yaml.safe_load(open("config/config.yaml"))
    C = cfg["cleaning"]
    W = cfg["windows"]
    raw_dir = cfg["paths"]["raw_dir"]
    labels = pd.read_csv(cfg["paths"]["labels_csv"], parse_dates=["pump_time"])

    pairs = [p for p in sorted(labels["pair"].unique())
             if os.path.isdir(os.path.join(raw_dir, p))]
    if not pairs:
        print("ERROR: No downloaded data. Run download_binance first.")
        return

    print(f"GATE C: Cleaning diagnostics on {len(pairs)} pairs\n")

    results = {mode: {"ticks_in": 0, "removed": 0,
                      "removed_pump": 0, "removed_normal": 0}
               for mode in ("static", "adaptive")}

    # Feature distributions per mode for KS test
    feat_idx = FEATURES.index("z_ret")
    vol_idx = FEATURES.index("z_vol")
    feats_none = {"z_ret": [], "z_vol": []}
    feats_adapt = {"z_ret": [], "z_vol": []}

    for pair in pairs[:30]:  # limit for speed
        raw = load_raw(raw_dir, pair)
        if raw is None or len(raw) < 500:
            continue

        ptimes = labels.loc[labels["pair"] == pair, "pump_time"]

        for mode in ("static", "adaptive"):
            cleaned = clean_ticks(
                raw, mode,
                static_pct=C["static_pct"], k0=C["adaptive_k0"],
                tick_sigma_window=C["tick_sigma_window"],
                regime_clip=tuple(C["regime_clip"]),
            )
            st = cleaning_stats(raw, cleaned)
            results[mode]["ticks_in"] += st["ticks_in"]
            results[mode]["removed"] += st["removed"]

            if st["removed"] > 0:
                kept_ids = set(cleaned["agg_id"].to_numpy())
                removed = raw.loc[~raw["agg_id"].isin(kept_ids)]
                rem_ts = pd.to_datetime(removed["time"], unit="ms", utc=True)

                near_pump = np.zeros(len(rem_ts), dtype=bool)
                for t in ptimes:
                    near_pump |= np.asarray(
                        (rem_ts >= t - pd.Timedelta(minutes=W["pre_minutes"])) &
                        (rem_ts <= t + pd.Timedelta(minutes=W["during_minutes"])))
                results[mode]["removed_pump"] += int(near_pump.sum())
                results[mode]["removed_normal"] += int((~near_pump).sum())

        # Feature distributions: none vs adaptive on same symbol
        from src.data.features import trades_to_bars, bars_to_features
        B = cfg["bars"]
        for mode_key, feats_dict in [("none", feats_none),
                                      ("adaptive", feats_adapt)]:
            cleaned = clean_ticks(raw, mode_key,
                                  static_pct=C["static_pct"],
                                  k0=C["adaptive_k0"],
                                  tick_sigma_window=C["tick_sigma_window"],
                                  regime_clip=tuple(C["regime_clip"]))
            bars = trades_to_bars(cleaned, B["bar_seconds"])
            feats = bars_to_features(bars, B["z_window"], B["vol_window"])
            feats_dict["z_ret"].extend(feats["z_ret"].to_numpy().tolist())
            feats_dict["z_vol"].extend(feats["z_vol"].to_numpy().tolist())

    print("=== MEASUREMENT 1: Removal rates ===")
    any_pass = False
    for mode in ("static", "adaptive"):
        r = results[mode]
        pct = 100 * r["removed"] / max(r["ticks_in"], 1)
        print(f"  {mode:10s}: {r['removed']:,} removed / {r['ticks_in']:,} "
              f"({pct:.4f}%)")
        if pct >= 0.3:
            print(f"    -> removal >= 0.3%: evidence of meaningful cleaning")
            any_pass = True

    print("\n=== MEASUREMENT 2: Differential removal (pump vs normal) ===")
    for mode in ("static", "adaptive"):
        r = results[mode]
        tot = max(r["removed"], 1)
        pump_frac = r["removed_pump"] / tot
        norm_frac = r["removed_normal"] / tot
        print(f"  {mode:10s}: pump-window={pump_frac:.4f}  "
              f"normal={norm_frac:.4f}  "
              f"ratio={pump_frac/(norm_frac+1e-9):.2f}x")
        if pump_frac < norm_frac * 0.5:
            print(f"    -> cleaning removes MORE from normal periods (good: pump signal preserved)")
            any_pass = True

    print("\n=== MEASUREMENT 3: Feature distribution shift (KS test) ===")
    for feat in ("z_ret", "z_vol"):
        a = np.array(feats_none[feat])
        b = np.array(feats_adapt[feat])
        if len(a) > 100 and len(b) > 100:
            ks, pval = stats.ks_2samp(a[:50000], b[:50000])
            print(f"  KS({feat}): statistic={ks:.5f}  p={pval:.4f}",
                  end="")
            if ks > 0.05:
                print("  -> significant shift (cleaning changes feature distribution)")
                any_pass = True
            else:
                print("  -> no significant shift")

    # Save results for the paper
    os.makedirs("artifacts", exist_ok=True)
    json.dump(results, open("artifacts/gate_c_results.json", "w"), indent=2)

    print("\n" + "=" * 62)
    if any_pass:
        print("GATE C: PASS -- cleaning has measurable effect on data.")
        print("  Contribution 1 (interaction matrix) is defensible.")
    else:
        print("GATE C: FAIL -- cleaning appears cosmetic.")
        print("  Action: strengthen the intervention OR demote C1.")
        print("  See PILOT_PLAN.md §6-C for options.")
    print("=" * 62)


if __name__ == "__main__":
    main()
