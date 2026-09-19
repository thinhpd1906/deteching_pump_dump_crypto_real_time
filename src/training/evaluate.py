"""Final aggregator: builds Tables 1 and 2 from all saved JSON results.

Reads artifacts/{mode}/{tag}_test.json -> prints both paper tables +
runs Gate 3 check automatically.

Table 1 -- method comparison (on `--clean` mode, default: adaptive)
Table 2 -- cleaning ablation (OURS at_pre across none/static/adaptive)

Usage:
    python -m src.training.evaluate
    python -m src.training.evaluate --clean adaptive
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import yaml

# ---- ordered rows for Table 1 ----
TABLE1_ROWS = [
    ("B1  Rule-based threshold",       "b1_rule"),
    ("B2  Rush-order rule",            "b2_rushorder"),
    ("B3  Microstructure (OFI/VPIN)",  "b3_microstructure"),
    ("B4  Random Forest",              "b4_randomforest"),
    ("B5  Logistic Regression",        "b5_logreg"),
    ("B6  XGBoost",                    "b6_xgboost"),
    ("B7  LSTM-AE (unsupervised)",     "b7_lstmae"),
    ("B8  CNN-BiLSTM (supervised)",    "b8_cnnbilstm"),        # also cnnbilstm_scratch
    ("B9  AT from scratch",            "at_scratch"),
    ("B10 AT pretrained unsupervised", "b10_at_unsup"),
    ("D   OURS: AT pretrained+ftune",  "at_pre"),
]
WINDOW_COLS = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
EVENT_COLS  = ["event_recall", "fa_per_symbol_day"]
CI_COLS     = ["f1_lo", "f1_hi"]


def _read(art: str, tag: str) -> dict | None:
    # try exact tag first, then startswith (covers seeds, fracs)
    for pattern in [
        os.path.join(art, f"{tag}_test.json"),
        os.path.join(art, f"{tag}*_test.json"),
    ]:
        hits = sorted(glob.glob(pattern))
        if hits:
            return json.load(open(hits[-1]))   # latest
    return None


def _fmt(d: dict | None, cols: list[str]) -> str:
    if d is None:
        return " | ".join(["  n/a  "] * len(cols))
    return " | ".join(f"{d.get(c, float('nan')):>7.4f}" for c in cols)


def _multi_seed_mean(art: str, tag_prefix: str, col: str) -> str:
    """Find all seeds (tag_prefix_s0, _s1, ...) and report mean ± std."""
    hits = sorted(glob.glob(os.path.join(art, f"{tag_prefix}_s*_test.json")))
    if not hits:
        single = _read(art, tag_prefix)
        return f"{single.get(col, float('nan')):.4f}      " if single else "  n/a  "
    vals = [json.load(open(h)).get(col, float("nan")) for h in hits]
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return "  n/a  "
    return f"{np.mean(vals):.4f}±{np.std(vals):.4f}"


def print_table1(art: str) -> list[dict]:
    w = max(len(n) for n, _ in TABLE1_ROWS) + 2
    header = (f"{'Method':<{w}} | "
              + " | ".join(f"{c:>7}" for c in WINDOW_COLS)
              + " | " + " | ".join(f"{c:>9}" for c in EVENT_COLS)
              + " | F1 95% CI")
    sep = "-" * len(header)
    print("\n=== TABLE 1 -- METHOD COMPARISON ===")
    print(header); print(sep)

    rows = []
    for name, tag in TABLE1_ROWS:
        d = _read(art, tag)
        wrow = _fmt(d, WINDOW_COLS)
        erow = _fmt(d, EVENT_COLS)
        ci = f"[{d.get('f1_lo', float('nan')):.4f},{d.get('f1_hi', float('nan')):.4f}]" if d else " n/a "
        print(f"{name:<{w}} | {wrow} | {erow} | {ci}")
        rows.append(d)
    return rows


def print_table2(art_root: str, tag: str = "at_pre") -> None:
    print("\n=== TABLE 2 -- CLEANING ABLATION (OURS) ===")
    cols = WINDOW_COLS + EVENT_COLS
    w = 10
    header = (f"{'Mode':<{w}} | "
              + " | ".join(f"{c:>7}" for c in WINDOW_COLS)
              + " | " + " | ".join(f"{c:>9}" for c in EVENT_COLS)
              + " | F1 95% CI")
    print(header); print("-" * len(header))
    for mode in ("none", "static", "adaptive"):
        art = os.path.join(art_root, mode)
        d = _read(art, tag)
        wrow = _fmt(d, WINDOW_COLS)
        erow = _fmt(d, EVENT_COLS)
        ci = f"[{d.get('f1_lo', float('nan')):.4f},{d.get('f1_hi', float('nan')):.4f}]" if d else " n/a "
        print(f"{mode:<{w}} | {wrow} | {erow} | {ci}")


def gate3_check(art: str) -> None:
    pre  = _read(art, "at_pre")
    scr  = _read(art, "at_scratch")
    cnn  = _read(art, "cnnbilstm_scratch") or _read(art, "cnnbilstm")

    print("\n" + "=" * 62)
    print("GATE 3 -- SELF-SUPERVISION CHECK")
    if pre and scr:
        f1p, f1s = pre.get("f1", 0), scr.get("f1", 0)
        print(f"  at_pre    test F1 = {f1p:.4f}")
        print(f"  at_scratch test F1 = {f1s:.4f}")
        # Check if gap is outside bootstrap CIs
        p_lo = pre.get("f1_lo", f1p)
        s_hi = scr.get("f1_hi", f1s)
        if f1p > f1s:
            print(f"  at_pre > at_scratch by {f1p - f1s:.4f}")
            if f1p > s_hi:
                print("  GATE 3: PASS -- gap is outside CI. Contribution 2 is real.")
            else:
                print("  GATE 3: MARGINAL -- check label-efficiency at 10% labels.")
        else:
            print(f"  GATE 3: FAIL -- pretraining did not improve over random init.")
            print("  Action: check label_efficiency.py at 10% labels before deciding.")
    else:
        print("  Missing results. Run finetune for at_pre and at_scratch first.")

    if pre and cnn:
        f1p, f1c = pre.get("f1", 0), cnn.get("f1", 0)
        print(f"  OURS vs CNN-BiLSTM: {f1p:.4f} vs {f1c:.4f} "
              f"({'OURS wins' if f1p > f1c else 'CNN wins/ties'})")
    print("=" * 62)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--clean", default="adaptive")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    art_root = cfg["paths"]["artifacts_dir"]
    art = os.path.join(art_root, args.clean)

    found = len(glob.glob(os.path.join(art_root, "*", "*_test.json")))
    print(f"Found {found} result files under {art_root}/")

    print_table1(art)
    print_table2(art_root)
    gate3_check(art)


if __name__ == "__main__":
    main()
