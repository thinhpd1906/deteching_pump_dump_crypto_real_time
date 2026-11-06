"""Label-efficiency study (M16) -- the key evidence for Contribution 2.

Aggregates finetune results for every label fraction in [0.10, 0.25, 0.50,
1.00] and every seed of {at_pre, at_scratch, cnnbilstm}. The runs themselves
are ordinary finetune jobs (one per model x fraction x seed, resumable), so
the 100% point is the same run as the main Table 1 row, not a retrain:

    python -m src.training.finetune --arch at --pretrained --label-frac 0.1 --seed 0
    ...
    (scripts/kaggle_queue.py launches all of them)

The money-shot figure: F1 vs label fraction. If pretraining pays off under
label scarcity, at_pre curve must dominate at_scratch at low fractions.
Fractions are of labelled EVENTS (see common.subsample_labels).

Usage:
    python -m src.training.label_efficiency --clean adaptive --seeds 3
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import yaml

from src.training.finetune import run_tag
from src.training.protocol import atomic_write_json

FRACTIONS = [0.10, 0.25, 0.50, 1.00]
MODELS = {"at_pre": ("at", True), "at_scratch": ("at", False),
          "cnnbilstm": ("cnnbilstm", False)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--clean", default="adaptive")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    art = os.path.join(cfg["paths"]["artifacts_dir"], args.clean)

    print(f"=== LABEL-EFFICIENCY RESULTS [{args.clean}] "
          f"(test F1, mean ± std over seeds; fraction of EVENTS) ===")
    summary, missing = {}, []
    print(f"{'frac':>6}" + "".join(f"  {n:>18}" for n in MODELS))
    for frac in FRACTIONS:
        row = f"{frac:>6.2f}"
        for name, (arch, pre) in MODELS.items():
            vals, evr = [], []
            for seed in range(args.seeds):
                p = os.path.join(art, f"{run_tag(arch, pre, frac, seed)}_test.json")
                if not os.path.exists(p):
                    missing.append(os.path.basename(p))
                    continue
                d = json.load(open(p))
                vals.append(d["f1"])
                evr.append(d["event_recall"])
            if vals:
                mu, std = float(np.mean(vals)), float(np.std(vals))
                row += f"  {mu:.4f}±{std:.4f} (n={len(vals)})"
                summary[f"{name}_{frac}"] = {
                    "mean": round(mu, 4), "std": round(std, 4),
                    "vals": [round(v, 4) for v in vals],
                    "event_recall_mean": round(float(np.mean(evr)), 4)}
            else:
                row += f"  {'n/a':>18}"
        print(row)

    out = os.path.join(art, "label_efficiency.json")
    atomic_write_json(summary, out)
    print(f"\nSaved -> {out}")
    if missing:
        print(f"Missing {len(missing)} runs (run scripts/kaggle_queue.py): "
              + ", ".join(missing[:6]) + (" ..." if len(missing) > 6 else ""))

    # Quick Gate 3 check at 10% labels
    pre, scr = summary.get("at_pre_0.1"), summary.get("at_scratch_0.1")
    print("\n" + "=" * 62)
    print("GATE 3 (label-efficiency probe at 10% of labelled events):")
    if not (pre and scr):
        print("  not enough runs yet")
    else:
        print(f"  at_pre     mean F1 = {pre['mean']:.4f} ± {pre['std']:.4f}")
        print(f"  at_scratch mean F1 = {scr['mean']:.4f} ± {scr['std']:.4f}")
        if pre["mean"] > scr["mean"]:
            print("  at_pre ahead -- confirm the gap against symbol-bootstrap "
                  "CIs before calling Gate 3 PASS.")
        else:
            print("  FAIL -- pretraining does not help at 10% labels.")
            print("  Check full-label result; if also tied, see CLAUDE.md §12.")
    print("=" * 62)


if __name__ == "__main__":
    main()
