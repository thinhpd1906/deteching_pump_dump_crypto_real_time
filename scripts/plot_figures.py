"""Generate all paper figures from saved results.

Figures produced:
  fig1_label_efficiency.png  -- F1 vs label fraction (money-shot for C2)
  fig2_score_timeline.png    -- anomaly score ±2h around a test pump
  fig3_cleaning_ablation.png -- bar chart Table 2 (F1 per cleaning mode)
  fig4_attention_heatmap.png -- AssDis / attention weights on pump window

Usage:
    python scripts/plot_figures.py --clean adaptive
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import yaml


STYLE = {
    "at_pre":     ("OURS: AT pre+ftune", "#1D4ED8", "-",  "o"),
    "at_scratch": ("AT from scratch",    "#EF4444", "--", "s"),
    "cnnbilstm":  ("CNN-BiLSTM",         "#10B981", "-.", "^"),
}


# ----------------------------------------------------------------- Fig 1
def fig_label_efficiency(le_path: str, out: str) -> None:
    if not os.path.exists(le_path):
        print(f"  [fig1] missing {le_path}  -- skipped")
        return

    data = json.load(open(le_path))
    fracs = sorted({float(k.split("_")[-1]) for k in data})
    fig, ax = plt.subplots(figsize=(7, 4))

    for name, (label, color, ls, marker) in STYLE.items():
        means, stds = [], []
        for frac in fracs:
            key = f"{name}_{frac}"
            if key in data:
                means.append(data[key]["mean"])
                stds.append(data[key]["std"])
            else:
                means.append(np.nan); stds.append(0)
        pct = [f * 100 for f in fracs]
        ax.plot(pct, means, label=label, color=color,
                linestyle=ls, marker=marker, linewidth=1.8)
        ax.fill_between(pct,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.15, color=color)

    ax.set_xlabel("Label fraction (%)", fontsize=11)
    ax.set_ylabel("Test F1", fontsize=11)
    ax.set_title("Label-Efficiency: Self-supervised Pre-training vs Baselines",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [fig1] -> {out}")


# ----------------------------------------------------------------- Fig 2
def fig_score_timeline(artifacts_dir: str, clean: str, out: str) -> None:
    """Load a test pump event and plot score over time."""
    art = os.path.join(artifacts_dir, clean)
    # Find any saved test predictions (needs to be added to finetune output)
    pred_path = os.path.join(art, "at_pre_test_preds.npz")
    if not os.path.exists(pred_path):
        print(f"  [fig2] missing {pred_path}  -- run finetune with --save-preds")
        return

    z = np.load(pred_path)
    scores, end_ns, y = z["scores"], z["end_ns"], z["y"]
    ends = np.array([np.datetime64(int(t), "ns") for t in end_ns])

    # Pick the first positive pump event
    pos_idx = np.where(y == 1)[0]
    if len(pos_idx) == 0:
        print("  [fig2] no positives in test set -- skipped")
        return

    center_ns = end_ns[pos_idx[0]]
    center = np.datetime64(int(center_ns), "ns")
    window = np.timedelta64(2 * 3600, "s")
    m = (ends >= center - window) & (ends <= center + window)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ends[m].astype("datetime64[s]"), scores[m],
            linewidth=1.0, color="#2563EB", label="anomaly score")
    ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1.2,
               label="alert threshold")
    ax.axvline(x=center.astype("datetime64[s]"), color="orange",
               linestyle="--", linewidth=1.5, label="pump label")
    ax.set_xlabel("Time (UTC)"); ax.set_ylabel("P(pump)")
    ax.set_title("Anomaly Score Timeline Around a Test Pump Event",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  [fig2] -> {out}")


# ----------------------------------------------------------------- Fig 3
def fig_cleaning_ablation(artifacts_dir: str, out: str) -> None:
    modes = ["none", "static", "adaptive"]
    f1s, cis_lo, cis_hi = [], [], []

    for mode in modes:
        p = os.path.join(artifacts_dir, mode, "at_pre_test.json")
        if os.path.exists(p):
            d = json.load(open(p))
            f1s.append(d.get("f1", 0))
            cis_lo.append(d.get("f1_lo", d.get("f1", 0)))
            cis_hi.append(d.get("f1_hi", d.get("f1", 0)))
        else:
            f1s.append(0); cis_lo.append(0); cis_hi.append(0)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(modes))
    bars = ax.bar(x, f1s, color=["#94A3B8", "#60A5FA", "#1D4ED8"],
                  width=0.5, zorder=3)
    ax.errorbar(x, f1s,
                yerr=[np.array(f1s) - np.array(cis_lo),
                      np.array(cis_hi) - np.array(f1s)],
                fmt="none", color="black", capsize=5, linewidth=1.5, zorder=4)
    for bar, v in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(modes)
    ax.set_ylabel("Test F1"); ax.set_ylim(0, 1.05)
    ax.set_title("Cleaning Ablation (OURS, AT pre+ftune)", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  [fig3] -> {out}")


# ----------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--clean", default="adaptive")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    art = cfg["paths"]["artifacts_dir"]
    fig_dir = os.path.join(art, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("Generating paper figures...")
    fig_label_efficiency(
        os.path.join(art, args.clean, "label_efficiency.json"),
        os.path.join(fig_dir, "fig1_label_efficiency.png"))
    fig_score_timeline(art, args.clean,
                       os.path.join(fig_dir, "fig2_score_timeline.png"))
    fig_cleaning_ablation(art, os.path.join(fig_dir, "fig3_cleaning_ablation.png"))
    print(f"\nFigures saved -> {fig_dir}/")


if __name__ == "__main__":
    main()
