"""Paired symbol-level bootstrap between two model arms (bootstrap-ci skill).

Uses the per-window test scores saved at the single test evaluation
(artifacts/<mode>/scores/<tag>_test.npz) and each run's own val-tuned
threshold. No model is re-evaluated; test.npz is read only for the labels
and symbols the saved scores were computed on (fingerprint-checked).

Seeds are paired by index (a_s0 with b_s0, ...). Each bootstrap replicate
resamples SYMBOLS with replacement ONCE and applies the same resample to both
arms and all seeds; the statistic is the mean over seed pairs of
F1(a) - F1(b). Also reports a Wilcoxon signed-rank test on per-symbol F1
(averaged over seeds, symbols with >= 1 positive window).

Usage:
    python scripts/paired_bootstrap.py --a at_pre_lf10_s{0,1,2} \
        --b at_scratch_lf10_s{0,1,2} --name gate3_lf10
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.training.common import load_scores, load_split  # noqa: E402
from src.training.protocol import atomic_write_json, git_sha  # noqa: E402


def f1_counts(y, pred):
    tp = np.sum(y & pred); fp = np.sum(~y & pred); fn = np.sum(y & ~pred)
    return tp, fp, fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--clean", default="adaptive")
    ap.add_argument("--a", nargs="+", required=True, help="tags of arm A (seeds)")
    ap.add_argument("--b", nargs="+", required=True, help="tags of arm B (seeds)")
    ap.add_argument("--name", required=True, help="output artifacts/<mode>/<name>.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if len(args.a) != len(args.b):
        raise SystemExit("--a and --b need the same number of seeds (paired)")

    cfg = yaml.safe_load(open(args.config))
    art = cfg["paths"]["artifacts_dir"]
    dd = os.path.join(cfg["paths"]["processed_dir"], args.clean)
    _, y, _, pair = load_split(dd, "test", scale=False)
    y = y.astype(bool)

    preds = {}
    for t in args.a + args.b:
        s, thr = load_scores(art, args.clean, t, y, pair)
        preds[t] = s >= thr

    symbols = np.unique(pair)
    idx_of = {s: np.where(pair == s)[0] for s in symbols}
    # per-symbol TP/FP/FN for each run -> bootstrap by summing counts
    C = {t: np.array([f1_counts(y[idx_of[s]], preds[t][idx_of[s]]) for s in symbols])
         for t in preds}

    def f1_from(c):
        tp, fp, fn = c
        d = 2 * tp + fp + fn
        return 0.0 if d == 0 else 2 * tp / d

    point = {t: f1_score(y, preds[t], zero_division=0) for t in preds}
    pairs = list(zip(args.a, args.b))
    diff_point = float(np.mean([point[a] - point[b] for a, b in pairs]))

    rng = np.random.default_rng(args.seed)
    n = len(symbols)
    diffs, fa_list, fb_list = [], [], []
    for _ in range(args.n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n)   # symbol multiplicity
        fa = [f1_from(w @ C[a]) for a, _ in pairs]
        fb = [f1_from(w @ C[b]) for _, b in pairs]
        diffs.append(np.mean(fa) - np.mean(fb))
        fa_list.append(np.mean(fa)); fb_list.append(np.mean(fb))
    lo, hi = np.percentile(diffs, [2.5, 97.5])

    # Wilcoxon on per-symbol F1 (mean over seeds), symbols with positives
    has_pos = np.array([y[idx_of[s]].any() for s in symbols])
    per_sym = lambda tags: np.mean([[f1_from(C[t][i]) for i in range(n)]
                                    for t in tags], axis=0)
    sa, sb = per_sym(args.a)[has_pos], per_sym(args.b)[has_pos]
    try:
        from scipy.stats import wilcoxon
        wp = float(wilcoxon(sa, sb).pvalue) if np.any(sa != sb) else 1.0
    except ImportError:
        wp = None

    ma = [point[t] for t in args.a]; mb = [point[t] for t in args.b]
    res = {
        "a": args.a, "b": args.b, "clean": args.clean,
        "f1_a": [round(v, 4) for v in ma], "f1_b": [round(v, 4) for v in mb],
        "mean_a": round(float(np.mean(ma)), 4), "std_a": round(float(np.std(ma)), 4),
        "mean_b": round(float(np.mean(mb)), 4), "std_b": round(float(np.std(mb)), 4),
        "mean_diff": round(diff_point, 4),
        "diff_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "p_boot_diff_le_0": round(float(np.mean(np.array(diffs) <= 0)), 4),
        "ci_a": [round(float(v), 4) for v in np.percentile(fa_list, [2.5, 97.5])],
        "ci_b": [round(float(v), 4) for v in np.percentile(fb_list, [2.5, 97.5])],
        "wilcoxon_p": None if wp is None else round(wp, 5),
        "n_symbols": int(n), "n_symbols_with_pos": int(has_pos.sum()),
        "n_boot": args.n_boot,
        "outside_seed_noise": bool(diff_point > max(np.std(ma), np.std(mb))),
        "significant": bool(lo > 0),
        "git_sha": git_sha(),
    }
    out = os.path.join(art, args.clean, f"{args.name}.json")
    atomic_write_json(res, out)

    print(f"=== PAIRED SYMBOL BOOTSTRAP [{args.clean}] {args.name} ===")
    print(f"A {res['mean_a']:.4f} ± {res['std_a']:.4f}  {res['f1_a']}")
    print(f"B {res['mean_b']:.4f} ± {res['std_b']:.4f}  {res['f1_b']}")
    print(f"mean F1 gap A-B: {res['mean_diff']:+.4f}  95% CI "
          f"[{res['diff_ci95'][0]:+.4f}, {res['diff_ci95'][1]:+.4f}]  "
          f"P(gap<=0)={res['p_boot_diff_le_0']}")
    print(f"Wilcoxon (per-symbol F1, {res['n_symbols_with_pos']} symbols): p={res['wilcoxon_p']}")
    print(f"outside seed noise: {res['outside_seed_noise']}   "
          f"CI excludes 0: {res['significant']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
