"""Baselines: Families 1 (domain/rule + microstructure) and 2 (classical ML).

  B1  rule-based threshold        (Kamps & Kleinberg 2018 style)
  B2  rush-order rule             (La Morgia 2020 style, adapted)
  B3  microstructure trio         (OFI, VPIN-proxy, Amihud) + logistic head
  B4  Random Forest               <-- GATE 2 lives here
  B5  Logistic Regression         (linear ceiling check)
  B6  XGBoost                     (strongest tree baseline; optional import)

All are re-implemented and run under OUR protocol: chronological split,
threshold tuned on val, NO point adjustment. Never copy numbers from papers.

Usage:
    python -m src.models.baselines --clean adaptive
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data.features import FEATURES
from src.training.common import (full_eval, load_split, save_result,
                                 tune_threshold)

I_RET = FEATURES.index("ret")
I_ZRET = FEATURES.index("z_ret")
I_ZVOL = FEATURES.index("z_vol")
I_ZNTR = FEATURES.index("z_ntr")
I_TBR = FEATURES.index("taker_buy_ratio")
I_LVOL = FEATURES.index("log_vol")


# ---------------------------------------------------------------- B1 rule
def fit_rule(Xv: np.ndarray, yv: np.ndarray) -> tuple[float, float]:
    """Grid-search (a, b) on validation: alert if max z_ret>a AND max z_vol>b."""
    rv, vv = Xv[:, :, I_ZRET].max(1), Xv[:, :, I_ZVOL].max(1)
    best, best_f1 = (2.0, 2.0), -1.0
    for a in np.arange(0.5, 8.01, 0.25):
        for b in np.arange(0.5, 8.01, 0.25):
            pred = (rv > a) & (vv > b)
            tp = int((pred & (yv == 1)).sum())
            fp = int((pred & (yv == 0)).sum())
            fn = int(((~pred) & (yv == 1)).sum())
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            if f1 > best_f1:
                best_f1, best = f1, (float(a), float(b))
    return best


def rule_score(X: np.ndarray, a: float, b: float) -> np.ndarray:
    """Continuous margin (so ROC-AUC is meaningful), not just a boolean."""
    return np.minimum(X[:, :, I_ZRET].max(1) - a, X[:, :, I_ZVOL].max(1) - b)


# ------------------------------------------------- B2 rush-order style rule
def rush_order_score(X: np.ndarray) -> np.ndarray:
    """La Morgia's key signal: a burst of aggressive BUY orders.

    Their StdRushOrder needs order types, which Binance does not expose
    publicly. We approximate the same construct: standardized trade-count
    surge weighted by taker-buy aggression.
    """
    buy_pressure = X[:, :, I_TBR] - 0.5          # >0 = taker buying
    surge = X[:, :, I_ZNTR]                      # trade-count z-score
    return (np.maximum(surge, 0) * np.maximum(buy_pressure, 0)).max(1)


# ------------------------------------------------ B3 microstructure trio
def microstructure_features(X: np.ndarray) -> np.ndarray:
    """OFI / VPIN-proxy / Amihud, computed per window -> (N, 6).

    OFI    : order-flow imbalance = sum of signed volume
    VPIN   : |buy - sell| / total volume (volume-synchronised toxicity proxy)
    Amihud : mean(|ret|) / mean(volume) -- price impact per unit volume
    """
    vol = np.expm1(X[:, :, I_LVOL])              # undo log1p
    tbr = X[:, :, I_TBR]
    ret = X[:, :, I_RET]

    buy = vol * tbr
    sell = vol * (1.0 - tbr)

    ofi = (buy - sell).sum(1)
    ofi_max = (buy - sell).max(1)

    tot = vol.sum(1) + 1e-9
    vpin = np.abs(buy.sum(1) - sell.sum(1)) / tot
    vpin_late = (np.abs(buy[:, -24:].sum(1) - sell[:, -24:].sum(1))
                 / (vol[:, -24:].sum(1) + 1e-9))     # last 2 minutes

    amihud = np.abs(ret).mean(1) / (vol.mean(1) + 1e-9)
    amihud_max = np.abs(ret).max(1) / (vol.mean(1) + 1e-9)

    return np.stack([ofi, ofi_max, vpin, vpin_late, amihud, amihud_max], 1)


# ------------------------------------------------------ classical ML feats
def window_stats(X: np.ndarray) -> np.ndarray:
    """Per-window summary: mean/std/max/min/last of each feature -> (N, 5F)."""
    return np.concatenate(
        [X.mean(1), X.std(1), X.max(1), X.min(1), X[:, -1, :]], axis=1)


# ------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--clean", default="adaptive")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dd = os.path.join(cfg["paths"]["processed_dir"], args.clean)
    art = cfg["paths"]["artifacts_dir"]
    labels_csv = cfg["paths"]["labels_csv"]

    Xtr, ytr, etr, ptr = load_split(dd, "train", scale=False)
    Xv, yv, ev, pv = load_split(dd, "val", scale=False)
    Xte, yte, ete, pte = load_split(dd, "test", scale=False)
    print(f"[{args.clean}] train={len(ytr):,} val={len(yv):,} test={len(yte):,}"
          f"  pos_rate_train={ytr.mean():.3%}")

    def report(tag, s_val, s_te):
        thr, f1v = tune_threshold(yv, s_val)
        res = full_eval(yte, s_te, thr, ete, pte, labels_csv, cfg)
        save_result(art, args.clean, tag, res)
        print(f"  [{tag:14s}] valF1={f1v:.4f} | testF1={res['f1']:.4f} "
              f"P={res['precision']:.4f} R={res['recall']:.4f} "
              f"AUC={res['roc_auc']:.4f} evR={res['event_recall']:.4f}")
        return res

    print(f"\n--- Family 1: domain / rule / microstructure ---")

    a, b = fit_rule(Xv, yv)
    print(f"  rule grid -> a={a:.2f}, b={b:.2f}")
    report("b1_rule", rule_score(Xv, a, b), rule_score(Xte, a, b))

    report("b2_rushorder", rush_order_score(Xv), rush_order_score(Xte))

    Mtr, Mv, Mte = (microstructure_features(A) for A in (Xtr, Xv, Xte))
    sc = StandardScaler().fit(Mtr)
    lr_micro = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr_micro.fit(sc.transform(Mtr), ytr)
    report("b3_microstructure",
           lr_micro.predict_proba(sc.transform(Mv))[:, 1],
           lr_micro.predict_proba(sc.transform(Mte))[:, 1])

    print(f"\n--- Family 2: classical ML (50-dim window stats) ---")
    Str, Sv, Ste = (window_stats(A) for A in (Xtr, Xv, Xte))

    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                n_jobs=-1, random_state=42)
    rf.fit(Str, ytr)
    res_rf = report("b4_randomforest",
                    rf.predict_proba(Sv)[:, 1], rf.predict_proba(Ste)[:, 1])

    sc2 = StandardScaler().fit(Str)
    lr = LogisticRegression(max_iter=3000, class_weight="balanced")
    lr.fit(sc2.transform(Str), ytr)
    report("b5_logreg",
           lr.predict_proba(sc2.transform(Sv))[:, 1],
           lr.predict_proba(sc2.transform(Ste))[:, 1])

    try:
        from xgboost import XGBClassifier
        spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
        xgb = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            scale_pos_weight=spw, eval_metric="logloss",
                            n_jobs=-1, random_state=42)
        xgb.fit(Str, ytr)
        report("b6_xgboost",
               xgb.predict_proba(Sv)[:, 1], xgb.predict_proba(Ste)[:, 1])
    except ImportError:
        print("  [b6_xgboost    ] skipped (pip install xgboost)")

    # ---------------------------------------------------------- GATE 2
    thr_rf, f1v_rf = tune_threshold(yv, rf.predict_proba(Sv)[:, 1])
    print("\n" + "=" * 62)
    print(f"GATE 2 -- Random Forest validation F1 = {f1v_rf:.4f}")
    if f1v_rf >= 0.60:
        print("GATE 2: PASS (>= 0.60). Signal exists. Proceed to deep models.")
    else:
        print("GATE 2: FAIL (< 0.60). DO NOT build deep models yet.")
        print("  Debug: (a) re-check Gate 1 alignment,")
        print("         (b) try during-only labels [t, t+5min],")
        print("         (c) inspect feature distributions pos vs neg.")
        print("  Never 'fix' a data problem with a bigger model.")
    print("=" * 62)


if __name__ == "__main__":
    main()
