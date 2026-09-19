"""Label-efficiency study (M16) -- the key evidence for Contribution 2.

For every label fraction in [0.10, 0.25, 0.50, 1.00] and every seed,
trains {at_pre, at_scratch, cnnbilstm} and records val+test F1.

The money-shot figure: F1 vs label fraction. If pretraining pays off under
label scarcity, at_pre curve must dominate at_scratch at low fractions.

Usage:
    python -m src.training.label_efficiency --clean adaptive --seeds 3
    python -m src.training.label_efficiency --clean adaptive --seeds 5  # full
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.models.anomaly_transformer import ATClassifier, AnomalyTransformer
from src.models.deep_baselines import CNNBiLSTM
from src.training.common import (EarlyStop, load_split, subsample_labels,
                                 tune_threshold, metric_table)

FRACTIONS = [0.10, 0.25, 0.50, 1.00]


def forward_logits(model, xb):
    out = model(xb)
    return out[0] if isinstance(out, tuple) else out


def make_at(win, d_in, M, pre_path, dev):
    enc = AnomalyTransformer(win, d_in, M["d_model"], M["n_heads"],
                             M["n_layers"], M["dropout"])
    if pre_path and os.path.exists(pre_path):
        ck = torch.load(pre_path, map_location="cpu")
        enc.load_state_dict(ck["state_dict"], strict=True)
    return ATClassifier(enc, M["d_model"], M["dropout"]).to(dev)


def make_cnn(d_in, dev):
    return CNNBiLSTM(d_in).to(dev)


@torch.no_grad()
def predict_scores(model, X, dev, bs=256):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(dev)
        out.append(torch.sigmoid(forward_logits(model, xb)).cpu().numpy())
    return np.concatenate(out)


def run_one(model, Xtr, ytr, Xv, yv, Xte, yte, TR, dev):
    """Train a model and return (val_f1, test_f1, thr)."""
    pw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    crit = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pw], device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=TR["lr"])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr),
                      torch.from_numpy(ytr.astype(np.float32))),
        batch_size=TR["batch_size"], shuffle=True)

    stopper = EarlyStop(TR["patience"])
    best_state, best_thr = None, 0.5

    for ep in range(TR["finetune_epochs"]):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            loss = crit(forward_logits(model, xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        sv = predict_scores(model, Xv, dev)
        thr, f1v = tune_threshold(yv, sv)
        if stopper.step(f1v):
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            best_thr = thr
        if stopper.should_stop:
            break

    # final test eval with best val checkpoint
    model.load_state_dict(best_state)
    st = predict_scores(model, Xte, dev)
    test_res = metric_table(yte, st, best_thr)
    return stopper.best, test_res["f1"], best_thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--clean", default="adaptive")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    M, TR = cfg["model"], cfg["train"]
    dd = os.path.join(cfg["paths"]["processed_dir"], args.clean)
    art = os.path.join(cfg["paths"]["artifacts_dir"], args.clean)
    os.makedirs(art, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== LABEL-EFFICIENCY STUDY [{args.clean}] seeds={args.seeds} "
          f"device={dev} ===")

    Xtr, ytr, *_ = load_split(dd, "train")
    Xv, yv, *_ = load_split(dd, "val")
    Xte, yte, *_ = load_split(dd, "test")
    win, d_in = Xtr.shape[1], Xtr.shape[2]
    pre_path = os.path.join(art, "at_pretrained.pt")

    results = {name: {frac: [] for frac in FRACTIONS}
               for name in ("at_pre", "at_scratch", "cnnbilstm")}

    for frac in FRACTIONS:
        for seed in range(args.seeds):
            torch.manual_seed(seed); np.random.seed(seed)
            Xs, ys = subsample_labels(Xtr, ytr, frac, seed)
            print(f"  frac={frac:.2f}  seed={seed}  "
                  f"N={len(ys):,} pos={int(ys.sum()):,}", end="  ")

            # at_pre
            m = make_at(win, d_in, M, pre_path, dev)
            _, tf, _ = run_one(m, Xs, ys, Xv, yv, Xte, yte, TR, dev)
            results["at_pre"][frac].append(tf)

            # at_scratch
            m = make_at(win, d_in, M, None, dev)
            _, tf, _ = run_one(m, Xs, ys, Xv, yv, Xte, yte, TR, dev)
            results["at_scratch"][frac].append(tf)

            # cnnbilstm
            m = make_cnn(d_in, dev)
            _, tf, _ = run_one(m, Xs, ys, Xv, yv, Xte, yte, TR, dev)
            results["cnnbilstm"][frac].append(tf)

            print(f"at_pre={results['at_pre'][frac][-1]:.3f}  "
                  f"at_scratch={results['at_scratch'][frac][-1]:.3f}  "
                  f"cnn={results['cnnbilstm'][frac][-1]:.3f}")

    # Aggregate and save
    summary = {}
    print("\n=== LABEL-EFFICIENCY RESULTS (test F1, mean ± std) ===")
    header = f"{'frac':>6}  {'at_pre':>14}  {'at_scratch':>14}  {'cnnbilstm':>14}"
    print(header)
    for frac in FRACTIONS:
        row = f"{frac:>6.2f}"
        for name in ("at_pre", "at_scratch", "cnnbilstm"):
            vals = results[name][frac]
            mu, std = np.mean(vals), np.std(vals)
            row += f"  {mu:.4f}±{std:.4f}"
            summary[f"{name}_{frac}"] = {"mean": round(float(mu), 4),
                                          "std": round(float(std), 4),
                                          "vals": [round(v, 4) for v in vals]}
        print(row)

    out = os.path.join(art, "label_efficiency.json")
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\nSaved -> {out}")

    # Quick Gate 3 check at 10% labels
    pre_10 = np.mean(results["at_pre"][0.10])
    scr_10 = np.mean(results["at_scratch"][0.10])
    print("\n" + "=" * 62)
    print(f"GATE 3 (label-efficiency probe at 10% labels):")
    print(f"  at_pre  mean F1 = {pre_10:.4f}")
    print(f"  at_scratch mean F1 = {scr_10:.4f}")
    if pre_10 > scr_10:
        print("  PASS -- pretraining helps under label scarcity.")
        print("  Contribution 2 is real. Figure 2 will show this clearly.")
    else:
        print("  FAIL -- pretraining does not help at 10% labels.")
        print("  Check full-label result; if also tied, see PILOT_PLAN §6-D.")
    print("=" * 62)


if __name__ == "__main__":
    main()
