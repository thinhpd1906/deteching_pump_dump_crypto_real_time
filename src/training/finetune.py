"""PHASE 2 -- Supervised training / fine-tuning on labeled windows.

One script covers every supervised row of the comparison table:

  D  OURS      : --arch at --pretrained          AT encoder init from pretrain
  B9 twin      : --arch at                       same arch, RANDOM init
  B8 baseline  : --arch cnnbilstm                Chadalapaka-style

GATE 3 is  at_pre > at_scratch  (the twin isolates the effect of pretraining,
controlling for architecture -- this is the single most important comparison
in the thesis).

Imbalance is handled twice:
  1. pos_weight in BCE  (= n_neg / n_pos, computed from the train split)
  2. decision threshold TUNED ON VALIDATION F1 -- never 0.5, never on test

Label-efficiency study (Contribution 2 evidence):
    --label-frac 0.1   trains on 10% of labels

Usage:
    python -m src.training.finetune --clean adaptive --arch cnnbilstm
    python -m src.training.finetune --clean adaptive --arch at
    python -m src.training.finetune --clean adaptive --arch at --pretrained
    python -m src.training.finetune --clean adaptive --arch at --pretrained \
        --label-frac 0.1 --seed 1
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.models.anomaly_transformer import ATClassifier, AnomalyTransformer
from src.models.deep_baselines import CNNBiLSTM
from src.training.common import (EarlyStop, full_eval, load_split, save_result,
                                 subsample_labels, tune_threshold)


def build_model(arch: str, win: int, d_in: int, M: dict,
                pretrained_path: str | None, dev: str):
    if arch == "at":
        enc = AnomalyTransformer(win, d_in, M["d_model"], M["n_heads"],
                                 M["n_layers"], M["dropout"])
        if pretrained_path:
            if not os.path.exists(pretrained_path):
                raise SystemExit(
                    f"Pretrained encoder not found: {pretrained_path}\n"
                    f"Run: python -m src.training.pretrain --clean <mode>")
            ck = torch.load(pretrained_path, map_location="cpu")
            enc.load_state_dict(ck["state_dict"], strict=True)
            print(f"  loaded pretrained encoder <- {pretrained_path}")
        return ATClassifier(enc, M["d_model"], M["dropout"]).to(dev)

    if arch == "cnnbilstm":
        return CNNBiLSTM(d_in).to(dev)

    raise ValueError(f"unknown arch: {arch}")


def forward_logits(model, xb):
    out = model(xb)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def predict_scores(model, X, dev, bs=256) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(dev)
        outs.append(torch.sigmoid(forward_logits(model, xb)).cpu().numpy())
    return np.concatenate(outs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--clean", default="adaptive")
    ap.add_argument("--arch", default="at", choices=["at", "cnnbilstm"])
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--label-frac", type=float, default=1.0,
                    help="fraction of labels to train on (label-efficiency)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    M, TR = cfg["model"], cfg["train"]
    seed = args.seed if args.seed is not None else TR["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    dd = os.path.join(cfg["paths"]["processed_dir"], args.clean)
    art_root = cfg["paths"]["artifacts_dir"]
    art = os.path.join(art_root, args.clean)
    os.makedirs(art, exist_ok=True)
    labels_csv = cfg["paths"]["labels_csv"]

    # tag encodes everything needed to identify this run
    base = f"{args.arch}" + ("_pre" if args.pretrained else "_scratch")
    tag = args.tag or base
    if args.label_frac < 1.0:
        tag += f"_lf{int(args.label_frac * 100)}"
    if args.seed is not None:
        tag += f"_s{seed}"

    Xtr, ytr, *_ = load_split(dd, "train")
    Xv, yv, ev, pv = load_split(dd, "val")
    Xte, yte, ete, pte = load_split(dd, "test")

    if args.label_frac < 1.0:
        Xtr, ytr = subsample_labels(Xtr, ytr, args.label_frac, seed)

    win, d_in = Xtr.shape[1], Xtr.shape[2]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    pre_path = os.path.join(art, "at_pretrained.pt") if args.pretrained else None
    model = build_model(args.arch, win, d_in, M, pre_path, dev)

    if TR["pos_class_weight"] == "auto":
        pw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    else:
        pw = float(TR["pos_class_weight"])

    crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=TR["lr"])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr),
                      torch.from_numpy(ytr.astype(np.float32))),
        batch_size=TR["batch_size"], shuffle=True)

    epochs = args.epochs or TR["finetune_epochs"]
    stopper = EarlyStop(TR["patience"])
    best_path = os.path.join(art, f"{tag}.pt")

    print(f"=== FINETUNE [{tag}] on '{args.clean}' ===")
    print(f"  train N={len(ytr):,} (pos={int(ytr.sum()):,}, "
          f"label_frac={args.label_frac})")
    print(f"  pos_weight={pw:.1f}  device={dev}  seed={seed}  epochs<={epochs}")

    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            loss = crit(forward_logits(model, xb), yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(xb)

        s_val = predict_scores(model, Xv, dev)
        thr, f1v = tune_threshold(yv, s_val)

        flag = ""
        if stopper.step(f1v):
            torch.save({"state_dict": model.state_dict(), "thr": thr,
                        "arch": args.arch, "win": win, "d_in": d_in,
                        "model_cfg": M}, best_path)
            flag = "  <- saved"
        print(f"  ep {ep:02d}  loss={tot / len(ytr):.4f}  "
              f"valF1={f1v:.4f}  thr={thr:.4f}{flag}")

        if stopper.should_stop:
            print(f"  early stop (no val improvement for {TR['patience']} epochs)")
            break

    # ---- final evaluation: test set touched EXACTLY ONCE, with saved threshold
    ck = torch.load(best_path, map_location=dev)
    model.load_state_dict(ck["state_dict"])
    s_te = predict_scores(model, Xte, dev)
    res = full_eval(yte, s_te, ck["thr"], ete, pte, labels_csv, cfg)
    res["val_f1"] = round(stopper.best, 4)
    res["label_frac"] = args.label_frac
    res["seed"] = seed
    save_result(art_root, args.clean, tag, res)

    print(f"\n  TEST  F1={res['f1']:.4f}  P={res['precision']:.4f}  "
          f"R={res['recall']:.4f}  ROC={res['roc_auc']:.4f}  "
          f"PR={res['pr_auc']:.4f}")
    print(f"  EVENT recall={res['event_recall']:.4f} "
          f"({res['events_detected']}/{res['events_total']})  "
          f"FA/symbol-day={res['fa_per_symbol_day']}")
    print(f"  F1 95% CI (bootstrap over symbols): "
          f"[{res['f1_lo']:.4f}, {res['f1_hi']:.4f}]")
    print(f"  -> {art}/{tag}_test.json")


if __name__ == "__main__":
    main()
