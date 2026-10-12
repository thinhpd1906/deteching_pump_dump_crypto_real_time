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
    --label-frac 0.1   trains on 10% of labelled EVENTS

Resumable (for time-limited Kaggle sessions):
  * the full training state is written to {tag}_last.pt after every epoch;
    rerunning the same command resumes from it automatically
  * --deadline <unix seconds>: if the next epoch would not finish before the
    deadline, save state and exit with code 75 (EXIT_INCOMPLETE)
  * a run whose {tag}_test.json already exists is skipped

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
import sys
import time

import numpy as np
import torch
import yaml

from src.models.anomaly_transformer import ATClassifier, AnomalyTransformer
from src.models.deep_baselines import CNNBiLSTM
from src.training.common import (EarlyStop, full_eval, load_split, save_result,
                                 subsample_labels, tune_threshold)

EXIT_INCOMPLETE = 75


def run_tag(arch: str, pretrained: bool, label_frac: float,
            seed: int | None) -> str:
    """Result tag; the JSON is artifacts/<mode>/<tag>_test.json."""
    tag = arch + ("_pre" if pretrained else "_scratch")
    if label_frac < 1.0:
        tag += f"_lf{int(round(label_frac * 100))}"
    if seed is not None:
        tag += f"_s{seed}"
    return tag


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
def predict_scores(model, X, dev, bs=2048, amp=False) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logit = forward_logits(model, xb)
        outs.append(torch.sigmoid(logit.float()).cpu().numpy())
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
                    help="fraction of labelled events to train on")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--deadline", type=float, default=None,
                    help="unix time; save state and exit 75 before it")
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

    tag = args.tag or run_tag(args.arch, args.pretrained, args.label_frac,
                              args.seed)
    if os.path.exists(os.path.join(art, f"{tag}_test.json")):
        print(f"[{tag}] result already exists -- skipping")
        return

    Xtr, ytr, etr, ptr = load_split(dd, "train")
    Xv, yv, ev, pv = load_split(dd, "val")

    if args.label_frac < 1.0:
        W = cfg["windows"]
        gap_ns = (W["pre_minutes"] + W["during_minutes"]) * 60 * 10**9
        Xtr, ytr = subsample_labels(Xtr, ytr, args.label_frac, seed,
                                    etr, ptr, gap_ns)

    win, d_in = Xtr.shape[1], Xtr.shape[2]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = bool(TR.get("amp", False)) and dev == "cuda"
    if dev == "cuda":
        torch.backends.cudnn.benchmark = True

    pre_path = os.path.join(art, "at_pretrained.pt") if args.pretrained else None
    model = build_model(args.arch, win, d_in, M, pre_path, dev)

    if TR["pos_class_weight"] == "auto":
        pw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    else:
        pw = float(TR["pos_class_weight"])

    crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=TR["lr"])
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # whole train split lives on the device: no per-sample DataLoader overhead
    Xg = torch.from_numpy(Xtr).to(dev)
    yg = torch.from_numpy(ytr.astype(np.float32)).to(dev)
    gen = torch.Generator(device=dev).manual_seed(seed)
    bs = TR["batch_size"]

    epochs = args.epochs or TR["finetune_epochs"]
    stopper = EarlyStop(TR["patience"])
    best_path = os.path.join(art, f"{tag}.pt")
    last_path = os.path.join(art, f"{tag}_last.pt")
    start_ep = 1

    if os.path.exists(last_path):
        st = torch.load(last_path, map_location=dev)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        stopper.best, stopper.count = st["best"], st["count"]
        stopper.should_stop = stopper.count >= stopper.patience
        start_ep = st["epoch"] + 1
        print(f"  resumed from {last_path} (epoch {st['epoch']} done, "
              f"best valF1={stopper.best:.4f})")

    print(f"=== FINETUNE [{tag}] on '{args.clean}' ===")
    print(f"  train N={len(ytr):,} (pos={int(ytr.sum()):,}, "
          f"label_frac={args.label_frac})")
    print(f"  pos_weight={pw:.1f}  device={dev}  amp={amp}  seed={seed}  "
          f"epochs<={epochs}")

    ep_time = None
    ep = start_ep
    while ep <= epochs and not stopper.should_stop:
        if (args.deadline and ep_time
                and time.time() + 1.3 * ep_time + 120 > args.deadline):
            print(f"  deadline near -- stopping before epoch {ep}; "
                  f"rerun to resume")
            sys.exit(EXIT_INCOMPLETE)

        t0 = time.time()
        model.train()
        tot = 0.0
        perm = torch.randperm(len(yg), device=dev, generator=gen)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            xb, yb = Xg[idx], yg[idx]
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logit = forward_logits(model, xb)
            loss = crit(logit.float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(idx)

        s_val = predict_scores(model, Xv, dev, amp=amp)
        thr, f1v = tune_threshold(yv, s_val)

        flag = ""
        if stopper.step(f1v):
            torch.save({"state_dict": model.state_dict(), "thr": thr,
                        "arch": args.arch, "win": win, "d_in": d_in,
                        "model_cfg": M}, best_path)
            flag = "  <- saved"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "best": stopper.best,
                    "count": stopper.count, "epoch": ep}, last_path)
        ep_time = time.time() - t0
        print(f"  ep {ep:02d}  loss={tot / len(ytr):.4f}  "
              f"valF1={f1v:.4f}  thr={thr:.4f}  ({ep_time:.0f}s){flag}",
              flush=True)
        ep += 1

    if stopper.should_stop:
        print(f"  early stop (no val improvement for {TR['patience']} epochs)")

    # ---- final evaluation: test set touched EXACTLY ONCE, with saved threshold
    del Xg, yg
    Xte, yte, ete, pte = load_split(dd, "test")
    ck = torch.load(best_path, map_location=dev)
    model.load_state_dict(ck["state_dict"])
    s_te = predict_scores(model, Xte, dev, amp=amp)
    res = full_eval(yte, s_te, ck["thr"], ete, pte, labels_csv, cfg)
    res["val_f1"] = round(float(stopper.best), 4)
    res["label_frac"] = args.label_frac
    res["seed"] = seed
    save_result(art_root, args.clean, tag, res)
    os.remove(last_path)

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
