"""PHASE 2b -- Unsupervised-only detection (Family 3, Approach C).

Two detectors trained WITHOUT labels, evaluated WITH them (standard
unsupervised-AD evaluation: labels only for threshold tuning + reporting).

  B7  LSTM Autoencoder   : train on unlabeled pool, score = recon MSE
  B10 Pretrained AT      : uses anomaly_score() from the saved pretrained encoder

Usage:
    python -m src.training.eval_unsupervised --clean adaptive
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.models.anomaly_transformer import AnomalyTransformer, anomaly_score
from src.models.deep_baselines import LSTMAE
from src.training.common import (full_eval, load_pretrain_pool, load_split,
                                 save_result, tune_threshold)


@torch.no_grad()
def _batched_at_scores(model, X, dev, bs=256):
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(dev)
        out.append(anomaly_score(model, xb).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def _batched_ae_scores(ae, X, dev, bs=256):
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(dev)
        out.append(ae.score(xb).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--clean", default="adaptive")
    ap.add_argument("--ae-epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    TR = cfg["train"]
    seed = args.seed if args.seed is not None else TR["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    dd = os.path.join(cfg["paths"]["processed_dir"], args.clean)
    art_root = cfg["paths"]["artifacts_dir"]
    art = os.path.join(art_root, args.clean)
    os.makedirs(art, exist_ok=True)
    labels_csv = cfg["paths"]["labels_csv"]

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    Xv, yv, ev, pv = load_split(dd, "val")
    Xte, yte, ete, pte = load_split(dd, "test")

    # ---------------------------------------------------------------- B10: AT unsup
    ck_path = os.path.join(art, "at_pretrained.pt")
    if os.path.exists(ck_path):
        print("=== B10: Anomaly Transformer (unsupervised) ===")
        ck = torch.load(ck_path, map_location=dev)
        m = AnomalyTransformer(
            ck["win"], ck["d_in"],
            ck["model_cfg"]["d_model"], ck["model_cfg"]["n_heads"],
            ck["model_cfg"]["n_layers"]).to(dev)
        m.load_state_dict(ck["state_dict"])
        m.eval()

        sv = _batched_at_scores(m, Xv, dev)
        thr, f1v = tune_threshold(yv, sv)
        st = _batched_at_scores(m, Xte, dev)
        res = full_eval(yte, st, thr, ete, pte, labels_csv, cfg)
        save_result(art_root, args.clean, "b10_at_unsup", res)
        print(f"  val F1={f1v:.4f} | test F1={res['f1']:.4f} "
              f"AUC={res['roc_auc']:.4f} evR={res['event_recall']:.4f}")
    else:
        print("  [B10 skipped] run pretrain first: python -m src.training.pretrain")

    # ---------------------------------------------------------------- B7: LSTM-AE
    print("\n=== B7: LSTM Autoencoder (unsupervised) ===")
    Xu = load_pretrain_pool(dd)
    d_in = Xu.shape[2]
    ae = LSTMAE(d_in).to(dev)
    opt = torch.optim.AdamW(ae.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xu)),
                        batch_size=TR["batch_size"], shuffle=True)

    for ep in range(args.ae_epochs):
        ae.train()
        tot, n = 0.0, 0
        t0 = time.time()
        for (xb,) in loader:
            xb = xb.to(dev)
            loss = torch.nn.functional.mse_loss(ae(xb), xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(xb); n += len(xb)
        print(f"  epoch {ep+1:02d}/{args.ae_epochs}  mse={tot/n:.6f}  "
              f"({time.time()-t0:.1f}s)")

    sv = _batched_ae_scores(ae, Xv, dev)
    thr, f1v = tune_threshold(yv, sv)
    st = _batched_ae_scores(ae, Xte, dev)
    res = full_eval(yte, st, thr, ete, pte, labels_csv, cfg)
    save_result(art_root, args.clean, "b7_lstmae", res)
    print(f"  val F1={f1v:.4f} | test F1={res['f1']:.4f} "
          f"AUC={res['roc_auc']:.4f} evR={res['event_recall']:.4f}")

    # Save trained AE for potential further analysis
    torch.save({"state_dict": ae.state_dict(), "d_in": d_in},
               os.path.join(art, "lstmae.pt"))
    print(f"\nResults saved -> {art}/b7_*/b10_*_test.json")


if __name__ == "__main__":
    main()
