"""PHASE 1 -- Self-supervised pretraining on UNLABELED windows.

This is Contribution 2. The encoder learns what NORMAL market microstructure
looks like from a large pool of unlabeled windows, before ever seeing a label.

Objective = Anomaly-Transformer minimax (reconstruction + association
discrepancy) PLUS random time-step masking (25%), so the model must infer
masked bars from context.

Sanity check at the end: mean unsupervised anomaly score on POSITIVE val
windows should exceed that on NEGATIVE val windows. If it doesn't, the
pretrained representation carries no P&D signal -- investigate before
fine-tuning.

Usage:
    python -m src.training.pretrain --clean adaptive
    python -m src.training.pretrain --clean adaptive --epochs 5   # pilot speed
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.models.anomaly_transformer import (AnomalyTransformer, anomaly_score,
                                            minimax_losses)
from src.training.common import load_pretrain_pool, load_split


def mask_input(x: torch.Tensor, ratio: float) -> torch.Tensor:
    """Zero out `ratio` of time steps (masked reconstruction)."""
    if ratio <= 0:
        return x
    B, T, _ = x.shape
    m = torch.rand(B, T, 1, device=x.device) < ratio
    return x.masked_fill(m, 0.0)


@torch.no_grad()
def _batched_scores(model, X, dev, bs=256):
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(dev)
        out.append(anomaly_score(model, xb).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--clean", default="adaptive")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    M, TR = cfg["model"], cfg["train"]
    seed = args.seed if args.seed is not None else TR["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    dd = os.path.join(cfg["paths"]["processed_dir"], args.clean)
    art = os.path.join(cfg["paths"]["artifacts_dir"], args.clean)
    os.makedirs(art, exist_ok=True)

    X = load_pretrain_pool(dd)
    win, d_in = X.shape[1], X.shape[2]

    loader = DataLoader(TensorDataset(torch.from_numpy(X)),
                        batch_size=TR["batch_size"], shuffle=True,
                        drop_last=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AnomalyTransformer(win, d_in, M["d_model"], M["n_heads"],
                               M["n_layers"], M["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=TR["lr"])
    epochs = args.epochs or TR["pretrain_epochs"]

    n_params = sum(p.numel() for p in model.parameters())
    print(f"=== SELF-SUPERVISED PRETRAINING [{args.clean}] ===")
    print(f"  unlabeled windows : {len(X):,}")
    print(f"  window / features : {win} x {d_in}")
    print(f"  params            : {n_params:,}")
    print(f"  device            : {dev}   epochs: {epochs}   seed: {seed}")
    print(f"  mask ratio        : {M['mask_ratio']}   lambda_k: {M['lambda_k']}")

    for ep in range(1, epochs + 1):
        model.train()
        t0, rec_sum, n = time.time(), 0.0, 0
        for (xb,) in loader:
            xb = xb.to(dev)
            x_in = mask_input(xb, M["mask_ratio"])

            x_hat, series_l, prior_l = model(x_in)
            loss_s, loss_p, rec = minimax_losses(
                xb, x_hat, series_l, prior_l, k=M["lambda_k"])

            opt.zero_grad()
            loss_s.backward(retain_graph=True)   # push series away from prior
            loss_p.backward()                    # pull prior toward series
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            rec_sum += rec.item() * len(xb)
            n += len(xb)
        print(f"  epoch {ep:02d}/{epochs}  recon={rec_sum / n:.6f}  "
              f"({time.time() - t0:.1f}s)")

    path = os.path.join(art, "at_pretrained.pt")
    torch.save({"state_dict": model.state_dict(), "win": win, "d_in": d_in,
                "model_cfg": M}, path)
    print(f"\n  saved encoder -> {path}")

    # ---- sanity: does the unsupervised score already separate pos from neg?
    Xv, yv, *_ = load_split(dd, "val")
    s = _batched_scores(model, Xv, dev)
    sp, sn = s[yv == 1].mean(), s[yv == 0].mean()
    print("\n" + "=" * 62)
    print(f"PRETRAIN SANITY: mean anomaly score")
    print(f"  positive (pump) windows : {sp:.6f}")
    print(f"  negative windows        : {sn:.6f}")
    if sp > sn:
        print("  OK -- pretrained representation separates pumps unsupervised.")
    else:
        print("  WARNING -- no unsupervised separation. Fine-tuning may still")
        print("  work, but investigate: more epochs / more data / mask 0.15.")
    print("=" * 62)


if __name__ == "__main__":
    main()
