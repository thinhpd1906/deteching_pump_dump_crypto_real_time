---
name: verify-scaler-parity
description: Verifies that the inference service produces byte-identical features to the training pipeline for the same input data. This is the guarantee that training-serving skew is zero. Runs a small end-to-end test with saved sample bars and asserts feature vectors match.
---

# Skill: verify-scaler-parity

Guards against training-serving skew.

## When to use

- After any change to `src/data/features.py`
- After any change to `src/pipeline/inference_service.py`
- Before deploying the pipeline for latency measurement
- When user asks "are online and offline features the same?"

## What it checks

1. Load a known cleaned tick dataset (e.g., first 1000 rows of BTCBTC 2018-01-01)
2. Offline path:
   - `trades_to_bars` → `bars_to_features` → apply scaler
3. Online path simulation:
   - Feed same bars one-by-one into `normalize_bars` + `bars_to_features`
   - Apply the same scaler
4. Assert final feature matrices are equal within 1e-6 tolerance

## Success criteria

- Feature vectors identical (float32 tolerance)
- Scaler mean/std match `data/processed/{mode}/scaler.json`
- Same window shape (120, 10)

## Common failure modes to look for

- Off-by-one in window indexing (online may have one fewer bar)
- ffill applied differently in online rolling buffer vs offline resample
- Scaler loaded from wrong mode directory
- Feature order mismatch (must match `FEATURES` list in features.py)

## Output format

```
=== SCALER PARITY CHECK ===
Loaded:       1000 sample trades from BTCUSDT 2020-01-01
Offline:      shape=(80, 120, 10)  mean=0.00023  std=1.00012
Online:       shape=(80, 120, 10)  mean=0.00023  std=1.00012
Max abs diff: 3.4e-8

VERDICT: PASS (identical within tolerance)
```

If FAIL, report exact diff location and stop — do NOT let user run deployment with skew.
