---
name: bootstrap-ci
description: Statistical rigor helper — computes bootstrap CIs resampled over SYMBOLS (not windows), and paired significance tests between OURS and each baseline. Use when a result JSON exists and you need CI or p-value for a paper claim.
---

# Skill: bootstrap-ci

Enforces the statistical protocol that top venues expect.

## When to use

- Before quoting any F1 number in the paper
- Before making a "significantly better than baseline" claim
- When user asks "is X vs Y difference real?"
- For every headline result in Table 1 and Table 2

## Why symbol-level bootstrap (not window-level)

- Windows from the same pump event are highly correlated
- Window-level bootstrap gives artificially narrow CIs (fake confidence)
- Reviewers who know statistics will catch this immediately
- Resampling over unique symbols is the correct choice

## Protocol

1. Load `y_true, scores, pair, threshold` from result JSON
2. For each of 1000 bootstrap iterations:
   - Sample unique symbols with replacement (same count as original)
   - Concatenate all window indices belonging to sampled symbols
   - Compute F1 on that subsample
3. Report 2.5% and 97.5% percentiles as [f1_lo, f1_hi]
4. For paired test between two models:
   - Sample same symbols for both models
   - Compute F1 difference
   - Report 2.5% percentile of difference distribution
   - If ≥ 0 → OURS is significantly better

## Paired significance test (for "OURS vs each baseline")

Wilcoxon signed-rank on per-symbol F1 scores:

```python
from scipy.stats import wilcoxon
stat, p_val = wilcoxon(ours_symbol_f1s, baseline_symbol_f1s)
```

Report `p_val` for each baseline in Table 1's caption or a supplementary table.

## Output format

```
=== BOOTSTRAP CI: at_pre on adaptive ===
Point estimate F1: 0.847
95% CI:            [0.812, 0.883]
n_symbols:         85

=== PAIRED TEST vs CNN-BiLSTM ===
Mean per-symbol F1 gap: +0.043
Bootstrap 95% CI on gap: [+0.018, +0.071]  (does not cross 0)
Wilcoxon p-value:        0.003
CONCLUSION: OURS significantly better than CNN-BiLSTM (p < 0.01)
```

## Rule

Never quote a single F1 number in the paper without accompanying CI.
