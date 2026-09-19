---
name: experiment-runner
description: Runs a training configuration across multiple seeds and aggregates results. Handles the pattern "train N seeds, save each JSON, compute mean±std, save summary." Use whenever a headline number needs multiple seeds instead of a single run.
---

# Skill: experiment-runner

The "run N seeds, aggregate" pattern for reproducible experiments.

## When to use

- Every training run for a paper headline (never one seed alone)
- Label-efficiency study (fractions × seeds × models)
- Cleaning ablation (modes × seeds)
- Any comparison between two configurations

## Standard seed sweep protocol

```bash
for SEED in 0 1 2; do
  python -m src.training.finetune \
    --config config/config.yaml \
    --clean adaptive \
    --arch at --pretrained \
    --seed $SEED
done
```

Output: `at_pre_s0_test.json, at_pre_s1_test.json, at_pre_s2_test.json`

## Aggregation

For each metric (precision, recall, f1, roc_auc, event_recall):

- Load all `{tag}_s*_test.json` files matching the pattern
- Compute mean and std across seeds
- Save summary to `{tag}_summary.json`:
  ```json
  {
    "tag": "at_pre",
    "n_seeds": 3,
    "f1": {"mean": 0.847, "std": 0.012, "vals": [0.836, 0.859, 0.846]},
    "precision": {"mean": ..., "std": ..., "vals": [...]},
    ...
  }
  ```

## Rules

- Always use consecutive seeds starting from 0 for reproducibility
- Minimum 3 seeds for pilot, 5 seeds for final paper submission
- If any seed diverges (NaN loss), report it — don't silently exclude
- Save individual seed logs; don't just keep the summary

## Progress reporting during run

```
[experiment-runner] at_pre on adaptive, 3 seeds
  seed 0: valF1=0.828 → testF1=0.836 (12 min)
  seed 1: valF1=0.855 → testF1=0.859 (11 min)
  seed 2: valF1=0.841 → testF1=0.846 (13 min)

Summary: test F1 = 0.847 ± 0.012
Saved: artifacts/adaptive/at_pre_summary.json
```

## Anti-patterns to avoid

- Running many seeds until you see the number you want (p-hacking)
- Excluding "bad" seeds (only exclude documented divergence)
- Reporting best-of-N instead of mean±std
