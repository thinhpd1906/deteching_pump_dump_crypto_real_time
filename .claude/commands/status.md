---
description: Show current thesis project status — which gates have passed, which experiments are done, what's next
---

Analyze the current state of the thesis project and report:

1. **Data status:**
   - Are labels downloaded? (`data/labels/pumps_binance.csv` exists?)
   - How many pair-days of Binance data are cached?
   - Which cleaning modes have been built?

2. **Gate status:**
   - Gate 1: check for `artifacts/gate1/gate1_results.csv`
   - Gate C: check for `artifacts/gate_c_results.json`
   - Gate 2: check for `artifacts/*/b4_randomforest_test.json`
   - Gate 3: check for both `at_pre_test.json` and `at_scratch_test.json`

3. **Experiments run:**
   - List all `_test.json` files in `artifacts/`
   - Note which have multiple seeds vs single seed

4. **Next recommended action:**
   - If Gate 1 not done → run `@data-agent` for Gate 1
   - If Gate 1 done, Gate 2 not done → `@baseline-agent`
   - If Gate 2 passed, deep models not trained → `@training-agent`
   - If all training done → `@evaluation-agent` for tables
   - If tables ready → `@paper-agent` for draft

Report in a compact status board format:

```
┌─ Thesis Status ─────────────────────────┐
│ Labels:      ✓ 338 events               │
│ Data:        ~4,200 pair-days cached    │
│ Gate 1:      ✓ PASS (8/10)              │
│ Gate C:      ✓ PASS (KS=0.087)          │
│ Gate 2:      ✓ PASS (RF F1=0.72)        │
│ Gate 3:      ⏳ pending                  │
│ Experiments: 8 runs across 3 seeds       │
│ Tables:      ⏳ not aggregated           │
│                                          │
│ NEXT: @training-agent for pretrain       │
└──────────────────────────────────────────┘
```
