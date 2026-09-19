---
description: Check the status of a specific gate (1, 2, 3, or C) and give the verdict
argument-hint: "1 | 2 | 3 | C"
---

Check gate $1 (from argument) and report PASS/FAIL/PENDING.

**Gate 1 (label alignment):**
- Look at `artifacts/gate1/gate1_results.csv`
- Count aligned events
- PASS if ≥ 7/10

**Gate 2 (RF signal):**
- Look at `artifacts/adaptive/b4_randomforest_test.json`
- Extract val F1 (from finetune) or test F1 (fallback)
- PASS if val F1 ≥ 0.60

**Gate C (cleaning matters):**
- Look at `artifacts/gate_c_results.json`
- Check: removal % ≥ 0.3% OR differential removal OR KS > 0.05
- PASS if any is true

**Gate 3 (SSL helps):**
- Look at `artifacts/adaptive/at_pre_summary.json` and `at_scratch_summary.json`
- Compare val F1 means
- PASS if at_pre > at_scratch by more than max(std_pre, std_scratch)

If PASS: recommend next gate/next subagent
If FAIL: recommend the pivot from `PILOT_PLAN.md §6-*`
If PENDING: recommend which subagent to invoke
