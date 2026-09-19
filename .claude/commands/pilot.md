---
description: Execute the full 8-week pilot experiment battery, delegating each phase to the appropriate subagent
---

Run the complete pilot battery in dependency order, delegating each phase to the right subagent:

**Phase 1 — Data (delegate to @data-agent):**
- Download labels
- Download 10-event sample
- Run Gate 1 alignment
- If Gate 1 PASS → full download
- If Gate 1 FAIL → STOP, report to user

**Phase 2 — Cleaning + Datasets (delegate to @cleaning-agent → @feature-agent):**
- Verify cleaning modes work
- Build datasets for all 3 modes
- Run Gate C diagnostics

**Phase 3 — Baselines (delegate to @baseline-agent):**
- Run B1-B6
- Report Gate 2 verdict
- If Gate 2 FAIL → STOP

**Phase 4 — Deep models (delegate to @model-agent → @training-agent):**
- Verify AT unit tests pass
- Run pretrain
- Run unsupervised eval (B7, B10)
- Run supervised fine-tune with 3 seeds each:
  - CNN-BiLSTM
  - AT from scratch
  - AT pretrained (OURS)

**Phase 5 — Label efficiency (delegate to @training-agent):**
- Run label-efficiency study with 3 seeds
- Report Gate 3 verdict

**Phase 6 — Cleaning ablation (delegate to @training-agent):**
- Pretrain + finetune OURS on `none` and `static` modes

**Phase 7 — Aggregation (delegate to @evaluation-agent):**
- Build Table 1 + Table 2
- Compute bootstrap CIs
- Generate figures

**Phase 8 — First draft (optional, delegate to @paper-agent):**
- Draft the 4-page pilot writeup
- Send to @reviewer-agent for hostile review

Stop at any gate failure and report to user. Do not skip gates for speed.

Between phases, print a clear banner:

```
============================================
 PHASE N COMPLETE. Moving to Phase N+1.
 Delegating to @<agent-name>
============================================
```
