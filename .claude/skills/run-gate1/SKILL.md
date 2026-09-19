---
name: run-gate1
description: Runs Gate 1 (label alignment verification). Downloads Binance data for first 10 pump events if not already cached, generates alignment plots ±2h around each pump, and prints PASS/FAIL verdict based on whether ≥7/10 events show visible price spike at/after the labeled timestamp.
---

# Skill: run-gate1

Complete Gate 1 workflow — the most important check in Week 1.

## When to use

- At the start of the project, before any modeling work
- After any change to label parsing (e.g., new label sources, timezone fixes)
- When user asks "is Gate 1 passing?" or "check alignment"

## Steps

1. Verify `data/labels/pumps_binance.csv` exists
   - If not: run `python -m src.data.download_labels` first
2. Check if data for first 10 events is already downloaded
   - Look under `data/raw/{PAIR}/` for parquet files
3. If not cached: `python -m src.data.download_binance --limit-events 10`
4. Run `python scripts/gate1_alignment.py`
5. Inspect `artifacts/gate1/*.png` — count how many show clear spike alignment
6. Report to user with clear verdict

## PASS criteria

- ≥ 7/10 events show visible price spike starting at/after the red label line
- Timestamps consistent (no random noise)

## FAIL response

If < 7/10 aligned:

1. Report the failure clearly
2. Suggest timezone shift experiments: ±1h, ±2h, ±4h, ±8h, ±12h
3. Do NOT recommend proceeding with modeling
4. Wait for user to fix labels before any handoff

## Output format

```
=== GATE 1 REPORT ===
Labels loaded:  338 events
Data available: 10/10 events
Alignment:      8/10 aligned (spike at/after label)

Details:
  [01] LRCBTC 2018-01-03 15:59 UTC  ALIGNED (spike +12.3%)
  [02] SEL BTC 2018-01-04 16:03 UTC  ALIGNED (spike +8.1%)
  ...
  [09] XYZBTC 2018-XX-XX HH:MM UTC   WEAK (+1.2%)
  [10] ABCBTC 2018-XX-XX HH:MM UTC   NO_DATA

VERDICT: PASS (>=7/10 aligned)
NEXT:    proceed to Phase 2 (full download)
```
