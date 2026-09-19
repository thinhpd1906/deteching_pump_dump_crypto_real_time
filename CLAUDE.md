# CLAUDE.md — Thesis Project Briefing

> **This file is loaded automatically at the start of every Claude Code session.**
> It's the "master briefing" that primes Claude with locked design decisions,
> so no context is lost between sessions.

## Project Identity

- **Thesis:** An End-to-End Real-time Streaming Pipeline for Pump-and-Dump Detection in Cryptocurrency Markets
- **Author:** Bryan (Phan Đức Thịnh), Master's student at HUST + Android dev at Viettel
- **Target venue:** IEEE ICBC 2027 (primary), KDD-ADS 2027 (stretch), ECML PKDD 2027 (alternate)
- **Timeline:** 7–9 months, starting now
- **Supervisor:** GCN/anomaly detection on behavioral graphs (ACIIDS 2023, RIVF 2021)

## Core Contribution (three claims, do not weaken)

1. **Adaptive streaming pipeline with data-quality ablation** (Table 2)
2. **Self-supervised pretraining overcomes label scarcity** (label-efficiency curves)
3. **End-to-end real-time deployment with measured latency**

## Locked Design Decisions (do not reopen)

| Item | Value |
|---|---|
| Data granularity | Per-trade (Binance aggTrades — not raw ticks; reviewers flag this) |
| Bar size | 5 seconds |
| Features | 10 microstructure: ret, hl_range, vwap_dev, taker_buy_ratio, log_vol, log_ntr, z_vol, z_ntr, z_ret, roll_sigma |
| Window | T=120 bars (10 min), stride 12 bars (1 min) |
| Positive label | window END ∈ [t_pump − 60 min, t_pump + 5 min] |
| Split | chronological by global pump-time quantiles 70/15/15 — NEVER shuffle |
| Scaler | fit on TRAIN only, saved to scaler.json, reused at serving |
| Model | Anomaly Transformer (d_model=64, 4 heads, 3 layers) |
| Pretraining | minimax association discrepancy + 25% masking, k=3 |
| Fine-tuning | BCE + pos_weight (auto), threshold TUNED ON VAL (never 0.5) |
| Evaluation | window-level F1 + event-level recall + bootstrap CI over SYMBOLS |
| **CRITICAL RULE** | **NO point-adjustment protocol (Kim et al. AAAI 2022)** |

## Four Go/No-Go Gates

- **Gate 1 (Week 1):** ≥7/10 pump events show price spike aligned with label → else timezone bug
- **Gate 2 (Week 3):** Random Forest val F1 ≥ 0.60 → else data problem, do NOT touch deep learning
- **Gate C (Week 3):** cleaning removes ≥0.3% OR differential removal OR KS>0.05 → else cleaning is cosmetic
- **Gate 3 (Week 7):** at_pre val F1 > at_scratch (outside CI) → else pivot to CNN-BiLSTM headline

## Key Data Sources (all free)

- **Labels:** github.com/SystemsLab-Sapienza/pump-and-dump-dataset (338 Binance events)
- **Historical trades:** data.binance.vision/data/spot/daily/aggTrades/{PAIR}/
- **Live stream:** wss://stream.binance.com:9443 (no API key)

## Subagents (one per pipeline stage)

Delegate to specialized subagents by name — each has its own context and tools:

- `@data-agent` — labels + Binance downloads + Gate 1
- `@cleaning-agent` — filters + Gate C
- `@feature-agent` — features.py + build_dataset.py (train↔serve parity)
- `@baseline-agent` — rule, RF, XGBoost, microstructure + Gate 2
- `@model-agent` — Anomaly Transformer, CNN-BiLSTM, LSTM-AE
- `@training-agent` — pretrain, finetune, label-efficiency + Gate 3
- `@evaluation-agent` — Tables 1 & 2, bootstrap CIs, figures
- `@pipeline-agent` — Kafka + Spark + inference service
- `@paper-agent` — LaTeX draft, novelty framing, related work
- `@reviewer-agent` — hostile IEEE reviewer simulation

## Code Style Rules (enforce always)

- No hardcoded constants — everything in `config/config.yaml`
- Feature code is SHARED between training and serving — never re-implement
- Always fit scaler on TRAIN split only
- Always tune threshold on VALIDATION, never on test
- Test set touched EXACTLY ONCE for final metrics
- Every experiment: 3–5 seeds, report mean ± std
- Bootstrap CIs resample over SYMBOLS (not windows)

## Common Pitfalls (already discovered — do not re-hit)

- pandas ≥2 timestamp resolution: always `index.as_unit("ns").asi8` before `.asi8`
- Binance CSV: 2025+ files use microseconds → normalize to ms
- Binance CSV: old files headerless, new have header → sniff row 1
- `is_buyer_maker` arrives as string "True"/"False", not bool
- HTTP 404 on Binance = pair not traded that day → skip silently
- Labels overlap across sources (La Morgia, ArdiaD, Fantazzini) → dedupe on `(pair, |Δt|<30min)`

## Do Not

- Do NOT change locked design decisions without user's explicit approval
- Do NOT use point-adjustment evaluation (headline error in this field)
- Do NOT compare our F1 to Anomaly Transformer paper's F1 (different protocols)
- Do NOT skip gates to move faster — gates exist to catch project-killing bugs early
- Do NOT create files outside `/config`, `/src`, `/scripts`, `/tests`, `/notebooks`
- Do NOT use bullet points when responding — use prose. User prefers prose over bullets when explaining.

## Language Preference

- User is Vietnamese; explanations OK in Vietnamese, code comments in English
- Technical discussions can be English-heavy — user is a professional developer
