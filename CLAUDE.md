CLAUDE.md — Master Project Briefing

Auto-loaded at the start of every Claude Code session. Contains all locked decisions, real run results, and lessons learned. Do not reopen locked decisions without Bryan's explicit approval.

1. WHO I AM WORKING WITH
Name: Bryan (Phan Đức Thịnh) — Master's student at HUST + Android dev at Viettel (TV360)
Language: English for technical work, Vietnamese OK for casual explanation
Style: Direct answers preferred. Complete working code over snippets.
Machine: Windows, VS Code + Claude Code CLI
Project folder: C:\Users\phant\Desktop\pnd-thesis-agents\
Virtual env: .venv\Scripts\python.exe — always activate before running Python
GitHub: github.com/thinhpd1906/detecting_pump_dump_crypto_real_time
Supervisor: GCN/anomaly detection on behavioral graphs (ACIIDS 2023, RIVF 2021)
2. THESIS IDENTITY

Title (EN): An End-to-End Real-time Streaming Pipeline for Pump-and-Dump Detection in Cryptocurrency Markets Title (VN): Xây dựng Pipeline Streaming Thời gian Thực để Phát hiện Pump-and-Dump trên Thị trường Tiền mã hóa

Target venue: IEEE ICBC 2027 (primary) → KDD Research Track 2027 → ECML PKDD 2027 (alternate) Timeline: 7–9 months total; currently in Week 2 of 8-week pilot

3. THREE CONTRIBUTIONS (do not weaken)

C1 — Adaptive streaming pipeline + data-quality ablation First P&D paper to ingest raw per-trade aggTrades through Kafka→Spark with volatility-adaptive cleaning, and quantify how cleaning mode affects detection (Table 2 interaction matrix).

C2 — Self-supervised pretraining overcomes label scarcity Only ~338 labeled events vs billions of normal trades. Pretrain Anomaly Transformer on unlabeled pool → fine-tune on labels. Evidence: label-efficiency curves at 10/25/50/100% showing pretrained curve dominates at-scratch at low fractions.

C3 — End-to-end real-time deployment with measured latency Live: Binance WebSocket → Kafka → Spark → inference → alerts. Feature code SHARED between training and serving (zero skew). Paper reports p50/p95 latency.

Novelty sentence (memorize this): "Unlike prior work that operates on pre-aggregated OHLCV data [Kamps, Chadalapaka] or requires social-media signals [Bolz, Perseus], we present the first end-to-end streaming system that ingests raw per-trade data, quantifies the impact of data cleaning on detection, and leverages self-supervised pre-training [SimMTM-style] to overcome label scarcity — evaluated under a rigorous, point-adjustment-free protocol [Kim 2022]."

4. LOCKED DESIGN DECISIONS (never reopen without Bryan's explicit approval)
Parameter	Value	Rationale
Data source	Binance aggTrades	Say "per-trade granularity" NOT "raw ticks" — reviewers flag it
Bar size	5 seconds	Balance between granularity and noise
Features	10: ret, hl_range, vwap_dev, taker_buy_ratio, log_vol, log_ntr, z_vol, z_ntr, z_ret, roll_sigma	Grounded in microstructure literature
Window	T=120 bars (10 min), stride=12 bars (1 min)	Captures full pump lifecycle
Positive label	window END ∈ [t_pump − 60 min, t_pump + 5 min]	Fantazzini & Xiao 2023 convention
Buffer	±3h around pumps excluded from negatives	Avoids ambiguous near-pump windows
Split	Chronological by global pump-time quantiles 70/15/15	NEVER shuffle — most common eval bug in the field
Scaler	Fit on TRAIN only → scaler.json	Serving path reloads this same file
Model	Anomaly Transformer (d_model=64, 4 heads, 3 layers)	Xu et al. ICLR 2022
Pretraining	Minimax association discrepancy + 25% masking, k=3	Self-supervised on unlabeled pool
Fine-tuning	BCE + pos_weight=auto, threshold tuned on VAL only	NEVER use 0.5, NEVER tune on test
Evaluation	Window F1 + event recall + bootstrap CI over SYMBOLS	NO point-adjustment (Kim et al. AAAI 2022)
5. FOUR GATES — CURRENT STATUS
Gate 1 — Label Alignment ✅ PASSED (Week 1, 10m 58s)
Result: 10/10 events aligned (perfect score, better than 7/10 minimum)
8 events spiked in exact labeled minute; ICNBTC spiked 1 minute later
Spike sizes: +13.7% (BNTBTC) to +142.2% (DGDBTC)
No timezone bug — labels are trustworthy
Agent fixed 2 bugs during this run:
Wrong filter (by pair not date) → was producing bogus 1/45 result
Windows emoji encoding crash in results CSV → fixed to ASCII
Artifacts: artifacts/gate1/gate1_results.csv + 10 plots
Gate 2 — Signal Exists (RF val F1 ≥ 0.60) ⚠️ FAIL (0.1740) — Bryan overrode, proceeding to deep models
B1-B6 val F1: B1 rule=0.049, B2 rushorder=0.045, B3 microstructure=0.127, B4 RF=0.174, B5 logreg=0.170, B6 xgboost=0.163 (expected ordering, no leakage/F1=0/suspicious-perfect scores)
Diagnostics run (per pivot plan below): Gate 1 re-verified clean; during-only relabel [t_pump, t_pump+5min] → RF val F1=0.849 (pump-active signal is strongly separable); feature dist pos vs neg shows only weak/moderate effect sizes in the −60min lead-in (z_ntr/roll_sigma/z_vol ~0.7–0.8σ, z_ret 0.35σ; max(z_vol) pos vs neg essentially identical: 4.995 vs 4.989)
Root cause: ~92% of positive-labeled windows are anticipatory (−60min lead-in) — hand-crafted static window-stats (mean/std/max/min/last) cannot see temporal build-up structure by construction
Bryan's decision (2026-09-22): proceed to deep models with the ORIGINAL locked label intact. Framing: this diagnostic is evidence FOR why a sequence model (Transformer) is needed — not a data bug to "fix", and not weakening the label to force a pass.
Artifacts: artifacts/adaptive/{b1_rule,b2_rushorder,b3_microstructure,b4_randomforest,b5_logreg,b6_xgboost}_test.json
Gate C — Cleaning Has Measurable Effect ✅ PASSED
Verdict: PASS — triggered by differential_removal_adaptive (1 of 3 criteria fired)
Removal rate: static 0.2509%, adaptive 0.0596% — neither crosses the 0.3% bar
Differential removal: static pump/normal ratio 1.53x (removes MORE near pumps — wrong direction, does not pass); adaptive ratio 0.037x (removes almost entirely from normal periods — PASSES, well under 0.5x)
KS test (none vs adaptive): z_ret stat=0.00162 p≈1.0, z_vol stat=0.00554 p=0.43 — neither exceeds 0.05
ANOMALY resolved: adaptive's k0=6.0×rolling-tick-σ threshold exceeds static's flat 2% for 62/84 pairs on average (thin/illiquid 2018-era altcoins) — verified pair-by-pair across all 84 pairs, confirmed as a genuine calibration finding, not a bug. Adaptive removes less overall but removes the RIGHT trades (preserves pump-adjacent signal) — a stronger result for C1 than raw removal count would have been.
Recommendation for paper: report as intentional trade-off; consider a k0 sensitivity sweep {3,4,5,6} as future-work for a second independent pass criterion (not applied — needs approval)
Artifacts: artifacts/gate_c_results.json
Gate 3 — SSL Helps (at_pre > at_scratch outside CI) ⏳ PENDING (pretrain done, fine-tune next)
Pretrain sanity check (2026-09-25): unsupervised anomaly score already separates pumps — positive=7.65 vs negative=0.022 (347x separation) with ZERO label supervision. Strong leading indicator for Gate 3.
Most important: check at 10% label fraction (SSL pays under scarcity)
If FAIL at 100% but PASS at 10% → Contribution 2 still alive
6. REAL RUN RESULTS (actual numbers, not estimates)
Part A — Data Download ✅ COMPLETE
Pair-days downloaded: 2,054
Unique pairs with data: 84/85
Genuinely unavailable (real 404s, delisted): 36 pair-days
Total disk: 154 MB in data/raw/
SSL errors in first pass were transient — retry recovered ~670 pair-days
Part B — Dataset Building ✅ COMPLETE (35m 53s)
All 18 output files present (3 modes × 6 files each)
No NaN in any split, correct shapes (120, 10), scaler has 10 features
No zero-positive splits in any mode
Bugs fixed by agent:
Chunking by consecutive calendar days (ARNBTC had 1.1M-window OOM → 8 chunks, 4,211 windows)
Two-pass label-then-rebuild for full dataset memory (2.75M windows / 12.3GB)
Static removal (0.2509%) > Adaptive (0.0596%) — flagged for Gate C review
Gate C — ✅ COMPLETE — PASS
Triggered by differential_removal_adaptive: static pump/normal removal ratio 1.53x (bad), adaptive 0.037x (good, <0.5x threshold)
Removal rate criterion (≥0.3%) not met by either mode; KS criterion (>0.05) not met (z_ret 0.00162, z_vol 0.00554)
Static > adaptive removal-rate inversion confirmed genuine (not a bug): adaptive threshold exceeds static's flat 2% for 62/84 pairs on avg, due to high tick-level volatility in thin altcoin pairs
Artifacts: artifacts/gate_c_results.json
Part D — Classical Baselines B1-B6 ✅ COMPLETE — Gate 2 FAIL, overridden by Bryan
RF (B4, Gate 2 baseline) val F1=0.1740, test F1=0.1473, event_recall=50/50, but 5.2 false alarms/symbol-day
Best test F1: B5 LogReg=0.1893; best precision: B6 XGBoost=0.214; B3 microstructure trio weakest of the "real" ML family (below B4/B5/B6) — no early sign of microstructure-trio-matches-DL risk for C2
During-only relabel diagnostic → RF val F1=0.849 → signal is real and strong once pump is active, deficit is concentrated in the −60min anticipatory portion of the label (~92% of positives)
Decision: proceed to @model-agent with original label intact (see Section 5 Gate 2 entry for full reasoning)
Part E — Self-Supervised Pretraining ✅ COMPLETE (Week 5, 2026-09-25)
5 epochs on adaptive dataset, 146,240 unlabeled windows, CPU
Loss curve: 0.276 → 0.236 → 0.222 → 0.213 → 0.207 (monotonic, converged)
Sanity check: mean anomaly score positive (pump) windows=7.65 vs negative=0.022 — 347x separation, unsupervised
Saved: artifacts/adaptive/at_pretrained.pt
Note: first attempt (pretrain_full.log) crashed silently after epoch 3 with no checkpoint saved (old code only saved once at the very end) — added --resume/--start-epoch flags plus per-epoch checkpointing to src/training/pretrain.py before rerunning; rerun (pretrain_resume.log) completed cleanly
7. DATA SOURCES (all free, no API key)
Source	URL	Notes
Labels	github.com/SystemsLab-Sapienza/pump-and-dump-dataset/master/pump_telegram.csv	338 Binance events, 85 symbols, 2018-2021
Historical	data.binance.vision/data/spot/daily/aggTrades/{PAIR}/{PAIR}-aggTrades-YYYY-MM-DD.zip	Headerless old files, 2025+ use µs
Live stream	wss://stream.binance.com:9443	No API key needed
8. BUGS ALREADY FIXED (do not re-hit)
pandas ≥2 timestamp: use index.as_unit("ns").asi8 NOT .asi8 directly
Binance CSV headers: old files headerless, new have header → sniff row 1
is_buyer_maker: arrives as string "True"/"False" not bool → cast explicitly
HTTP 404 on Binance: pair not traded that day → skip silently, log to unavailable list
Windows emoji encoding: gate1_alignment.py → crashed Windows console → fixed to ASCII
Gate1 filter bug: was filtering by pair not specific labeled dates → bogus 1/45 result → fixed
OOM on large pairs: ARNBTC 1.1M windows → chunk by consecutive calendar days
Full dataset memory: 2.75M windows / 12.3GB → two-pass label-then-rebuild
Labels overlap across sources: dedupe on (pair, |Δt|<30min) when merging La Morgia + ArdiaD + Fantazzini
9. EVALUATION PROTOCOL (state this explicitly in the paper)
Chronological split by global pump-time quantiles — NEVER shuffle
Threshold tuned on VALIDATION F1 (grid search) — never 0.5, never test
NO point-adjustment (Kim et al. AAAI 2022 arXiv:2109.05257) — cite explicitly
Test set touched EXACTLY ONCE for final metrics
Bootstrap CIs resampled over SYMBOLS not windows
Every headline number = mean ± std over ≥3 seeds
Event-level recall reported alongside window-level F1
10. COMPARISON TABLE DESIGN

Table 1 — Method comparison (on adaptive cleaning): B1 Rule-based | B2 Rush-order | B3 OFI/VPIN/Amihud | B4 Random Forest | B5 LogReg | B6 XGBoost | B7 LSTM-AE | B8 CNN-BiLSTM | B9 AT-scratch | B10 AT-unsup | D OURS: AT-pre+finetune

Table 2 — Cleaning ablation (OURS model): none | static | adaptive — same metric columns

11. KEY PAPERS
La Morgia et al. ICCCN 2020 + ACM TOIT 2022 — dataset source, RF baseline, paper structure template
Chadalapaka et al. arXiv:2205.04646 (2022) — closest DL baseline (CNN+BiLSTM)
Xu et al. ICLR 2022 — Anomaly Transformer architecture
Kim et al. AAAI 2022 arXiv:2109.05257 — anti-point-adjustment (CRITICAL)
SimMTM NeurIPS 2023 + TS2Vec AAAI 2022 — SSL paradigm legitimacy
Perseus arXiv:2503.01686 (2025) — complementary OSN-side work, not competing
Fantazzini & Xiao MDPI 2023 — 60-min labeling convention
12. PIVOT OPTIONS (pre-decided — use these if gates fail)

If Gate 2 FAILS (RF F1 < 0.60): Try during-only labels [t, t+5min]. Check feature distributions pos vs neg. Never fix with a bigger model.
ACTUAL OUTCOME (2026-09-22): Gate 2 FAILed (RF val F1=0.174). Ran both diagnostics above — during-only relabel hit F1=0.849 (signal exists, strongly separable), feature-dist check showed the −60min lead-in is genuinely weak for STATIC window-stats specifically (not for the model class in general). Bryan's call: this is NOT "fixing a data problem with a bigger model" — it's recognizing that the classical baseline family (mean/std/max/min/last summaries) structurally cannot see temporal build-up patterns, which is exactly the premise for trying a sequence model next. Proceeded to @model-agent with the original locked label unchanged. If deep models ALSO fail to beat this pooled baseline meaningfully, that would be the stronger signal to revisit the label window definition.

If Gate C FAILS (cleaning cosmetic): Option 1: Strengthen intervention (winsorize, wash-trade collapse, gap imputation). Option 2: Demote C1, report honest negative finding, lead with C2+C3.

If Gate 3 FAILS (SSL doesn't help): Check 10% label fraction first. If also tied → CNN-BiLSTM becomes headline, report SSL failure as honest negative result.

13. SUBAGENTS — WHO DOES WHAT
Agent	Owns	Never touches
@data-agent	Labels, Binance download, Gate 1	Models, features
@cleaning-agent	Filters, Gate C	Features, models
@feature-agent	features.py, build_dataset.py, train↔serve parity	Models, training
@baseline-agent	B1-B6, Gate 2	Deep models
@model-agent	AT, CNN-BiLSTM, LSTM-AE architectures	Training loops
@training-agent	Pretrain, finetune, label-efficiency, Gate 3	Architecture code
@evaluation-agent	Table 1, Table 2, bootstrap CIs, figures	Training
@pipeline-agent	Kafka, Spark, inference, latency	Models
@paper-agent	LaTeX draft, novelty framing	Code
@reviewer-agent	Hostile IEEE reviewer simulation	Everything (read-only)
14. CODE RULES (enforce always)
No hardcoded constants — everything in config/config.yaml
Feature code SHARED between training AND serving (src/data/features.py) — NEVER re-implement online
Scaler fit on TRAIN only — NEVER on val or test
Threshold tuned on VAL only — NEVER on test
Test set touched EXACTLY ONCE
NO point-adjustment evaluation
NEVER shuffle the chronological split
Bootstrap CIs over SYMBOLS not windows
Every training result needs ≥3 seeds, report mean ± std
Do NOT create files outside /config, /src, /scripts, /tests, /notebooks
15. DO NOT
Do NOT reopen locked design decisions without Bryan's explicit approval
Do NOT use point-adjustment evaluation
Do NOT compare our F1 to Anomaly Transformer paper's F1 (different protocols)
Do NOT skip gates to move faster
Do NOT say "raw ticks" — say "per-trade granularity"
Do NOT use bullet points when explaining to user — use prose
16. CURRENT WEEKLY PLAN
[✓] Week 1: Setup + labels + Gate 1 PASS (10/10)
[✓] Week 2: Full download ✓ + datasets ✓ + Gate C PASS ✓
[✓] Week 3: Baselines B1-B6 ✓ + Gate 2 FAIL (0.174) — overridden by Bryan, proceeding to deep models (see Section 12)
[✓] Week 4: Microstructure baselines (OFI/VPIN/Amihud) — done as B3 during Week 3's baseline run
[✓] Week 5: Model implementation + pretrain — pretrain complete, 347x unsupervised separation (see Section 6 Part E)
[→] Week 6-7: Fine-tuning on Kaggle GPU (at_pre, at_scratch, cnnbilstm × 3 seeds each) + unsupervised eval (B7, B10) + label-efficiency + Gate 3
[ ] Week 8: Tables + figures + 4-page draft → GO/PIVOT decision
17. SESSION START CHECKLIST
Run /status to see current state
Check artifacts/ for latest results
Ask: "What is my next step based on current progress?" if unsure
Always activate venv before running Python: .venv\Scripts\Activate.ps1
Always run claude from project root so CLAUDE.md is auto-loaded
## 18. SELF-UPDATE RULE (IMPORTANT)

After every completed task, gate, or week milestone, Claude Code MUST:
1. Update Section 5 (gate status) with ✅ PASSED + real numbers
2. Update Section 16 (weekly plan) with ✅ for completed weeks
3. Update Section 6 (real run results) with new data
4. Update "Gate C — IN PROGRESS" to final verdict when done

Do this without being asked. A stale CLAUDE.md causes session confusion.

Do NOT modify .claude/settings.json — file is read-only, managed by Bryan only.