# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

CLAUDE.md — Master Project Briefing

Auto-loaded at the start of every Claude Code session. Contains all locked decisions, real run results, and lessons learned. Do not reopen locked decisions without Bryan's explicit approval.

0. COMMANDS & CODE ARCHITECTURE

Commands (run from project root with the venv active; every entry point is a module taking `--config config/config.yaml`, and most take `--clean {none,static,adaptive}`, default adaptive):
```
pip install -r requirements.txt
python -m pytest tests/ -v                                   # architecture smoke tests only (no training)
python -m pytest tests/test_anomaly_transformer.py -v -k <name>   # single test
python -m src.data.download_labels                            # -> data/labels/pumps_binance.csv
python -m src.data.download_binance [--limit-events N]        # -> data/raw/<PAIR>/
python scripts/gate1_alignment.py [--limit-events N]          # Gate 1
python -m src.data.build_dataset --clean all                  # -> data/processed/<mode>/
python scripts/gate_c_cleaning.py                             # Gate C
python -m src.models.baselines --clean adaptive               # B1-B6, Gate 2
python -m src.training.pretrain [--resume --start-epoch N]    # SSL -> artifacts/<mode>/at_pretrained.pt
python -m src.training.eval_unsupervised                      # B7 LSTM-AE, B10 AT-unsup
python -m src.training.finetune --arch {at,cnnbilstm} [--pretrained] --seed S [--label-frac 0.1] [--deadline T]
python -m src.training.label_efficiency --seeds 3
python -m src.training.evaluate                               # Table 1, Table 2, Gate 3 check
python scripts/kaggle_queue.py --list | --restore <prev_output> [--budget-hours 11]
docker compose up -d                                          # Kafka + UI (localhost:8080)
python -m src.pipeline.kafka_producer / python_bars / spark_streaming / inference_service --model at_pre
```
`scripts/run_pilot.sh` and `scripts/run_all_experiments.sh` show the full dependency order. No linter is configured.

Data flow: aggTrades zips (data/raw/<PAIR>/) → `src/cleaning/filters.py` (clean_ticks, mode none/static/adaptive) → `src/data/features.py` (trades_to_bars → normalize_bars → bars_to_features → make_windows → label_windows) → `src/data/build_dataset.py` writes per-mode `{train,val,test}.npz` (keys X, y, end_ns, pair), `unlabeled_pretrain.npz`, `scaler.json`. Split boundaries come from global pump-time quantiles of the labels CSV.

Training/eval: `src/training/common.py` is the single home of the evaluation protocol (load_split applies scaler.json, tune_threshold on val, metric_table, event_metrics, symbol-level bootstrap_ci, full_eval, save_result). Every model script ends in `save_result` → `artifacts/<mode>/<tag>_test.json`; `evaluate.py` and `kaggle_queue.py` discover results purely by that filename. Tags are built by `finetune.run_tag`: `{arch}_{pre|scratch}[_lf{pct}][_s{seed}]` — keep this naming stable or the queue/tables break. Models live in `src/models/anomaly_transformer.py` (AnomalyTransformer for pretraining, ATClassifier head for fine-tuning) and `src/models/deep_baselines.py` (CNNBiLSTM, LSTMAE); classical baselines in `src/models/baselines.py`.

Kaggle: `kaggle-notebook/pnd_finetuning_kaggle.ipynb` drives `scripts/kaggle_queue.py`, which skips jobs whose result JSON exists; finetune writes `{tag}_last.pt` each epoch and exits with code 75 (EXIT_INCOMPLETE) near `--deadline` so the next session resumes. Downloaded Kaggle outputs land in `thesis_kaggle_output/` locally.

Serving: `src/pipeline/inference_service.py` imports `bars_to_features`/`normalize_bars` from `src/data/features.py` and loads the same `scaler.json` and checkpoint threshold — this import is the train↔serve parity guarantee (C3); never duplicate feature logic in the pipeline. `python_bars.py` is the no-Spark fallback for `spark_streaming.py`.

1. WHO I AM WORKING WITH
Name: Bryan (Phan Đức Thịnh) — Master's student at HUST + Android dev at Viettel (TV360)
Language: English for technical work, Vietnamese OK for casual explanation (Bryan often writes in Vietnamese; answer in Vietnamese when he does)
Style: Direct answers preferred. Complete working code over snippets. Explain in prose, not bullet lists. Be honest about weak results and academic risks.
TWO MACHINES, BOTH IN ACTIVE USE (since 2026-11-06). Bryan works with Claude Code on either one; both must understand the full project from this file.
Mac mini M4 Pro: project folder ~/pnd-thesis, venv `source .venv/bin/activate`, set `PYTORCH_ENABLE_MPS_FALLBACK=1`, long/unattended jobs (training, search, latency, later 24/7 automation) run here inside tmux session `thesis`.
Windows laptop: project folder C:\Users\phant\Desktop\pnd-thesis-agents\, venv .venv\Scripts\python.exe (PowerShell: .venv\Scripts\Activate.ps1). Used for coding, review, writing, Kaggle notebook work.
Sync rules between the two: git is the only channel. `git pull` at the start of every session, commit + push at the end. The Mac's unattended work goes to branch `mac/phase-a` (later `auto/<date>`); Bryan merges into `main`. Result JSONs and logs are committed so both machines see them; data and `*.pt` are not in git, so they move via the Kaggle datasets or scp, and `python scripts/make_data_manifest.py --verify` confirms both machines hold identical data. Claude's local memory does NOT sync between machines: anything a future session on either machine must know goes into this file (Section 20 for current state).
GPU: Apple MPS on the Mac (code needs task A1 to use it), Kaggle T4 x2 for heavy runs.
Commits: author thinhpd only. NEVER add a Co-Authored-By line or any Claude/AI attribution.
GitHub: github.com/thinhpd1906/deteching_pump_dump_crypto_real_time
Supervisor: GCN/anomaly detection on behavioral graphs (ACIIDS 2023, RIVF 2021)
2. THESIS IDENTITY

Title (EN): An End-to-End Real-time Streaming Pipeline for Pump-and-Dump Detection in Cryptocurrency Markets Title (VN): Xây dựng Pipeline Streaming Thời gian Thực để Phát hiện Pump-and-Dump trên Thị trường Tiền mã hóa

Target venue: IEEE ICBC 2027 (primary) → KDD Research Track 2027 → ECML PKDD 2027 (alternate) Timeline: 7–9 months total; currently in Week 6–7 of 8-week pilot. Thesis defense June 2027, needs an ACCEPTED paper before then.
Deadlines (checked 2026-11-06): ICBC 2027 submit 3 Dec 2026, decision 19 Feb 2027 (primary). ACIIDS 2027 hard deadline 14 Nov 2026 (alternative, not both). KDD 2027 cycle 2 ~Feb 2027 (fallback after a rejection). Same paper never submitted to two venues at once; backups are sequential. Method freeze for ICBC: 2026-11-23.
Two machines: Windows laptop (paths below) and Mac mini M4 Pro (~/pnd-thesis, `source .venv/bin/activate`). Mac setup + full Mac work plan (search mode, test ledger, tuning, latency, 24/7 automation): Section 19. gitignored data reaches the Mac via Kaggle datasets phancthnh/pnd-thesis-processed-data + phancthnh/pnd-thesis-checkpoint + scp of raw; verify with `python scripts/make_data_manifest.py --verify`.

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
Gate 3 — SSL Helps (at_pre > at_scratch outside CI) ⚠️ FAIL at 100% labels — 10% check PENDING
Full-label result (2026-10-12, Kaggle, 3 seeds, test window F1): at_pre 0.1791 ± 0.0089 vs at_scratch 0.1899 ± 0.0054 → pretraining did not help at full labels (scratch slightly ahead, gap ~1 std)
Both AT variants ≈ B5 LogReg test F1 0.1893 → deep sequence model does NOT yet beat the pooled classical baseline (see §12 Gate 2 note: this is the trigger to discuss revisiting the label window — Bryan's call)
Latest Kaggle output (thesis_kaggle_output/, 3 seeds, test F1): at_pre 0.1850 ± 0.0043, at_scratch 0.1929 ± 0.0043, cnnbilstm 0.1860 ± 0.0048; all reach event recall ≈1.0 (50/50); fa/symbol-day 6–10. Label-efficiency file only contains the 100% point so far.
Still needed before verdict: 10%-of-events label-efficiency runs, event recall / PR-AUC / CIs from the result JSONs, B7/B10
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
Label-efficiency sampled WINDOWS not EVENTS (fixed 2026-10-12): 10% of windows still covered 225/225 train events (~65 overlapping pos windows/event) → not real scarcity. subsample_labels now samples by event (10% = 22 events). Never revert.
Kaggle notebook Step 3 extracted zip to ./processed instead of data/processed (fixed 2026-10-12, now globs /kaggle/input). Notebook also lacked eval_unsupervised (B7/B10) — added as Step 5b.
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
    First single-session attempt needed >8h → reworked (2026-10-12) into a resumable multi-session queue: scripts/kaggle_queue.py runs 37 jobs in priority order (Gate 3 core first), skips any job whose {tag}_test.json exists, finetune saves {tag}_last.pt every epoch and exits 75 before the deadline. Chain sessions by attaching the previous notebook version's output as input (Step 5 --restore). Speed-ups: GPU-resident batches, fp16 AMP (train.amp), 2 parallel jobs on T4 x2, vectorised threshold tuning (verified identical to sklearn), label-efficiency 100% point reuses main fine-tune runs instead of retraining.
    Status 2026-11-06: 100%-label runs DONE (thesis_kaggle_output/: at_pre 0.1850±0.0043, at_scratch 0.1929±0.0043, cnnbilstm 0.1860±0.0048 test F1). 10%-label runs, B7/B10, 25/50% points NOT done.
[→] Week 6-7b (2026-11-06): moving to the Mac mini + Phase A of Section 19 (device support, val-only search mode + test ledger, Gate 3 at 10%, fair tuning until 2026-11-23 freeze, C3 latency).
[ ] 2026-11-23: method freeze → single test run → ICBC paper writing → submit 2026-12-03
[ ] Week 8: Tables + figures + draft → GO/PIVOT decision (now merged into the ICBC paper push)
17. SESSION START CHECKLIST
Read Section 20 (handoff state) first: it says where we stopped and what's open.
Run /status to see current state; check artifacts/ and thesis_kaggle_output/ for latest results.
Mac: `cd ~/pnd-thesis && source .venv/bin/activate`, work inside `tmux attach -t thesis`.
Ask: "What is my next step based on current progress?" if unsure.
Always run claude from project root so CLAUDE.md is auto-loaded.
Keep Section 20 current at the end of each session (what was done, what's open).
## 18. SELF-UPDATE RULE (IMPORTANT)

After every completed task, gate, or week milestone, Claude Code MUST:
1. Update Section 5 (gate status) with ✅ PASSED + real numbers
2. Update Section 16 (weekly plan) with ✅ for completed weeks
3. Update Section 6 (real run results) with new data
4. Update "Gate C — IN PROGRESS" to final verdict when done

Do this without being asked. A stale CLAUDE.md causes session confusion.

Do NOT modify .claude/settings.json — file is read-only, managed by Bryan only.

## 19. MAC MINI PLAN: one-command setup, Phase A (before ICBC), Phase B (24/7 automation)

Agreed with Bryan 2026-11-06. This section is the full instruction set for Claude Code running on the Mac mini M4 Pro. Work through the tasks in order, report each acceptance check, and stop at every "STOP: ask Bryan".

### 19.1 How the Mac gets started
On the Mac, Bryan runs once in Terminal:
```
curl -fsSL https://raw.githubusercontent.com/thinhpd1906/deteching_pump_dump_crypto_real_time/main/scripts/mac/bootstrap.sh -o /tmp/pnd-bootstrap.sh && bash /tmp/pnd-bootstrap.sh
```
`scripts/mac/bootstrap.sh` is idempotent. It installs Homebrew and the tools (uv, tmux, libomp for xgboost, openjdk@17 for pyspark, colima+docker for Kafka, Tailscale), clones to `~/pnd-thesis`, creates a Python 3.11 venv, downloads the gitignored data from Kaggle (`phancthnh/pnd-thesis-processed-data`, `phancthnh/pnd-thesis-checkpoint`), extracts `~/pnd-dl/pnd-raw.tgz` if present, verifies parity with `scripts/make_data_manifest.py --verify`, sets never-sleep/autorestart/no-auto-macOS-update/SSH, installs Claude Code, creates branch `mac/phase-a`, and starts Claude in tmux session `thesis` with a kickoff prompt pointing here. Log: `~/pnd-bootstrap.log`. Manual items it could not do: `~/pnd-bootstrap-todo.txt`.
Things only a human can do: type the sudo password, paste kaggle.json, log in to Claude and Tailscale, copy raw data from Windows (`scp C:\Users\phant\Desktop\pnd-raw.tgz <user>@<mac>:~/pnd-dl/`), decide FileVault (ON = after a power cut the Mac waits at the unlock screen until unlocked; test unlock-over-SSH after a reboot), buy a UPS. Remind Bryan of open items from `~/pnd-bootstrap-todo.txt` at the start of the session.
Daily access from Windows: `ssh <user>@<mac-tailscale-name>` then `tmux attach -t thesis`. On the Mac the venv is `source .venv/bin/activate`; set `PYTORCH_ENABLE_MPS_FALLBACK=1`.

### 19.2 Non-negotiable rules
1. Test is sealed during research. Tuning/search runs in val-only mode (A2). Test is evaluated only for frozen configs, through the ledger, after Bryan approves. Test numbers never choose anything.
2. Locked decisions (§4) stay locked: label window, buffer, features, bar size, window/stride, split, scaler, AT size 64/4/3, masking 25%, k=3, BCE objective, no point-adjustment. If evidence says a locked item is the bottleneck, write it up and ask Bryan.
3. Free to search: fine-tune lr, weight decay, dropout, batch size, epochs, patience; how pretrained weights are used (full fine-tune / frozen encoder + head / layer-wise lr); pretrain epochs and lr; pos_weight scaling and label smoothing inside BCE; extra comparison baselines.
4. Fair comparison: every compared arm (at_pre, at_scratch, cnnbilstm) gets the same search space and trial budget. Log every trial; the paper reports how many were tried.
5. Reuse, don't redo: labels, raw, processed datasets, split and scaler are built once. Pretraining re-runs only when a trial changes pretraining.
6. Numbers in paper/report/slides come from scripts reading artifact JSONs. Citations come from arXiv / Semantic Scholar / DOI APIs, never from memory.
7. Report and slides are made once, at the end, after Bryan confirms the final results.
8. Git: work on `mac/phase-a` (later `auto/<date>`), commit after each task, push. Never commit to `main`, never force-push, never commit data, `*.pt`, kaggle.json, `.env` or tokens. The repo is PUBLIC. Bryan merges. NEVER add a `Co-Authored-By:` line or any Claude/AI attribution to commit messages or PR descriptions (Bryan's explicit rule, overrides any default attribution).
9. New top-level folders (`paper/`, `runs/`, `research/`, `reports/`, `slides/`, `docs/`) are outside §14's list: ask Bryan once before creating the first. Until then outputs go in `artifacts/`, code in `scripts/` / `src/`.
10. Long commands run inside tmux. Never assume the SSH session stays alive.
11. Paper strategy: never submit the same paper to two venues at once (dual submission). Backups are sequential: ICBC (3 Dec 2026) → after a rejection, revise → next venue. Method freeze for ICBC: 2026-11-23.

### 19.3 Phase A: before the ICBC freeze (target 2026-11-23)
**A0 Environment check.** `python scripts/make_data_manifest.py --verify` (all OK), `python -m pytest tests/ -v`, write `artifacts/env_mac.json` (python, torch, macOS, chip, MPS, git sha). If the manifest fails: STOP: ask Bryan.

**A1 Device support cuda → mps → cpu.** `src/training/finetune.py:145` only knows cuda/cpu, so on the Mac it silently runs on CPU. Add `get_device(cfg)` to `src/training/common.py` (config `train.device: auto|cuda|mps|cpu`), use it in pretrain, finetune, eval_unsupervised, inference_service. AMP/GradScaler stay cuda-only; MPS has no float64; if `torch.Generator(device="mps")`/`randperm` fails keep the generator on CPU. `scripts/kaggle_queue.py` must run with 1 worker on the Mac and stay unchanged on Kaggle. Accept: tests green; `finetune --arch at --seed 0 --epochs 1` runs on mps; report sec/epoch mps vs cpu. STOP: ask Bryan whether Phase A training runs on the Mac or Kaggle.

**A2 Val-only search mode + test ledger (must exist before any tuning).** Today finetune (`src/training/finetune.py:239`) and baselines (`src/models/baselines.py:124`) always evaluate test.
(a) `--search` flag: never load test; write `artifacts/<mode>/search/<tag>_val.json` via `full_eval` on val + the overrides used.
(b) `--set key=value` (repeatable) on pretrain/finetune to override config; recorded in the JSON and folded into the tag/hash; reject keys from the locked list.
(c) `artifacts/test_ledger.jsonl`, append-only: every run that touches test appends {time, tag, git sha, config hash, overrides, checkpoint sha256}; refuse a repeated tag+config hash unless `--ledger-override "<reason>"`. Backfill existing result JSONs with `"backfilled": true`.
(d) Atomic writes (`*.tmp` then `os.replace`) for all JSON/checkpoints.
Accept: a test proving `--search` never opens test.npz; ledger refusal test; default behaviour unchanged.

**A3 Gate 3 at 10% labels (start right after A1).** If not already done on Kaggle: default config, normal mode (protocol runs, not tuning): `at --pretrained` and `at` at `--label-frac 0.1`, seeds 0,1,2; then cnnbilstm at 0.1; then eval_unsupervised (B7/B10). Use `scripts/kaggle_queue.py` (skips finished tags). Then label_efficiency and evaluate. Accept: Gate 3 verdict at 10% with mean ± std and paired symbol-bootstrap (bootstrap-ci skill); update §5/§6. STOP: report to Bryan. If at_pre doesn't win, apply §12 only after Bryan confirms.

**A4 Time-boxed fair tuning.** `scripts/search.py`, Optuna with SQLite at `artifacts/search/optuna_<arm>.db` (resumable). Config block:
```yaml
automation:
  search:
    arms: [at_pre, at_scratch, cnnbilstm]
    budget_trials_per_arm: 30     # equal per arm; Bryan may change
    stop_if_no_improve: 10
    confirm_top_k: 3              # top configs re-run with seeds 0,1,2, still val-only
    stop_at: "2026-11-23T23:59:00+07:00"
    label_frac: 1.0
    space: {}                     # only the free knobs in 19.2 rule 3
  target:                         # VAL only
    must_beat: b5_logreg
    margin: seed_std
```
Every trial = finetune `--search --set ...`. Pretrained checkpoints cached by hash of pretrain settings (`artifacts/adaptive/pretrain_cache/<hash>.pt`). Log `artifacts/search/trials.jsonl`, write `artifacts/search/leaderboard.md`. Accept: survives `kill -9` and resumes without losing finished trials. Then STOP: human checkpoint 1. Show the leaderboard and proposed frozen config per arm. Do not run test.

**A5 C3 latency (parallel with A3/A4, CPU).** `docker-compose.yml` uses `bitnami/kafka:3.7`; if it doesn't pull on arm64 under colima, switch to `apache/kafka` (uses `KAFKA_*` env names, not `KAFKA_CFG_*`). Pipeline kafka_producer → python_bars (or spark_streaming) → inference_service --model at_pre. Add `scripts/measure_latency.py`: ≥2 h live, raw samples `artifacts/latency/samples.csv`, `artifacts/latency/summary.json` (p50/p95/p99, count, symbols, device, git sha). Report separately (a) bar-close → alert and (b) trade time → alert (includes the ≤5 s bar wait). Record clock offset vs NTP (`sntp time.apple.com`) at start and end. Run the verify-scaler-parity skill. Accept: summary exists, parity passes, target p95 (a) < 5 s.

**A6 Freeze + single test run (only after checkpoint 1 approval).** Git tag `freeze-icbc-v1`, frozen configs in `config/frozen/icbc_v1.yaml`. Each frozen arm in normal mode, seeds 0,1,2, at 100% and 10% labels, tags suffixed `_final`, all through the ledger. evaluate.py + paired bootstrap vs every baseline. STOP: human checkpoint 2, Bryan confirms the results are final.

**A7 Paper support (after checkpoint 2; ask for `paper/` first).** `scripts/make_tables.py` (LaTeX tables from JSON), extend `scripts/plot_figures.py`, `scripts/fetch_bib.py` (BibTeX by arXiv id / DOI), `scripts/check_numbers.py` (fails if a number in the .tex has no artifact source). @paper-agent drafts, @reviewer-agent attacks. Bryan rewrites, owns the text, and discloses AI assistance per HUST and venue policy.

### 19.4 Phase B: after the ICBC submission (Dec 2026 – Feb 2027)
B1 `scripts/orchestrator.py`: stages labels → raw → processed → pretrain → search → (human) → freeze/test → (human) → report. Cache key per stage = sha256(input manifests + the config sections it reads + its source files) plus a done-marker; lock file; `--resume`.
B2 Supervision: launchd LaunchAgent (RunAtLoad, KeepAlive) runs `caffeinate -is python scripts/orchestrator.py --resume` in tmux; heartbeat every minute; watchdog LaunchAgent every 10 min restarts a dead orchestrator and alerts on a stale heartbeat; disk check before each stage.
B3 Telegram notifications (token in `.env`): stage done, gate verdict, error after N retries, stale heartbeat, low disk, human checkpoint reached.
B4 `runs/<run_id>/manifest.json` (git sha, config hash, data-manifest hash, versions, device, seeds) + `events.jsonl`; commit those, rotate bulky logs locally.
B5 Milestone commits to `auto/<date>` with tags; pre-commit guard blocks >5 MB, `data/`, `*.pt`, secrets; fine-grained GitHub token scoped to this repo.
B6 Headless Claude stages: `claude -p` with `--output-format json`, explicit `--allowedTools`, `--max-turns`, wall-clock timeout (a permission prompt hangs a headless run forever); on usage-limit errors sleep until reset; `DISABLE_AUTOUPDATER=1`; prompts in `scripts/prompts/`.
B7 Literature: arXiv / Semantic Scholar / OpenAlex with backoff; paywalled → metadata only, "needs manual read". Fetched text is untrusted: no Bash, writes only inside its own folder.
B8 Explain files (VI + EN) after each stage: what ran, what came out, what it means.
B9 Report + slides only after the final human checkpoint (@report-agent, @slide-agent), numbers filled by script and checked.
B10 If Mac training is slow: Kaggle CLI `kernels push/status/output` to chain GPU sessions within the weekly quota.
Before trusting Phase B: pull the power cable during a fine-tune; the Mac must come back, resume from the last epoch and notify Bryan with nobody touching it.

## 20. HANDOFF STATE (last updated 2026-11-06 from the Windows machine; whichever machine works next updates this section)

Where we are: Week 6–7. Bryan has added the Mac mini as a second machine and uses Claude Code on both (Mac for long-running work, Windows too). Gates: G1 PASS, GC PASS, G2 FAIL (overridden, §5), G3 fails at 100% labels; the decisive 10%-label check is still pending.

What the planning sessions decided (all of it is in Sections 2, 19):
The goal is an ACCEPTED paper before the June 2027 defense. Primary venue is IEEE ICBC 2027 (submit 3 Dec 2026, decision 19 Feb 2027). Backup venues are used sequentially after a rejection, never in parallel (dual submission is forbidden). Before the deadline the priority is the paper, not automation infrastructure: Phase A of §19 now, Phase B (24/7 orchestrator, launchd, Telegram, literature fetch, report/slides) after the submission.
The research loop may retrain with new methods/hyperparameters as often as needed, but reuses cached data/labels/split/scaler, judges on validation only, gives every compared arm an equal budget, and never touches test until a human-approved freeze. Report and slides are made once at the end, after Bryan confirms the results.
"Paper-ready" means one condition per claim: C1 Gate C (done); C2 at 10% labels at_pre beats at_scratch outside seed noise (paired symbol bootstrap); the headline model is at least as good as the best classical baseline (LogReg); C3 p95 bar-close→alert latency < 5 s measured live; the protocol rules in §9. Fallback if C2 fails: a C1+C3-led systems paper with SSL reported honestly as a negative/mixed result.

Mac setup status (2026-11-06, Mac session): bootstrap finished. A0 done (artifacts/env_mac.json: M4 Pro 24 GB, Python 3.11.18, torch 2.15.0, MPS OK; manifest OK for labels/processed/checkpoints, raw not yet copied; tests green). A1 done (commit 29ba278): get_device() in common.py, config train.device: auto (cuda > mps > cpu); MPS 68 s/epoch vs CPU 316 s/epoch for full-label AT fine-tune. Waiting on Bryan: run Phase A training on the Mac or on Kaggle? Note: thesis_kaggle_output/*_test.txt are the 100%-label result JSONs (renamed .txt); copy them to artifacts/adaptive/<tag>_test.json so kaggle_queue skips them.
Raw data (`data/raw`, 154 MB) is NOT on Kaggle or GitHub. It exists as `C:\Users\phant\Desktop\pnd-raw.tgz` on the Windows laptop; copy it to `~/pnd-dl/` on the Mac (scp over Tailscale, USB, or Drive) so both machines have it. Training does not need it; dataset rebuilds and Gate 1 do.
Kaggle auth on Windows uses the new OAuth style (~/.kaggle/credentials.json + access_token), not kaggle.json.

Open items for Bryan:
1. Security on the Mac: 2FA on the account used for Tailscale; SSH key-only (password login still ON after bootstrap); Remote Login limited to his user; firewall ON; check devices in the Tailscale admin console. If the Mac is company-managed (Viettel), ask IT before running the Tailscale VPN.
2. Start the 10%-label runs (Kaggle or Mac, after task A1). This is the most important missing result.
3. Ask the supervisor: ICBC (3 Dec) vs ACIIDS (hard deadline 14 Nov), not both; does HUST require acceptance or only submission before the defense; is a C1+C3-led paper acceptable if C2 fails; keep the 60-min lead-in label window locked for this submission?
4. Approve the Optuna "free to search" list and the trial budget (default 30 per arm) before task A4.
5. Approve new top-level folders (paper/, runs/, research/, reports/, slides/, docs/) when first needed.
6. The GitHub repo is public: consider making it private before submission (double-blind review).

Known code facts to remember: finetune.py and baselines.py evaluate test on every run today (must get the `--search` val-only mode, task A2, before any tuning). The docstring at the top of src/training/common.py says the bootstrap resamples over "EVENTS" but the code (correctly) resamples over SYMBOLS; fix the docstring. The data manifest (config/data_manifest.json) was generated on Windows; Kaggle copies of processed data and at_pretrained.pt were verified byte-identical to it.