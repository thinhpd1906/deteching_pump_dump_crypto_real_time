#!/usr/bin/env bash
# =============================================================
# MASTER EXPERIMENT RUNNER
# Runs the complete 8-week pilot battery in dependency order.
# Assumes: Gate 1 already passed, data downloaded.
#
# Usage:
#   bash scripts/run_pilot.sh                 # full pilot
#   bash scripts/run_pilot.sh --seeds 3       # pilot with 3 seeds per model
#   SKIP_BASELINES=1 bash scripts/run_pilot.sh  # skip if already done
# =============================================================
set -e

CFG="config/config.yaml"
SEEDS=${SEEDS:-3}
CLEAN_PRIMARY="adaptive"

echo "================================================================"
echo " PND PILOT EXPERIMENT BATTERY"
echo "================================================================"
echo " Config  : $CFG"
echo " Seeds   : $SEEDS"
echo " Primary cleaning mode: $CLEAN_PRIMARY"
echo ""

# ---- STEP 1: Build datasets (all 3 cleaning modes) ---------------
echo "[STEP 1] Building windowed datasets (3 cleaning modes)..."
python -m src.data.build_dataset --config $CFG --clean all
echo ""

# ---- GATE C: Cleaning diagnostics --------------------------------
echo "[GATE C] Running cleaning diagnostics..."
python scripts/gate_c_cleaning.py
echo ""

# ---- STEP 2: Classical + microstructure baselines ----------------
if [ -z "$SKIP_BASELINES" ]; then
  echo "[STEP 2] Running baselines (GATE 2 check included)..."
  python -m src.models.baselines --config $CFG --clean $CLEAN_PRIMARY
  echo ""
fi

# ---- STEP 3: Unsupervised pretrain + B7/B10 ----------------------
echo "[STEP 3] Self-supervised pretraining (Contribution 2)..."
python -m src.training.pretrain --config $CFG --clean $CLEAN_PRIMARY
echo ""

echo "[STEP 3b] Unsupervised detectors (B7 LSTM-AE, B10 AT-unsup)..."
python -m src.training.eval_unsupervised --config $CFG --clean $CLEAN_PRIMARY
echo ""

# ---- STEP 4: Supervised deep models (multiple seeds) --------------
echo "[STEP 4] Supervised deep models ($SEEDS seeds each)..."
for SEED in $(seq 0 $((SEEDS-1))); do
  echo "  --- CNN-BiLSTM seed=$SEED ---"
  python -m src.training.finetune --config $CFG \
    --clean $CLEAN_PRIMARY --arch cnnbilstm --seed $SEED

  echo "  --- AT from scratch seed=$SEED ---"
  python -m src.training.finetune --config $CFG \
    --clean $CLEAN_PRIMARY --arch at --seed $SEED

  echo "  --- AT pretrained (OURS) seed=$SEED ---"
  python -m src.training.finetune --config $CFG \
    --clean $CLEAN_PRIMARY --arch at --pretrained --seed $SEED
done
echo ""

# ---- STEP 5: Label-efficiency study (M16) -------------------------
echo "[STEP 5] Label-efficiency study (money-shot figure)..."
python -m src.training.label_efficiency --config $CFG \
  --clean $CLEAN_PRIMARY --seeds $SEEDS
echo ""

# ---- STEP 6: Ablation -- all 3 cleaning modes ---------------------
echo "[STEP 6] Cleaning ablation (Table 2)..."
for MODE in none static adaptive; do
  if [ "$MODE" != "$CLEAN_PRIMARY" ]; then
    echo "  --- AT pretrained on '$MODE' cleaned data ---"
    # Pretrain on that mode's unlabeled pool
    python -m src.training.pretrain --config $CFG --clean $MODE
    # Fine-tune
    python -m src.training.finetune --config $CFG \
      --clean $MODE --arch at --pretrained
  fi
done
echo ""

# ---- STEP 7: Aggregate results -- both paper tables ---------------
echo "[STEP 7] Building Table 1 and Table 2..."
python -m src.training.evaluate --config $CFG --clean $CLEAN_PRIMARY
echo ""

echo "================================================================"
echo " PILOT COMPLETE. Results in artifacts/"
echo " Next: check gate outputs, then run scripts/plot_figures.py"
echo "================================================================"
