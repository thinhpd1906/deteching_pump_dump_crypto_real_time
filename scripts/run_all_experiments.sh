#!/usr/bin/env bash
# Full experiment battery -> both paper tables. Run after Phase 1-3 (data).
set -e
CFG=config/config.yaml

for MODE in none static adaptive; do
  echo "=============== cleaning mode: $MODE ==============="
  python -m src.data.build_dataset  --config $CFG --clean $MODE
  python -m src.models.baselines    --config $CFG --clean $MODE
  python -m src.training.pretrain   --config $CFG --clean $MODE
  python -m src.training.eval_unsupervised --config $CFG --clean $MODE
  python -m src.training.finetune   --config $CFG --clean $MODE --arch cnnbilstm
  python -m src.training.finetune   --config $CFG --clean $MODE --arch at
  python -m src.training.finetune   --config $CFG --clean $MODE --arch at --pretrained
done

python -m src.training.evaluate --config $CFG --clean adaptive
