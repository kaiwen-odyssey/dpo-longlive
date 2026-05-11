#!/bin/bash
# DPO hyperparameter ablation on the MQ reward dimension.
# Sweep lr (3 values) at fixed beta=5000, batch_size=1.
set -e
source .venv/bin/activate
export PYTHONPATH=src:$PYTHONPATH

PAIRS=${PAIRS:-data/pairs/index.jsonl}
STEPS=${STEPS:-30}
WANDB_MODE=${WANDB_MODE:-offline}

for lr in 1e-7 1e-6 1e-5; do
  for beta in 1000 5000 20000; do
    NAME=ablation_mq_lr${lr}_beta${beta}
    OUT=runs/${NAME}
    if [ -d "$OUT" ]; then
      echo "[skip] $OUT exists"
      continue
    fi
    echo "[run] $NAME"
    python -m dpo_longlive.dpo_trainer \
      --pairs_index "$PAIRS" --reward_dim MQ \
      --out_dir "$OUT" --max_steps "$STEPS" \
      --beta "$beta" --lr "$lr" --save_every 100000 \
      --wandb_project dpo-longlive --wandb_name "$NAME" --wandb_mode "$WANDB_MODE" \
      2>&1 | tee logs/${NAME}.log
  done
done

echo "[done] ablation"
