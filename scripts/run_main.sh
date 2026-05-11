#!/bin/bash
# Main runs: 3 DPO + 3 Re-DMD on (VQ, MQ, TA) at the picked hyperparameters.
set -e
source .venv/bin/activate
export PYTHONPATH=src:$PYTHONPATH

PAIRS=${PAIRS:-data/pairs/index.jsonl}
STEPS=${STEPS:-400}
LR=${LR:-1e-6}
BETA_DPO=${BETA_DPO:-5000}
BETA_REDMD=${BETA_REDMD:-2.0}
WANDB_MODE=${WANDB_MODE:-offline}

for dim in MQ TA VQ; do
  NAME=dpo_${dim}
  OUT=runs/${NAME}
  if [ ! -d "$OUT" ]; then
    echo "[run] $NAME"
    python -m dpo_longlive.dpo_trainer \
      --pairs_index "$PAIRS" --reward_dim "$dim" \
      --out_dir "$OUT" --max_steps "$STEPS" \
      --beta "$BETA_DPO" --lr "$LR" --save_every "$STEPS" \
      --wandb_project dpo-longlive --wandb_name "$NAME" --wandb_mode "$WANDB_MODE" \
      2>&1 | tee logs/${NAME}.log
  fi
done

for dim in MQ TA VQ; do
  NAME=redmd_${dim}
  OUT=runs/${NAME}
  if [ ! -d "$OUT" ]; then
    echo "[run] $NAME"
    python -m dpo_longlive.redmd_trainer \
      --pairs_index "$PAIRS" --reward_dim "$dim" \
      --out_dir "$OUT" --max_steps "$STEPS" \
      --beta "$BETA_REDMD" --lr "$LR" --save_every "$STEPS" \
      --wandb_project dpo-longlive --wandb_name "$NAME" --wandb_mode "$WANDB_MODE" \
      2>&1 | tee logs/${NAME}.log
  fi
done

echo "[done] main"
