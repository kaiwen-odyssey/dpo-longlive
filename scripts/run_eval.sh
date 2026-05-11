#!/bin/bash
# Evaluate base + 6 trained checkpoints on the 20-prompt eval set.
set -e
source .venv/bin/activate
export PYTHONPATH=src:$PYTHONPATH

EVAL_PROMPTS=${EVAL_PROMPTS:-data/eval_prompts.jsonl}
OUT_DIR=runs/eval

mkdir -p "$OUT_DIR"

# Base
if [ ! -f "$OUT_DIR/base.jsonl" ]; then
  echo "[eval] base"
  python scripts/eval_run.py --prompts "$EVAL_PROMPTS" --name base \
    --out "$OUT_DIR/base.jsonl" 2>&1 | tee "$OUT_DIR/base.log"
fi

for run in dpo_MQ dpo_TA dpo_VQ redmd_MQ redmd_TA redmd_VQ; do
  if [ -f "$OUT_DIR/$run.jsonl" ]; then
    echo "[skip] $run already evaluated"
    continue
  fi
  CKPT=$(ls -t runs/$run/policy_step*.pt 2>/dev/null | head -1)
  if [ -z "$CKPT" ]; then
    echo "[skip] $run: no checkpoint"
    continue
  fi
  echo "[eval] $run from $CKPT"
  python scripts/eval_run.py --prompts "$EVAL_PROMPTS" --name "$run" \
    --ckpt "$CKPT" --out "$OUT_DIR/$run.jsonl" 2>&1 | tee "$OUT_DIR/$run.log"
done

# Compare
python scripts/compare_runs.py "$OUT_DIR"/*.jsonl --baseline base \
  --out "$OUT_DIR/summary.json"

echo "[done] eval"
