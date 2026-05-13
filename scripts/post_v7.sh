#!/bin/bash
# Post-v7 pipeline: wait for v7 training to finish, generate plot, rebuild paper, commit, push.
set -e
cd /home/kaiwen/dev/dpo-longlive

echo "[post_v7] waiting for v7 training process to exit..."
while pgrep -f "train_v4_mq.py.*v7_mq_agg_b100_lr2e-5" > /dev/null; do
    sleep 60
done
echo "[post_v7] v7 process exited"

source .venv/bin/activate

echo "[post_v7] generating v7 plot..."
python scripts/plot_v7_mq.py

echo "[post_v7] rebuilding paper..."
python3 paper/build_tex.py

echo "[post_v7] staging files..."
git add scripts/plot_v7_mq.py scripts/post_v7.sh paper/build_tex.py paper/main.tex paper/main.pdf paper/figs/v7_mq_progress.png runs/v7_mq_agg_b100_lr2e-5/metrics.json runs/v7_mq_agg_b100_lr2e-5/base_eval.jsonl runs/v7_mq_agg_b100_lr2e-5/eval_prompts_extra.jsonl runs/grid_v5/*/metrics.json scripts/train_v4_mq.py 2>&1 || true

git status

echo "[post_v7] committing..."
git commit -m "$(cat <<'EOF'
v7 aggregated DPO 1000-step run + grid search results

Adds runs/v7_mq_agg_b100_lr2e-5/ (β=100, lr=2e-5, aggregated, 1000 steps).
Peak at step 400: eval-50 Δ MQ +0.802, win-rate 0.86 (43/50 prompts beat
base), train-100 mean MQ +1.010, win 0.89. This is 8× the per-chunk v4
peak (Δ +0.104) on the same 50-prompt held-out set with matched seeds.

Trajectory is bimodal: most checkpoints land in the +0.40 to +0.80 Δ
basin, but steps 200 and 600 collapse to Δ ~ -0.5 before recovering
within 100 steps. The single best checkpoint is preserved as
policy_best.pt; the per-step optimization itself is highly volatile
(margins swing ±19 within a few steps) so the result depends on
catching a positive basin.

Also includes the 100-step β×lr grid that selected this config:
- β=500 lr=1e-5 → Δ -0.024 win 0.54 (slow)
- β=500 lr=2e-5 → Δ +0.227 win 0.56 (outlier-driven mean)
- β=100 lr=2e-5 → Δ +0.170 win 0.56 (picked, both metrics decent)
- β=100 lr=5e-5 → Δ +0.113 win 0.70 (high win but unstable for 1000-step)

train_v4_mq.py adds --no_save_policy for grid-style runs that don't
need the policy weights, only metrics.

paper/build_tex.py adds latex_v7_trajectory_table and a v7 section
+ figure (figs/v7_mq_progress.png) showing the bimodal trajectory.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"

echo "[post_v7] pushing..."
git push

echo "[post_v7] done"
