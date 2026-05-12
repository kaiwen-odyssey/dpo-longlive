#!/usr/bin/env python
"""Plot eval-50 mean MQ + most-recent-20 training prompts mean MQ vs baseline.

For each checkpoint in runs/v4_mq/metrics.json:
  - eval50_mean_MQ           (50 held-out prompts)
  - mean of train_per_prompt[-20:] MQ (the 20 most-recent training prompts in the
    100-step window — i.e. prompts used in steps step_id-20 .. step_id)

Outputs paper/figs/v4_mq_eval_vs_recent20.png.
"""
from __future__ import annotations
import json, sys, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    metrics_path = ROOT / "runs/v4_mq/metrics.json"
    if not metrics_path.exists():
        print(f"[err] {metrics_path} not found"); sys.exit(1)
    m = json.loads(metrics_path.read_text())
    base_MQ = m["baseline_eval_mean_MQ"]
    ckpts = m["checkpoints"]
    if not ckpts:
        print("[err] no checkpoints yet"); sys.exit(1)

    steps = [c["step"] for c in ckpts]
    eval_mean = [c["eval50_mean_MQ"] for c in ckpts]
    eval_win  = [c["eval50_win_rate"] for c in ckpts]
    recent20_mean = []
    recent20_win  = []
    for c in ckpts:
        rows = c.get("train_per_prompt", [])
        last20 = rows[-20:] if len(rows) >= 20 else rows
        if last20:
            recent20_mean.append(statistics.mean(r["MQ"] for r in last20))
            wins = [1.0 if r["MQ"] > r["base_MQ"] else 0.0 for r in last20]
            recent20_win.append(statistics.mean(wins))
        else:
            recent20_mean.append(float("nan"))
            recent20_win.append(float("nan"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    ax = axes[0]
    ax.axhline(base_MQ, color="gray", linestyle="--", linewidth=1.0,
               label=f"base eval-50 mean MQ ({base_MQ:+.3f})")
    ax.plot(steps, eval_mean, marker="s", color="#1f77b4",
            label="eval-50 mean MQ (online policy, same noise)")
    ax.plot(steps, recent20_mean, marker="o", color="#d62728",
            label="recent-20 training mean MQ (online policy)")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("MQ (normalised)")
    ax.set_title("v4 dpo_MQ: mean reward")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, label="chance 0.5")
    ax.plot(steps, eval_win, marker="s", color="#1f77b4",
            label="eval-50 win-rate vs base@matched-seed")
    ax.plot(steps, recent20_win, marker="o", color="#d62728",
            label="recent-20 win-rate vs base@matched-seed")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("win-rate"); ax.set_ylim(0, 1)
    ax.set_title("v4 dpo_MQ: win-rate vs base")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    cfg = m.get("config", {})
    fig.suptitle(
        f"v4 dpo_MQ live: β={cfg.get('beta')} lr={cfg.get('lr')} α={cfg.get('anchor_alpha')} "
        f"steps={cfg.get('max_steps')} pool={cfg.get('train_pool')} eval={cfg.get('n_eval_prompts')}",
        fontsize=10
    )
    fig.tight_layout()
    out_path = ROOT / "paper/figs/v4_mq_eval_vs_recent20.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"[saved] {out_path}")

    # also print the current state
    print(f"\nbase eval-50 mean MQ: {base_MQ:+.4f}")
    print(f"{'step':>5}  {'eval50':>10}  {'Δ':>8}  {'win':>5}  {'rec20':>10}  {'win':>5}")
    for c in ckpts:
        i = ckpts.index(c)
        print(f"{c['step']:>5}  {eval_mean[i]:+10.4f}  {eval_mean[i]-base_MQ:+8.4f}  {eval_win[i]:>5.2f}  {recent20_mean[i]:+10.4f}  {recent20_win[i]:>5.2f}")


if __name__ == "__main__":
    main()
