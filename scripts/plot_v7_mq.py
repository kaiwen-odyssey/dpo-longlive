#!/usr/bin/env python
"""Plot eval-50 + train-100 trajectories for v7 (aggregated, β=100, lr=2e-5)."""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    metrics_path = ROOT / "runs/v7_mq_agg_b100_lr2e-5/metrics.json"
    if not metrics_path.exists():
        print(f"[err] {metrics_path} not found"); sys.exit(1)
    m = json.loads(metrics_path.read_text())
    base_MQ = m["baseline_eval_mean_MQ"]
    ckpts = m["checkpoints"]
    if not ckpts:
        print("[err] no checkpoints"); sys.exit(1)
    steps = [c["step"] for c in ckpts]
    eval_mean = [c["eval50_mean_MQ"] for c in ckpts]
    eval_win  = [c["eval50_win_rate"] for c in ckpts]
    train_mean = [c["train_window_mean_MQ"] for c in ckpts]
    train_win  = [c["train_window_win_rate"] for c in ckpts]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    ax = axes[0]
    ax.axhline(base_MQ, color="gray", linestyle="--", linewidth=1.0,
               label=f"base eval-50 mean MQ ({base_MQ:+.3f})")
    ax.plot(steps, eval_mean, marker="s", color="#1f77b4",
            label="eval-50 mean MQ (50 held-out prompts)")
    ax.plot(steps, train_mean, marker="o", color="#d62728",
            label="train-window-100 mean MQ (100 prompts/checkpoint)")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("MQ (normalised)")
    ax.set_title("v7 dpo_MQ: mean reward (aggregated, β=100, lr=2e-5)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, label="chance 0.5")
    ax.plot(steps, eval_win, marker="s", color="#1f77b4",
            label="eval-50 win-rate vs base@matched-seed")
    ax.plot(steps, train_win, marker="o", color="#d62728",
            label="train-window-100 win-rate vs base@matched-seed")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("win-rate"); ax.set_ylim(0, 1)
    ax.set_title("v7 dpo_MQ: win-rate vs base (matched per-prompt seed)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    cfg = m.get("config", {})
    agg = cfg.get("aggregate_chunks", True)
    fig.suptitle(
        f"v7 dpo_MQ: β={cfg.get('beta')} lr={cfg.get('lr')} α={cfg.get('anchor_alpha')} "
        f"agg={agg} steps={cfg.get('max_steps')} pool={cfg.get('train_pool')} eval={cfg.get('n_eval_prompts')}",
        fontsize=10
    )
    fig.tight_layout()
    out_path = ROOT / "paper/figs/v7_mq_progress.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
