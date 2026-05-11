#!/usr/bin/env python
"""Plot MQ training metrics: v3 dpo_MQ (picked recipe, 400 steps, windowed averages)
on top of v2 dpo_MQ (first 30 steps, per-step) for context.

Outputs:
  paper/figs/mq_training.png   (panel: loss, implicit margin, accuracy)
  paper/figs/mq_training.pdf
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

v2_metrics = json.loads((ROOT / "runs/main_v2/training_metrics.json").read_text())
v2_mq = v2_metrics["dpo_MQ"]
v3_summary = json.loads((ROOT / "runs/main_v3/dpo_MQ/training_summary.json").read_text())

# v3 windowed averages -> (center_step, value)
windows = [
    ("steps 50–150",  100, v3_summary["window_steps_50_150"]),
    ("steps 150–250", 200, v3_summary["window_steps_150_250"]),
    ("steps 250–350", 300, v3_summary["window_steps_250_350"]),
    ("final 20",      390, v3_summary["final_window_last_20"]),
]
keys   = ["loss", "margin", "acc"]
titles = ["DPO loss", "Implicit margin", "DPO accuracy"]
v2_keys = {"loss": "dpo_loss", "margin": "dpo_margin", "acc": "dpo_accuracy"}

fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
v2_color = "#888"
v3_color = "#1f77b4"

for ax, key, title in zip(axes, keys, titles):
    # v2 per-step curve (30 steps)
    steps_v2 = [m["step"] for m in v2_mq]
    vals_v2  = [m[v2_keys[key]] for m in v2_mq]
    ax.plot(steps_v2, vals_v2, color=v2_color, marker="o", markersize=3,
            linewidth=1.0, alpha=0.65, label="v2 (β=500, 30 steps, per-step)")

    # v3 windowed averages (4 windows over 400 steps)
    xs = [c for _, c, _ in windows]
    ys = [w[key] for _, _, w in windows]
    ax.plot(xs, ys, color=v3_color, marker="s", markersize=8,
            linewidth=2.0, label="v3 (picked, 400 steps, window avg)")

    # zero-line on margin panel for reference
    if key == "margin":
        ax.axhline(0.0, color="k", linewidth=0.5, alpha=0.4, linestyle="--")
    if key == "acc":
        ax.axhline(0.5, color="k", linewidth=0.5, alpha=0.4, linestyle="--")

    ax.set_xlim(0, 410)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel(title)
    ax.grid(alpha=0.3)

axes[0].legend(loc="upper right", fontsize=8, framealpha=0.95)
fig.suptitle("MQ training metrics: v3 picked recipe (windowed) vs v2 (per-step)",
             fontsize=11)
fig.tight_layout()

out = ROOT / "paper" / "figs" / "mq_training"
out.parent.mkdir(parents=True, exist_ok=True)
for ext in (".png", ".pdf"):
    fig.savefig(out.with_suffix(ext), bbox_inches="tight", dpi=150)
    print(f"[saved] {out.with_suffix(ext)} ({out.with_suffix(ext).stat().st_size//1024} KB)")
plt.close(fig)
