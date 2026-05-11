#!/usr/bin/env python
"""Live plot of the in-progress v3 dpo_TA training metrics from logs/main_v3.log."""
import re, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "main_v3.log"

# Match: "  [ 110/400] loss=1.5048 margin=+6.7763 acc=0.46 grad=78.0 mem=70.3GB"
step_re = re.compile(
    r"\[\s*(\d+)/\s*400\]\s+loss=([+-]?[\d.]+)\s+margin=([+-]?[\d.]+)\s+acc=([\d.]+)\s+grad=([\d.]+)"
)

records = []
cur_run = None
for line in LOG.read_text().splitlines():
    m = re.match(r"=+\s+(\S+)\s+=+", line.strip())
    if m:
        cur_run = m.group(1)
        continue
    if cur_run != "dpo_TA":
        continue
    sm = step_re.search(line)
    if not sm:
        continue
    step, loss, margin, acc, grad = sm.groups()
    records.append({"step": int(step), "loss": float(loss),
                    "margin": float(margin), "acc": float(acc),
                    "grad": float(grad)})

if not records:
    print("no dpo_TA records yet"); sys.exit(1)

steps  = [r["step"] for r in records]
loss   = [r["loss"] for r in records]
margin = [r["margin"] for r in records]
acc    = [r["acc"] for r in records]
grad   = [r["grad"] for r in records]

print(f"parsed {len(records)} dpo_TA records, last step {steps[-1]}/400")

# Windowed summary (last 25% of records)
tail = records[max(0, int(len(records) * 0.75)):]
def mean(xs): return sum(xs)/len(xs) if xs else float("nan")
print(f"final-window mean: loss={mean([r['loss'] for r in tail]):.3f}  "
      f"margin={mean([r['margin'] for r in tail]):+.3f}  "
      f"acc={mean([r['acc']  for r in tail]):.3f}")
pos = sum(1 for r in records if r['margin'] > 0)
print(f"margin sign: {pos}/{len(records)} positive ({100*pos/len(records):.0f}%)")

fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
color = "#2ca02c"
panels = [
    ("DPO loss",        loss,   None),
    ("Implicit margin", margin, 0.0),
    ("DPO accuracy",    acc,    0.5),
    ("Grad norm (clip=1.0 internal)", grad, None),
]
for ax, (title, ys, hline) in zip(axes, panels):
    ax.plot(steps, ys, color=color, marker="o", markersize=4, linewidth=1.2)
    if hline is not None:
        ax.axhline(hline, color="k", linewidth=0.5, alpha=0.4, linestyle="--")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel(title)
    ax.set_xlim(0, max(410, steps[-1]+10))
    ax.grid(alpha=0.3)

fig.suptitle(f"v3 dpo_TA in-progress training (step {steps[-1]}/400, "
             f"β=500, lr=2e-5, grad_accum=4, anchor α=1.0, min_gap=0.2)",
             fontsize=11)
fig.tight_layout()
out = ROOT / "paper" / "figs" / "ta_training_live.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight", dpi=150)
print(f"[saved] {out} ({out.stat().st_size//1024} KB)")
plt.close(fig)
