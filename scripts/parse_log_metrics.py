#!/usr/bin/env python
"""Backup parser: extract per-step metrics from logs/main.log if training_metrics.json is missing."""
import re, json, sys
from pathlib import Path

LOG = Path("logs/main.log")
OUT = Path("runs/main/training_metrics.json")

run_re = re.compile(r"^==========\s+(\S+)\s+==========")
dpo_step_re = re.compile(
    r"\[\s*(\d+)/\s*(\d+)\]\s+loss=([+-]?[\d.]+)\s+margin=([+-]?[\d.]+)\s+acc=([\d.]+)\s+grad=([\d.]+)\s+mem=([\d.]+)GB"
)
redmd_step_re = re.compile(
    r"\[\s*(\d+)/\s*(\d+)\]\s+loss_pol=([+-]?[\d.]+)\s+reward=([+-]?[\d.]+)\s+w=([\d.]+)\s+grad=([\d.]+)\s+mem=([\d.]+)GB"
)

def main():
    if not LOG.exists():
        print("no log"); return
    metrics = {}
    cur = None
    for line in LOG.read_text().splitlines():
        m = run_re.search(line)
        if m:
            cur = m.group(1).strip()
            metrics[cur] = []
            continue
        if cur is None: continue
        if cur.startswith("dpo"):
            mm = dpo_step_re.search(line)
            if mm:
                metrics[cur].append({
                    "step": int(mm.group(1)),
                    "max_steps": int(mm.group(2)),
                    "dpo_loss": float(mm.group(3)),
                    "dpo_margin": float(mm.group(4)),
                    "dpo_accuracy": float(mm.group(5)),
                    "grad_norm": float(mm.group(6)),
                    "mem_gb": float(mm.group(7)),
                })
        elif cur.startswith("redmd"):
            mm = redmd_step_re.search(line)
            if mm:
                metrics[cur].append({
                    "step": int(mm.group(1)),
                    "max_steps": int(mm.group(2)),
                    "redmd_loss_pol": float(mm.group(3)),
                    "reward": float(mm.group(4)),
                    "exp_beta_reward": float(mm.group(5)),
                    "grad_norm": float(mm.group(6)),
                    "mem_gb": float(mm.group(7)),
                })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, indent=2))
    summary = {k: len(v) for k, v in metrics.items()}
    print(f"[parsed] {OUT} | {summary}")


if __name__ == "__main__":
    main()
