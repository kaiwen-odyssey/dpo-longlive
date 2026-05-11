#!/usr/bin/env python
"""Inject training and eval results into the paper's tables and produce final PDF inputs.

Reads:
  runs/main/training_metrics.json  (per-run lists of step metrics)
  runs/main/eval_summary.json      (per-run mean rewards on the eval set)
  runs/abl/training_metrics.json   (ablation: lr sweep on MQ)

Writes:
  paper/tables/results.tex          (main results table)
  paper/tables/ablation.tex         (lr ablation table)
  paper/figs/dpo_dynamics.pdf       (loss / margin / acc curves for main DPO)
  paper/figs/redmd_dynamics.pdf     (loss / weight curves for Re-DMD)
"""
import json, os, sys, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fmt(x):
    if x is None or (isinstance(x, float) and (x != x)):
        return "--"
    return f"{x:+.3f}"


def load_metrics(path):
    if not Path(path).exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_eval_summary(path):
    if not Path(path).exists():
        return {}
    with open(path) as f:
        return json.load(f)


def write_main_table(eval_summary, out):
    runs_order = ["base", "dpo_MQ", "dpo_TA", "dpo_VQ", "redmd_MQ", "redmd_TA", "redmd_VQ"]
    pretty = {
        "base": r"LongLive-1.3B (base)",
        "dpo_MQ": r"DPO (target=MQ)",
        "dpo_TA": r"DPO (target=TA)",
        "dpo_VQ": r"DPO (target=VQ)",
        "redmd_MQ": r"Re-DMD (target=MQ)",
        "redmd_TA": r"Re-DMD (target=TA)",
        "redmd_VQ": r"Re-DMD (target=VQ)",
    }
    base = eval_summary.get("base", {}).get("means", {})
    rows = []
    for r in runs_order:
        if r not in eval_summary:
            rows.append((pretty[r], "--", "--", "--", "--"))
            continue
        m = eval_summary[r]["means"]
        if r == "base":
            cells = (fmt(m.get("VQ")), fmt(m.get("MQ")), fmt(m.get("TA")), fmt(m.get("Overall")))
        else:
            cells = []
            for d in ["VQ", "MQ", "TA", "Overall"]:
                if d in m and d in base and base[d] is not None:
                    delta = m[d] - base[d]
                    cells.append(f"{m[d]:+.3f} ({delta:+.3f})")
                else:
                    cells.append("--")
            cells = tuple(cells)
        rows.append((pretty[r],) + cells)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Held-out evaluation (20 VidProm prompts) on the three VideoAlign reward dimensions plus their sum. "
                 r"Each cell is the mean normalized reward; the parenthesized $\Delta$ is the change vs.\ the un-tuned LongLive-1.3B base on the same prompt-and-seed pairs.}")
    lines.append(r"  \label{tab:main}")
    lines.append(r"  \begin{tabular}{lcccc}")
    lines.append(r"    \toprule")
    lines.append(r"    Run & VQ & MQ & TA & Overall \\")
    lines.append(r"    \midrule")
    for row in rows:
        if row[0] == r"DPO (target=MQ)":
            lines.append(r"    \midrule")
        if row[0] == r"Re-DMD (target=MQ)":
            lines.append(r"    \midrule")
        lines.append("    " + " & ".join(row) + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    Path(out).write_text("\n".join(lines) + "\n")


def write_ablation_table(abl_metrics, out):
    """abl_metrics: {ablation_dpo_MQ_lr1e-07: [step_metrics, ...], ...}"""
    rows = []
    for name in sorted(abl_metrics.keys()):
        if not name.startswith("ablation_dpo_MQ_lr"):
            continue
        lr = name.split("_lr")[-1]
        ms = abl_metrics[name]
        if not ms: continue
        last = ms[-1]
        # mean over last 25% of steps for stability
        tail = ms[max(0, int(len(ms) * 0.75)):]
        avg = lambda k: statistics.mean([m[k] for m in tail if k in m])
        rows.append((lr, fmt(avg("dpo_loss")), fmt(avg("dpo_margin")),
                     f"{avg('dpo_accuracy'):.2f}", fmt(avg("grad_norm"))))

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Learning-rate ablation on the MQ reward head (10 steps each, 60 training prompts). "
                 r"Reported values are means over the final 25\% of steps. We pick the row with positive margin and bounded gradient norm.}")
    lines.append(r"  \label{tab:abl}")
    lines.append(r"  \begin{tabular}{lcccc}")
    lines.append(r"    \toprule")
    lines.append(r"    Learning rate & DPO loss & Implicit margin & DPO accuracy & Grad norm \\")
    lines.append(r"    \midrule")
    for row in rows:
        lines.append("    " + " & ".join(row) + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    Path(out).write_text("\n".join(lines) + "\n")


def make_dynamics_plot(metrics, out_path, kind="dpo"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    if kind == "dpo":
        # Plot dpo_loss, dpo_margin, dpo_accuracy for each main DPO run
        for name in ["dpo_MQ", "dpo_TA", "dpo_VQ"]:
            ms = metrics.get(name, [])
            if not ms: continue
            steps = [m["step"] for m in ms]
            for ax, key, ylabel in zip(
                axes,
                ["dpo_loss", "dpo_margin", "dpo_accuracy"],
                ["DPO loss", "Implicit margin", "DPO accuracy"],
            ):
                vals = [m[key] for m in ms]
                ax.plot(steps, vals, label=name)
                ax.set_xlabel("step"); ax.set_ylabel(ylabel)
        for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=0.3)
    else:
        for name in ["redmd_MQ", "redmd_TA", "redmd_VQ"]:
            ms = metrics.get(name, [])
            if not ms: continue
            steps = [m["step"] for m in ms]
            for ax, key, ylabel in zip(
                axes,
                ["redmd_loss_pol", "exp_beta_reward", "denoising_gap"],
                ["Policy denoising loss", r"$\exp(\beta \cdot r)$", "Pol $-$ Ref denoising loss"],
            ):
                if key not in ms[0]: continue
                vals = [m[key] for m in ms]
                ax.plot(steps, vals, label=name)
                ax.set_xlabel("step"); ax.set_ylabel(ylabel)
        for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", dpi=150)
    plt.close(fig)


def main():
    out_tables = ROOT / "paper" / "tables"
    out_figs = ROOT / "paper" / "figs"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    main_metrics = load_metrics(ROOT / "runs" / "main" / "training_metrics.json")
    abl_metrics = load_metrics(ROOT / "runs" / "abl" / "training_metrics.json")
    eval_summary = load_eval_summary(ROOT / "runs" / "main" / "eval_summary.json")

    write_main_table(eval_summary, out_tables / "results.tex")
    write_ablation_table(abl_metrics, out_tables / "ablation.tex")
    if main_metrics:
        make_dynamics_plot(main_metrics, out_figs / "dpo_dynamics.pdf", kind="dpo")
        make_dynamics_plot(main_metrics, out_figs / "redmd_dynamics.pdf", kind="redmd")
    print(f"[paper] wrote {out_tables}/{{results,ablation}}.tex and {out_figs}/{{dpo,redmd}}_dynamics.pdf")


if __name__ == "__main__":
    main()
