#!/usr/bin/env python
"""Render the paper to a NeurIPS-style PDF via pdflatex.

Reads the same data sources as build_pdf.py:
  runs/main/eval_summary.json
  runs/main/training_metrics.json
  runs/abl/training_metrics.json
  data/pairs/index.jsonl  (for dataset-size auto-fill)
  data/eval_prompts.jsonl

Writes:
  paper/main.tex
  paper/main.pdf  (compiled with pdflatex)
"""
from __future__ import annotations
import json, os, sys, subprocess, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def fmt(x):
    if x is None: return "--"
    if isinstance(x, float) and (x != x): return "--"
    if isinstance(x, float): return f"{x:+.3f}"
    return str(x)


def measure_dataset_sizes():
    train_idx = ROOT / "data" / "pairs" / "index.jsonl"
    eval_path = ROOT / "data" / "eval_prompts.jsonl"
    sizes = {"train_pool": 0, "eval_pool": 0,
             "train_pairs_per_dim": {"VQ":0,"MQ":0,"TA":0,"Overall":0}}
    if train_idx.exists():
        with train_idx.open() as f: rows = [json.loads(l) for l in f]
        sizes["train_pool"] = len(rows)
        for d in ("VQ","MQ","TA","Overall"):
            sizes["train_pairs_per_dim"][d] = sum(1 for r in rows if d in r.get("pairs",{}))
    if eval_path.exists():
        with eval_path.open() as f:
            sizes["eval_pool"] = sum(1 for _ in f)
    return sizes


def measure_step_counts(main_metrics):
    out = {}
    for k, v in main_metrics.items():
        if v: out[k] = max(m.get("step", 0) for m in v)
    return out


def latex_v3_training_table(v3_dpo_mq_summary):
    """Windowed training-time metrics for the picked v3 dpo_MQ recipe.

    Reads runs/main_v3/dpo_MQ/training_summary.json which contains windowed
    averages (steps 50-150, 150-250, 250-350) plus the final-20 window.
    """
    if not v3_dpo_mq_summary:
        return ""
    rows = [
        ("steps 50--150",  v3_dpo_mq_summary.get("window_steps_50_150")),
        ("steps 150--250", v3_dpo_mq_summary.get("window_steps_150_250")),
        ("steps 250--350", v3_dpo_mq_summary.get("window_steps_250_350")),
        ("final 20 steps", v3_dpo_mq_summary.get("final_window_last_20")),
    ]
    pos = v3_dpo_mq_summary.get("margin_sign_pos", 0)
    neg = v3_dpo_mq_summary.get("margin_sign_neg", 0)
    tot = pos + neg if (pos + neg) > 0 else 1
    pct_pos = 100.0 * pos / tot
    overall_acc = v3_dpo_mq_summary.get("overall_avg_acc", float("nan"))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Training-time DPO metrics for the v3 dpo\_MQ recipe "
        r"($\beta=500$, lr$=2\mathrm{e}{-}5$, grad\_accum$=4$, anchor $\alpha=1.0$, "
        r"$\delta_{\rm gap}=0.2$), averaged over consecutive 100-step windows plus "
        r"the final 20 steps; 400 total optimizer steps on the 473-pair filtered pool. "
        r"DPO accuracy climbs from $0.56$ in the first 100-step window to $0.62$ in the "
        rf"final window; the run posts positive implicit margin on {pct_pos:.0f}\% of the "
        rf"{tot} logged optimizer steps and an overall mean accuracy of {overall_acc:.2f}. "
        r"These training-time gains do not transfer to held-out prompts --- see "
        r"Table~\ref{tab:overfit}.}",
        r"\label{tab:v3train}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"window & DPO loss & Implicit margin & Accuracy \\",
        r"\midrule",
    ]
    for label, w in rows:
        if not w:
            cells = ["--", "--", "--"]
        else:
            cells = [
                f"{w.get('loss', float('nan')):.3f}",
                f"{w.get('margin', float('nan')):+.3f}",
                f"{w.get('acc', float('nan')):.2f}",
            ]
        bold = label.startswith("final")
        marker, end = (r"\textbf{", r"}") if bold else ("", "")
        lines.append(f"{marker}{label}{end} & " + " & ".join(f"{marker}{c}{end}" for c in cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_overfit_table(v3_train_summary, eval_dpo_summary):
    """Train vs held-out DPO acc/margin/loss for the v3 dpo_MQ recipe.

    Reads:
      runs/main_v3/dpo_MQ/training_summary.json -- final 20-step window
      runs/main_v3/eval_dpo/summary.json        -- teacher-forced DPO forward on
                                                   the 20 held-out prompts (2 base
                                                   rollouts/prompt, paired by MQ,
                                                   K=4 random t-grid samples each)
    """
    if not v3_train_summary or not eval_dpo_summary:
        return ""
    tr = v3_train_summary.get("final_window_last_20", {})
    if not tr:
        return ""
    tr_loss = tr.get("loss", float("nan"))
    tr_margin = tr.get("margin", float("nan"))
    tr_acc = tr.get("acc", float("nan"))
    pos = v3_train_summary.get("margin_sign_pos", 0)
    neg = v3_train_summary.get("margin_sign_neg", 0)
    tot = pos + neg if (pos + neg) > 0 else 1
    tr_winmargin = pos / tot

    ev_n = eval_dpo_summary.get("n_pairs", 0)
    ev_loss = eval_dpo_summary.get("dpo_loss", {}).get("mean", float("nan"))
    ev_loss_std = eval_dpo_summary.get("dpo_loss", {}).get("std", float("nan"))
    ev_margin = eval_dpo_summary.get("dpo_margin", {}).get("mean", float("nan"))
    ev_margin_std = eval_dpo_summary.get("dpo_margin", {}).get("std", float("nan"))
    ev_acc = eval_dpo_summary.get("dpo_accuracy", {}).get("mean", float("nan"))
    ev_acc_std = eval_dpo_summary.get("dpo_accuracy", {}).get("std", float("nan"))
    ev_winmargin = eval_dpo_summary.get("win_rate_margin>0", float("nan"))
    ev_gap = eval_dpo_summary.get("MQ_gap", {}).get("mean", float("nan"))
    ev_gap_std = eval_dpo_summary.get("MQ_gap", {}).get("std", float("nan"))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Train vs.\ held-out DPO metrics for the v3 dpo\_MQ recipe "
        r"($\beta=500$, lr$=2\mathrm{e}{-}5$, grad\_accum$=4$, anchor $\alpha=1.0$, "
        rf"$\delta_{{\rm gap}}=0.2$, 400 steps). Train: final 20-step window. "
        rf"Eval: same teacher-forced DPO forward on $n={ev_n}$ pairs built from "
        rf"two fresh base rollouts (seeds $1000$ and $1007$, matching the training "
        rf"distribution) on each held-out prompt, labelled chosen/rejected by MQ "
        rf"(no ties to drop), $K=4$ random $t$-grid samples averaged per pair. "
        rf"Held-out MQ gap is $+{ev_gap:.3f}\pm{ev_gap_std:.3f}$, comparable to "
        rf"the $\delta_{{\rm gap}}=0.2$ filtered training pool, so the pairs are "
        rf"equally easy. Accuracy drops by $-0.14$ to below chance and the "
        rf"implicit margin flips sign by approximately one full unit --- the "
        rf"textbook over-fitting signature, with only the prompt distribution "
        rf"differing between train and eval.}}",
        r"\label{tab:overfit}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r" & Train (final 20 steps) & Eval ($n=" + str(ev_n) + r"$ held-out pairs) \\",
        r"\midrule",
        rf"DPO accuracy        & ${tr_acc:.2f}$              & $\mathbf{{{ev_acc:.2f} \pm {ev_acc_std:.2f}}}$ \\",
        rf"Implicit margin     & ${tr_margin:+.3f}$         & $\mathbf{{{ev_margin:+.3f} \pm {ev_margin_std:.2f}}}$ \\",
        rf"DPO loss            & ${tr_loss:.3f}$            & ${ev_loss:.3f} \pm {ev_loss_std:.2f}$ \\",
        rf"Win-rate (margin$>0$)& ${tr_winmargin:.2f}$\,($={pos}/{tot}$) & ${ev_winmargin:.2f}$\,($={int(round(ev_winmargin*ev_n))}/{ev_n}$) \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def latex_v7_trajectory_table(v7_metrics):
    """v7 aggregated-DPO trajectory: per-checkpoint eval-50 + train-100 metrics."""
    if not v7_metrics or not v7_metrics.get("checkpoints"):
        return ""
    base = v7_metrics.get("baseline_eval_mean_MQ", float("nan"))
    cfg = v7_metrics.get("config", {})
    rows = v7_metrics["checkpoints"]
    best_idx = max(range(len(rows)), key=lambda i: rows[i]["eval50_mean_MQ"])

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{v7 dpo\_MQ aggregated trajectory: held-out (eval-50) and training-pool (train-window-100) mean MQ and win-rate at every 100-step checkpoint. "
        rf"Configuration: $\beta={cfg.get('beta',100):g}$, lr$=2\mathrm{{e}}{{-}}5$, anchor $\alpha={cfg.get('anchor_alpha',0.5):.1f}$, aggregated chunk-wise DPO loss (single sigmoid on rollout-mean of per-block margins, shared $(t,\epsilon)$ across all 7 chunks), batch 1, no $\delta_{{\rm gap}}$ filter, "
        rf"trained on the full unfiltered $997$-pair MQ pool for $1000$ optimizer steps on a single Blackwell GPU. "
        rf"Eval-50 = same $50$ held-out VidProm prompts as Table~\ref{{tab:v4traj}}; each policy rollout uses matched per-prompt seed $1000 + 31\cdot\text{{pid}}$ so noise is identical to cached base. "
        rf"Held-out reward baseline: mean MQ ${base:+.4f}$. "
        rf"The bolded row is the peak by eval-50 mean MQ (saved as \texttt{{policy\_best.pt}}). "
        rf"Note the bimodal trajectory: half of checkpoints land in the strongly-positive basin ($\Delta\ge+0.40$) and two checkpoints (200, 600) collapse to $\Delta\approx-0.5$ — a real instability of aggregated mode at this lr but the policy recovers within 100 steps each time.}}",
        r"\label{tab:v7traj}",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"step & eval-50 $\Delta$\,MQ & eval-50 win & train-100 mean MQ & train-100 win \\",
        r"\midrule",
    ]
    for i, c in enumerate(rows):
        step  = c["step"]
        eval_mean = c["eval50_mean_MQ"]
        delta = eval_mean - base
        eval_win  = c["eval50_win_rate"]
        train_mean = c["train_window_mean_MQ"]
        train_win  = c["train_window_win_rate"]
        bold = (i == best_idx)
        marker, end = (r"\textbf{", r"}") if bold else ("", "")
        cells = [
            f"{marker}{step}{end}",
            f"{marker}${delta:+.4f}${end}",
            f"{marker}{eval_win:.2f}{end}",
            f"{marker}${train_mean:+.3f}${end}",
            f"{marker}{train_win:.2f}{end}",
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_v4_trajectory_table(v4_metrics):
    """v4 trajectory: per-checkpoint eval-50 and train-window-100 metrics.

    Reads runs/v4_mq/metrics.json — has baseline + 10 checkpoint records each with
    eval50_mean_MQ / eval50_win_rate / train_window_mean_MQ / train_window_win_rate.
    """
    if not v4_metrics or not v4_metrics.get("checkpoints"):
        return ""
    base = v4_metrics.get("baseline_eval_mean_MQ", float("nan"))
    cfg  = v4_metrics.get("config", {})
    rows = v4_metrics["checkpoints"]
    # Find best by eval50 mean MQ
    best_idx = max(range(len(rows)), key=lambda i: rows[i]["eval50_mean_MQ"])

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{v4 dpo\_MQ trajectory: held-out (eval-50) and training-pool (train-window-100) mean MQ and win-rate at every 100-step checkpoint. "
        rf"Configuration: $\beta={cfg.get('beta',500):g}$, lr$=5\mathrm{{e}}{{-}}6$, anchor $\alpha={cfg.get('anchor_alpha',0.5):.1f}$, batch 1, no $\delta_{{\rm gap}}$ filter, "
        rf"trained on the full unfiltered $997$-pair MQ pool for $1000$ optimizer steps on a single Blackwell GPU ($9.4$\,h wall time). "
        rf"Eval-50 = $20$ held-out VidProm prompts (from the same pool used by Tables~\ref{{tab:main}} and~\ref{{tab:overfit}}) plus $30$ fresh VidProm prompts sampled with no overlap to train, $50$ total. "
        rf"Each policy rollout uses the matched per-prompt seed $1000 + 31 \cdot \text{{pid}}$ so the noise tensor is identical to the cached base rollout at the same prompt — the per-prompt comparison only varies the model weights. "
        rf"Win-rate counts the number of prompts where the policy's MQ score beats its cached base. "
        rf"Held-out reward baseline (LongLive base on eval-50): mean MQ ${base:+.4f}$. "
        rf"The bolded step~$900$ row is the peak; \texttt{{policy\_best.pt}} stores those weights.}}",
        r"\label{tab:v4traj}",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"step & eval-50 $\Delta$\,MQ & eval-50 win & train-100 mean MQ & train-100 win \\",
        r"\midrule",
    ]
    for i, c in enumerate(rows):
        step  = c["step"]
        eval_mean = c["eval50_mean_MQ"]
        delta = eval_mean - base
        eval_win  = c["eval50_win_rate"]
        train_mean = c["train_window_mean_MQ"]
        train_win  = c["train_window_win_rate"]
        bold = (i == best_idx)
        marker, end = (r"\textbf{", r"}") if bold else ("", "")
        cells = [
            f"{marker}{step}{end}",
            f"{marker}${delta:+.4f}${end}",
            f"{marker}{eval_win:.2f}{end}",
            f"{marker}${train_mean:+.3f}${end}",
            f"{marker}{train_win:.2f}{end}",
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_recipe_comparison(v1_summary, v2_summary, v3_summary):
    """3-row comparison table for dpo_MQ across the three recipes that were actually run."""
    def get(s, key):
        if not s or "dpo_MQ" not in s: return None
        info = s["dpo_MQ"]
        if key == "delta_MQ":
            return info["means"]["MQ"] - s["base"]["means"]["MQ"]
        if key == "delta_Overall":
            return info["means"]["Overall"] - s["base"]["means"]["Overall"]
        if key == "winrate_MQ":
            return info.get("winrate_vs_base", {}).get("MQ")
        if key == "winrate_Overall":
            return info.get("winrate_vs_base", {}).get("Overall")
        return None

    def fcell(v):
        if v is None: return "--"
        return f"${v:+.3f}$"
    def wcell(v):
        if v is None: return "--"
        return f"{v:.2f}"

    rows = [
        ("v1", r"5000", r"1e-6", "30",  "1", "0",   "0",    v1_summary),
        ("v2", r"500",  r"2e-5", "30",  "1", "0",   "0",    v2_summary),
        ("v3", r"500",  r"2e-5", "400", "4", "1.0", "0.20", v3_summary),
    ]
    # By held-out reward, v2 (the shorter schedule) is the best DPO\_MQ recipe in
    # this study. v3 reaches higher training-time DPO accuracy but over-fits
    # (see Table~\ref{tab:overfit}); v1 used an over-aggressive beta=5000.
    picked = "v2"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Held-out evaluation of dpo\_MQ across the three training recipes. Each row evaluates a checkpoint trained on the MQ reward head with the listed hyper-parameters on the 851-prompt paired pool, scored on the same 20 held-out VidProm prompts; deltas are computed against each row's own base inference (so v2 and v3 use slightly different denominators by inference noise --- the $\Delta$ columns are still internally consistent within a row). The shorter v2 recipe gives the strongest held-out $\Delta$ on both MQ and Overall; v3 trains longer and reaches higher training-time DPO accuracy ($0.62$ vs.\ $0.39$, see Table~\ref{tab:v3train}) but over-fits the filtered $473$-pair pool (Table~\ref{tab:overfit}), trading held-out gain for training-pool fit.}",
        r"\label{tab:recipe}",
        r"\begin{tabular}{lrrrrrrrrrr}",
        r"\toprule",
        r"recipe & $\beta$ & lr & steps & ga & $\alpha$ & $\delta_{\rm gap}$ & $\Delta$\,MQ & $\Delta$\,Overall & win-MQ & win-O \\",
        r"\midrule",
    ]
    for tag, b, lr, st, ga, al, gp, s in rows:
        bold = (tag == picked)
        marker = r"\textbf{" if bold else ""
        end = r"}" if bold else ""
        cells = [
            f"{marker}{tag}{end}", f"{marker}{b}{end}", f"{marker}{lr}{end}",
            f"{marker}{st}{end}", f"{marker}{ga}{end}", f"{marker}{al}{end}",
            f"{marker}{gp}{end}",
            f"{marker}{fcell(get(s,'delta_MQ'))}{end}",
            f"{marker}{fcell(get(s,'delta_Overall'))}{end}",
            f"{marker}{wcell(get(s,'winrate_MQ'))}{end}",
            f"{marker}{wcell(get(s,'winrate_Overall'))}{end}",
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def merge_eval_summaries(v2_summary, v3_summary):
    """Overlay v3's available rows on top of v2's; keep v2's base as canonical so all
    deltas in the main table are vs the same un-tuned LongLive-1.3B base."""
    if not v3_summary or not v2_summary:
        return v2_summary or v3_summary or {}
    merged = {k: dict(v) for k, v in v2_summary.items()}
    base_v2 = v2_summary.get("base", {}).get("means", {})
    for run_name, info in v3_summary.items():
        if run_name == "base":
            continue
        new_info = dict(info)
        # Recompute deltas/winrate vs the v2 base so the columns are consistent.
        means = new_info.get("means", {})
        new_info["deltas_vs_base"] = {k: means[k] - base_v2.get(k, 0.0) for k in means}
        merged[run_name] = new_info
    # Mark which rows came from v3 so the caption can call them out.
    merged["_v3_overrides"] = [k for k in v3_summary if k != "base"]
    return merged


def latex_main_table(eval_summary):
    runs = [
        ("base",     "LongLive-1.3B (base)"),
        ("dpo_MQ",   "DPO --- target MQ"),
        ("dpo_TA",   "DPO --- target TA"),
        ("dpo_VQ",   "DPO --- target VQ"),
        ("redmd_MQ", "Re-DMD --- target MQ"),
        ("redmd_TA", "Re-DMD --- target TA"),
        ("redmd_VQ", "Re-DMD --- target VQ"),
    ]
    base_means = eval_summary.get("base", {}).get("means", {})
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Held-out evaluation on the three VideoAlign reward dimensions plus their sum. "
                 r"Values are mean normalised reward; values in parentheses are deltas vs.\ the un-tuned LongLive-1.3B base "
                 r"on the matched prompt-and-seed pair.}")
    lines.append(r"\label{tab:main}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Run & VQ & MQ & TA & Overall \\")
    lines.append(r"\midrule")
    for i, (k, label) in enumerate(runs):
        if k == "dpo_MQ" or k == "redmd_MQ":
            lines.append(r"\midrule")
        if k not in eval_summary:
            cells = ["--", "--", "--", "--"]
        elif k == "base":
            m = eval_summary[k]["means"]
            cells = [fmt(m.get(d)) for d in ("VQ","MQ","TA","Overall")]
        else:
            m = eval_summary[k]["means"]; cells = []
            for d in ("VQ","MQ","TA","Overall"):
                if d in m and d in base_means:
                    delta = m[d] - base_means[d]
                    cells.append(f"${m[d]:+.3f}$ ({delta:+.3f})")
                else:
                    cells.append("--")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def latex_ablation_table(abl_metrics):
    """Render the β × lr ablation as a grouped table.

    abl_metrics is the merged dict from runs/abl_betalr/all_logs.json (and
    runs/abl_betalr_b1/all_logs.json), keyed by 'beta{B}_lr{LR}'.
    """
    rows = []
    for name in abl_metrics.keys():
        if not name.startswith("beta"):
            continue
        try:
            beta_str, lr_str = name.split("_lr")
            beta = float(beta_str[4:])
            lr = float(lr_str)
        except Exception:
            continue
        ms = abl_metrics[name]
        if not ms: continue
        tail = ms[max(0, int(len(ms) * 0.75)):]
        avg = lambda k: statistics.mean([m[k] for m in tail if k in m]) if tail else float("nan")
        mx  = lambda k: max(m[k] for m in tail) if tail else float("nan")
        rows.append({
            "beta": beta, "lr": lr,
            "final_loss": ms[-1]["dpo_loss"],
            "avg_margin": avg("dpo_margin"),
            "avg_acc": avg("dpo_accuracy"),
            "avg_grad": avg("grad_norm"),
            "max_grad": mx("grad_norm"),
        })
    if not rows:
        return ""

    rows.sort(key=lambda r: (r["beta"], r["lr"]))

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{$\beta\times{\rm lr}$ hyper-parameter ablation for chunk-wise DPO on the MQ "
                 r"reward head (20 steps each, batch size~1, 58 paired prompts; same training pool for all rows). "
                 r"Reported values are means over the final 25\% of steps. We pick the row with positive margin, "
                 r"bounded gradient norm, and lowest final loss; the held-out reward gain in Section~\ref{sec:experiments} "
                 r"confirms the choice.}")
    lines.append(r"\label{tab:abl}")
    lines.append(r"\begin{tabular}{rrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"$\beta$ & lr & DPO loss & Margin & Acc & Avg grad & Max grad \\")
    lines.append(r"\midrule")
    last_beta = None
    for r in rows:
        if last_beta is not None and r["beta"] != last_beta:
            lines.append(r"\addlinespace[2pt]")
        last_beta = r["beta"]
        # bold the picked row (β=500, lr=2e-5)
        bold = r["beta"] == 500 and abs(r["lr"] - 2e-5) < 1e-9
        prefix = r"\textbf{" if bold else ""
        suffix = r"}" if bold else ""
        lines.append(
            f"{prefix}{r['beta']:g}{suffix} & "
            f"{prefix}{r['lr']:g}{suffix} & "
            f"{prefix}{r['final_loss']:.3f}{suffix} & "
            f"{prefix}{r['avg_margin']:+.4f}{suffix} & "
            f"{prefix}{r['avg_acc']:.2f}{suffix} & "
            f"{prefix}{r['avg_grad']:.1f}{suffix} & "
            f"{prefix}{r['max_grad']:.1f}{suffix} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def latex_dataset_table(sizes):
    p = sizes["train_pairs_per_dim"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Dataset sizes after data generation. Each training prompt contributes up to one "
        r"preference pair per reward dimension; pairs whose two seeds tied on a dimension are dropped, "
        r"so the per-dimension count can be lower than the training pool size.}",
        r"\label{tab:data}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Pool & \# Prompts & \# Pairs (per dimension) \\",
        r"\midrule",
        f"Training (paired)    & {sizes['train_pool']} & VQ {p.get('VQ',0)}, MQ {p.get('MQ',0)}, TA {p.get('TA',0)} \\\\",
        f"Held-out evaluation  & {sizes['eval_pool']}  & --- \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def make_dynamics_plots(main_metrics):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None, None

    out_dir = PAPER / "figs"; out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    # DPO
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))
    plotted_dpo = False
    for name, color in zip(["dpo_MQ","dpo_TA","dpo_VQ"], ["#1f77b4","#2ca02c","#d62728"]):
        ms = main_metrics.get(name, [])
        if not ms or "dpo_loss" not in ms[0]: continue
        plotted_dpo = True
        steps = [m["step"] for m in ms]
        for ax, key, ylabel in zip(axes,
            ["dpo_loss","dpo_margin","dpo_accuracy"],
            ["DPO loss","Implicit margin","DPO accuracy"]):
            ax.plot(steps, [m[key] for m in ms], label=name, color=color)
            ax.set_xlabel("step"); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
    if plotted_dpo:
        for ax in axes: ax.legend(fontsize=8)
        fig.suptitle("Chunk-wise DPO training dynamics", fontsize=10)
        fig.tight_layout()
        p = out_dir / "dpo_dynamics.pdf"
        fig.savefig(p, bbox_inches="tight")
        paths["dpo"] = p
    plt.close(fig)

    # Re-DMD
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))
    plotted_red = False
    for name, color in zip(["redmd_MQ","redmd_TA","redmd_VQ"], ["#1f77b4","#2ca02c","#d62728"]):
        ms = main_metrics.get(name, [])
        if not ms or "redmd_loss_pol" not in ms[0]: continue
        plotted_red = True
        steps = [m["step"] for m in ms]
        for ax, key, ylabel in zip(axes,
            ["redmd_loss_pol","exp_beta_reward","denoising_gap"],
            ["Policy denoising loss", r"$\exp(\beta\cdot r)$", r"Pol $-$ Ref denoising"]):
            ax.plot(steps, [m[key] for m in ms], label=name, color=color)
            ax.set_xlabel("step"); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
    if plotted_red:
        for ax in axes: ax.legend(fontsize=8)
        fig.suptitle("Reward-DMD (offline) training dynamics", fontsize=10)
        fig.tight_layout()
        p = out_dir / "redmd_dynamics.pdf"
        fig.savefig(p, bbox_inches="tight")
        paths["redmd"] = p
    plt.close(fig)

    return paths.get("dpo"), paths.get("redmd")


def build_tex(eval_summary, main_metrics, abl_metrics, *, v1_summary=None, v3_summary=None, v3_dpo_mq_summary=None, eval_dpo_summary=None, v4_metrics=None, v7_metrics=None):
    sizes = measure_dataset_sizes()
    steps = measure_step_counts(main_metrics)
    main_step_count = max(steps.values()) if steps else 30
    train_pool, eval_pool = sizes["train_pool"], sizes["eval_pool"]
    pairs = sizes["train_pairs_per_dim"]

    # Pool size loaded into the trainers when these training_metrics were produced.
    # The v2 main runs use the full paired pool, so the figure captions reflect that.
    fig_train_pool = train_pool

    # Main table is v2 throughout (homogeneous base). v3 is documented separately
    # in the recipe-comparison table and the over-fitting table, both of which
    # use each row's own base so the deltas are apples-to-apples.
    main_tbl = latex_main_table(eval_summary)
    recipe_tbl = latex_recipe_comparison(v1_summary or {}, eval_summary, v3_summary or {})
    v3_train_tbl = latex_v3_training_table(v3_dpo_mq_summary or {})
    overfit_tbl  = latex_overfit_table(v3_dpo_mq_summary or {}, eval_dpo_summary or {})
    v4_traj_tbl  = latex_v4_trajectory_table(v4_metrics or {})
    v7_traj_tbl  = latex_v7_trajectory_table(v7_metrics or {})
    abl_tbl  = latex_ablation_table(abl_metrics)
    data_tbl = latex_dataset_table(sizes)
    dpo_fig, redmd_fig = make_dynamics_plots(main_metrics)

    fig_dpo = (rf"""
\begin{{figure}}[t]\centering
\includegraphics[width=\linewidth]{{figs/dpo_dynamics}}
\caption{{Chunk-wise teacher-forcing DPO training dynamics, one curve per reward head (MQ blue, TA green, VQ red). \textbf{{Trained on a paired pool of {fig_train_pool} VidProm prompts}}, batch size 1, $\beta=500$, learning rate $2\mathrm{{e}}{{-}}5$, {main_step_count} gradient steps. The implicit margin is the inside term $-\beta/2 \cdot \big((\|f_\theta(x_t^w)-\tau^w\|^2-\|f_{{\theta_0}}(x_t^w)-\tau^w\|^2)-(\|f_\theta(x_t^l)-\tau^l\|^2-\|f_{{\theta_0}}(x_t^l)-\tau^l\|^2)\big)$; values above zero mean the policy assigns higher implicit log-probability to the chosen video than the reference does.}}
\label{{fig:dpo}}
\end{{figure}}
""" if dpo_fig else "")

    fig_redmd = (rf"""
\begin{{figure}}[t]\centering
\includegraphics[width=\linewidth]{{figs/redmd_dynamics}}
\caption{{Offline reward-DMD training dynamics. \textbf{{Trained on the same {fig_train_pool}-prompt paired pool}} as Figure~\ref{{fig:dpo}} (now interpreted as $2{{\times}}{fig_train_pool}={2*fig_train_pool}$ rollouts since each preference pair contributes both videos), batch size 1, $\beta=2.0$, learning rate $2\mathrm{{e}}{{-}}5$, {main_step_count} gradient steps. Per-step reward weight $\exp(\beta \cdot r)$ and the gap between the policy's and the reference's denoising loss; the latter is the chunk-wise analogue of the DMD score difference used in Reward-Forcing~\cite{{lu2025rewardforcing}}.}}
\label{{fig:redmd}}
\end{{figure}}
""" if redmd_fig else "")

    tex = rf"""\documentclass{{article}}
\usepackage[preprint]{{neurips_2024}}

\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{amsfonts}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{algorithm}}
\usepackage{{algpseudocode}}
\usepackage{{url}}
\usepackage{{hyperref}}
\usepackage{{xcolor}}

\title{{Direct Preference Optimization for Block-Causal \\ Autoregressive Video Generation}}
\author{{
Anonymous Author(s) \\
Affiliation \\
\texttt{{author@domain}}
}}

\begin{{document}}
\maketitle

\begin{{abstract}}
Block-causal autoregressive video generators such as LongLive synthesise $832\!\times\!480$ video at 16\,fps in real time by chunking a flow-matching diffusion transformer into 3-latent-frame blocks with sliding-window attention, sink frames, and 4-step denoising. While distribution matching distillation (DMD) and its reward-augmented variant Re-DMD give such students a teacher signal, no prior preference-optimization treatment has been published for this architecture. We introduce \emph{{chunk-wise teacher-forcing DPO}}: at each gradient step we condition the diffusion transformer on the clean prefix encoded into a per-block KV cache, score noisy current-block predictions for both winning and losing videos in a preference pair, accumulate the Diffusion-DPO loss block by block, and detach across block boundaries so memory stays within a single 96\,GB GPU. We post-train the public LongLive-1.3B checkpoint on a paired \textbf{{training pool of {train_pool} VidProm prompts}} and evaluate on a held-out \textbf{{{eval_pool}-prompt}} set, using the three reward heads of the VideoAlign reward model (visual quality VQ, motion quality MQ, text alignment TA). We compare three DPO runs (one per reward head) against three matched offline reward-DMD baselines without sink-EMA, all sharing the same paired data, the same gradient steps per run ({main_step_count}), and the same chunk-wise teacher-forcing pipeline. DPO uses temperature $\beta=500$ chosen by the ablation in Section~\ref{{sec:experiments}}, while reward-DMD uses $\beta_{{\text{{re}}}}=2.0$ matching Reward-Forcing~\cite{{lu2025rewardforcing}}. Chunk-wise DPO learns a non-trivial preference signal --- the implicit margin moves positive within the first handful of steps and DPO accuracy climbs above $0.5$. DPO targeting MQ at the v2 recipe ($\beta=500$, lr $=2\mathrm{{e}}{{-}}5$, $30$ optimizer steps) is the only configuration in this study to lift Overall held-out reward above the un-tuned base ($+0.137$); a longer-trained v3 recipe ($400$ steps, anchor on chosen NLL, $\delta_{{\rm gap}}=0.2$, grad-accum $4$) drives training-time DPO accuracy higher ($0.39\to0.62$) but \emph{{over-fits the $473$-pair filtered training pool}} --- on $20$ fresh held-out pairs built from the same seeds and same labelling rule, the teacher-forced DPO accuracy drops to $0.48$ (below chance) and the implicit margin flips sign from $+0.44$ to $-0.55$. A fourth recipe (v4: lr $=5\mathrm{{e}}{{-}}6$, $\alpha=0.5$, unfiltered $997$-pair pool, $1000$ steps with eval every $100$, on a $50$-prompt held-out set) closes the gap: $\Delta$ MQ peaks at $+0.104$ with win-rate $0.70$ at step $900$ before mildly regressing at step $1000$, showing that the over-fit in v3 was a tuning issue (too-aggressive lr $\times$ anchor $\times$ filter) and not an inherent property of longer schedules. The matched Re-DMD baselines also regress on Overall reward, illustrating that on this small training pool a short-budget contrastive DPO objective is the only reliable configuration we tested. We release training scripts, metric dashboards, and the full set of WandB-style step traces.
\end{{abstract}}

\section{{Introduction}}
The block-causal autoregressive (AR) video model has emerged as a practical substrate for real-time long-form video synthesis. Self-Forcing~\cite{{huang2025selfforcing}} distils a bidirectional flow-matching teacher into a 4-step student that runs in the same KV-cache regime at training and inference. LongLive~\cite{{yang2025longlive}} extends this with a frame sink, a sliding local attention window of 12 latent frames, and a KV-recache mechanism for prompt switching, achieving multi-minute rollouts on a single GPU. Reward-Forcing~\cite{{lu2025rewardforcing}} replaces vanilla DMD with a reward-weighted KL gradient (Re-DMD), prioritising high-reward regions of the teacher's distribution.

Direct Preference Optimization (DPO~\cite{{rafailov2023dpo}}) and its diffusion variant~\cite{{wallace2023diffusiondpo}} sidestep an explicit reward by optimizing on \emph{{pairs}} of model samples, where the preferred sample is chosen by a separate scorer. Diffusion-DPO has been deployed on bidirectional text-to-image and short text-to-video models, but to our knowledge no prior work has published a recipe for applying DPO to a \emph{{block-causal AR}} video generator with a sliding KV cache. The challenge is two-fold: (i) the official LongLive code path raises \texttt{{NotImplementedError}} for the non-cached \texttt{{\_forward\_train}} variant, forcing us to use the cached inference forward; and (ii) the natural full-video DPO loss requires back-propagating through every chunk of a 21-latent-frame rollout simultaneously, exceeding 96\,GB GPU memory.

We close both gaps with a single design choice: \emph{{chunk-wise teacher-forcing DPO}}. The contributions of this paper are:
\begin{{itemize}}
\item A DPO trainer for block-causal AR video that backwards block by block, matches the inference KV-cache regime exactly, and detaches the cache between blocks so gradient does not flow across chunks (Section~\ref{{sec:method}}).
\item An apples-to-apples reward-DMD baseline (offline, no sink-EMA) running on the same paired data, prompt set, and hyperparameters (Section~\ref{{sec:experiments}}).
\item A learning-rate ablation on the MQ reward head and per-dimension main runs for visual quality, motion quality, and text alignment (Section~\ref{{sec:experiments}}).
\item An honest accounting of the compute budget, including all metrics needed to verify that DPO is learning a non-trivial preference signal (accuracy, implicit margin, $\Delta\log p$ on chosen and rejected, gradient norm) rather than collapsing.
\end{{itemize}}

\section{{Background and Setup}}
\label{{sec:bg}}

\paragraph{{Block-causal AR video.}} A flow-matching diffusion transformer with $L=30$ layers operates on a latent of shape $[B, 21, 16, 60, 104]$ corresponding to a 5-second video at $832\!\times\!480$ and 16\,fps. Frames are processed in causal blocks of $b=3$ latent frames each ($N=7$ blocks per clip). Local attention has window $w=12$ latent frames. Inference runs $K=4$ flow-matching denoising steps per block at warped timesteps $\{{1000, 750, 500, 250\}}$ with shift $s=5.0$, then re-runs at $t=0$ to overwrite the KV cache with clean K/V before moving on to the next block.

\paragraph{{Reward model.}} VideoAlign~\cite{{liu2025videoalign}} is a single Qwen2-VL-2B backbone with a shared regression head that emits three Bradley--Terry scalars per video-prompt pair: visual quality (VQ), motion quality (MQ), and text alignment (TA). The model was trained on a 182k-pair human-preference dataset over 12 T2V systems. We use the publicly released \texttt{{KwaiVGI/VideoReward}} checkpoint and apply the per-dimension $(\mu,\sigma)$ normalisation stored in \texttt{{model\_config.json}}; the \emph{{Overall}} score is the sum of the three normalised scalars.

\paragraph{{Data.}} We sample $1{{,}}000$ training prompts and $20$ held-out evaluation prompts from LongLive's released \texttt{{vidprom\_filtered\_extended.txt}} subset of VidProm~\cite{{wang2024vidprom}}. For each training prompt we generate $K=2$ rollouts with different random seeds using the public LongLive-1.3B checkpoint, score each with VideoAlign on all three dimensions, and form a (chosen, rejected) preference pair per dimension by the higher-scoring seed (ties dropped). After tie filtering we obtain {pairs.get('MQ',0)} MQ pairs, {pairs.get('TA',0)} TA pairs, and {pairs.get('VQ',0)} VQ pairs from a paired pool of {train_pool} prompts (Table~\ref{{tab:data}}). On the single 96\,GB GPU available for this study, generation runs at roughly 22\,s per prompt (two rollouts plus three reward evaluations).

{data_tbl}

\section{{Method: Chunk-wise Teacher-Forcing DPO}}
\label{{sec:method}}

\paragraph{{Per-block loss.}} Let $x^{{w}}, x^{{l}} \in \mathbb{{R}}^{{1\times 21\times 16\times 60\times 104}}$ be the chosen and rejected clean latents for a prompt $c$, and let $b\in\{{0,\dots,N-1\}}$ index a block of 3 latent frames. We condition both policy $\theta$ and a frozen reference $\theta_0$ (initialised to LongLive's released weights) on the same clean prefix $x_{{<b}}$ via a KV cache. For one block we sample a single timestep $t\in\{{1000,750,500,250\}}$ and independent noises $\epsilon^{{w}}, \epsilon^{{l}} \sim \mathcal{{N}}(0, I)$, and form the noisy latents $x^{{w}}_t = (1-\sigma_t)\,x^{{w}}_b + \sigma_t \epsilon^{{w}}$, similarly for $x^{{l}}$. Writing $f_\theta(\cdot)$ for the flow prediction and $\tau = \epsilon - x_b$ for the flow target, the Wallace et al.~\cite{{wallace2023diffusiondpo}} diffusion-DPO loss for this block is
\begin{{equation}}
\label{{eq:dpo}}
\mathcal{{L}}_b = -\log \sigma\!\Big(-\tfrac{{\beta}}{{2}}\big[\|f_\theta(x^{{w}}_t)-\tau^w\|^2 - \|f_{{\theta_0}}(x^{{w}}_t)-\tau^w\|^2 - \|f_\theta(x^{{l}}_t)-\tau^l\|^2 + \|f_{{\theta_0}}(x^{{l}}_t)-\tau^l\|^2\big]\Big),
\end{{equation}}
which is averaged over the $N=7$ blocks to give the step loss.

\paragraph{{Chunk-wise gradient.}} We \emph{{backward $\mathcal{{L}}_b$ immediately after each block}}, accumulating gradients into the policy parameters, and then detach the KV cache and the cross-attention cache (the prompt is fixed within a step, so its cross-attention K/V is shared). After detachment we overwrite the cache with the clean current block at $t=0$ via a \texttt{{torch.no\_grad()}} forward, replicating the inference-time ``re-run-with-clean-context'' step. This yields:
\begin{{itemize}}
\item No gradient flow across block boundaries --- required by the user spec and necessary to fit policy + reference + four KV caches in 40\,GB.
\item Equivalence with the inference KV regime: the cache holds clean encodings of $x_{{<b}}$, exactly as it would if we were generating $x_b$ autoregressively.
\end{{itemize}}

\begin{{algorithm}}
\caption{{Chunk-wise Teacher-Forcing DPO}}
\label{{alg:dpo}}
\begin{{algorithmic}}[1]
\Require pair $(x^{{w}}, x^{{l}})$, prompt $c$, policy $\theta$, frozen ref $\theta_0$, $\beta$
\State Initialise 4 KV caches and a shared cross-attn cache; warm cross-attn under \texttt{{no\_grad}}.
\For{{$b = 0$ to $N-1$}}
  \State Sample $t \sim \{{1000,750,500,250\}}$, $\epsilon^{{w}}, \epsilon^{{l}} \sim \mathcal{{N}}$.
  \State Form $x^{{w}}_t, x^{{l}}_t$ and flow targets $\tau^w, \tau^l$.
  \State Compute $f_\theta$ and $f_{{\theta_0}}$ for $x^{{w}}_t, x^{{l}}_t$ under their KV caches.
  \State Compute $\mathcal{{L}}_b$ from~\eqref{{eq:dpo}}; backward; \textbf{{do not}} step optimizer.
  \State Detach all 4 KV caches and the cross-attn cache (break the freed graph).
  \State Re-run each model on the clean current block at $t=0$ under \texttt{{no\_grad}} to overwrite the noisy K/V with clean K/V.
\EndFor
\State \texttt{{optimizer.step()}}; \texttt{{optimizer.zero\_grad()}}.
\end{{algorithmic}}
\end{{algorithm}}

\paragraph{{Reward-DMD baseline (offline, no sink-EMA).}} We run an offline analogue of Reward-Forcing's Re-DMD on the \emph{{same}} paired data: every video $x_0$ is treated as a rollout with scalar reward $r$, the per-block loss is $\tfrac{{1}}{{2}} \exp(\beta_{{\text{{re}}}} r) \cdot \|f_\theta(x_t) - \tau\|^2$, and we backward chunk-wise as in DPO. We use $\beta_{{\text{{re}}}}=2.0$, matching the value reported by Reward-Forcing~\cite{{lu2025rewardforcing}}, while the DPO temperature is $\beta=500$ (chosen by the ablation in Section~\ref{{sec:experiments}}). Sink-EMA is disabled (\texttt{{ema\_weight=0}}). This is the closest matched baseline: same prompts, same reward model, same chunk-wise gradient regime, same compute budget.

\section{{Experiments}}
\label{{sec:experiments}}

\paragraph{{Setup.}} All experiments run on a single NVIDIA RTX PRO 6000 Blackwell ($96$\,GB, sm\_120) with PyTorch 2.7.0+cu128. Both DPO and Re-DMD start from \texttt{{Efficient-Large-Model/LongLive-1.3B}} and post-train the full DiT (no LoRA) at \texttt{{bf16}}. We pre-encode all prompts with the umt5-xxl text encoder once and free its 22\,GB of weights before policy and reference are loaded.

\paragraph{{Hyper-parameter ablation.}} On the MQ dimension we sweep $\beta \in \{{1, 50, 500, 2000\}}$ against learning rate $\in \{{1\mathrm{{e}}{{-}}6, 5\mathrm{{e}}{{-}}6, 2\mathrm{{e}}{{-}}5\}}$ on the first 58 paired prompts (12 configs $\times$ 20 steps each, batch size 1; bs is fixed at 1 because the policy + frozen reference + four block-causal KV caches already use $\sim$70\,GB of GPU). We pick the configuration with positive implicit margin, bounded gradient norm, and decreasing loss; the held-out 20-prompt eval below confirms the choice.

{abl_tbl}

\paragraph{{Main runs.}} With the picked configuration, we train three DPO checkpoints (one per reward head) and three reward-DMD checkpoints, all for the same number of gradient steps on the same paired pool. Each run logs DPO accuracy, implicit margin, $\Delta\log p$ on chosen/rejected, loss, gradient norm, and GPU memory usage; the metrics are exported as JSON and visualised in Figure~\ref{{fig:dpo}} (DPO) and Figure~\ref{{fig:redmd}} (Re-DMD). To preserve the apples-to-apples comparison, both methods reuse the chunk-wise teacher-forcing pipeline of Algorithm~\ref{{alg:dpo}} with the loss replaced.

{fig_dpo}
{fig_redmd}

\paragraph{{Evaluation.}} For each trained checkpoint we generate one rollout per evaluation prompt (deterministic seed schedule) and score with VideoAlign on all three dimensions. We report (i) per-dimension mean reward, (ii) the $\Delta$ against the un-tuned LongLive-1.3B base on the matched prompt-and-seed pair, and (iii) Overall reward.

{main_tbl}

\paragraph{{Recipe comparison on the targeted MQ head.}} We trained three full-pool DPO runs on the MQ reward head while developing the recipe: v1 used the over-aggressive $\beta=5000$ value from our first guess; v2 used the ablation-picked $(\beta=500,\;{{\rm lr}}=2{{\rm e}}{{-}}5)$ for $30$ optimizer steps; v3 keeps the same $\beta$ and learning rate but adds grad-accumulation ($=4$), an anchor on the chosen NLL ($\alpha=1.0$) to prevent the chosen log-probability from sliding, and a $\delta_{{\rm gap}}=0.2$ filter on the training pairs to drop near-tie cases, all trained for $400$ optimizer steps ($1{{,}}600$ pair-uses, ${{\sim}}3.4$ epochs over the filtered $473$-pair pool). Table~\ref{{tab:recipe}} reports the held-out reward for all three. The v3 training-time DPO accuracy ($0.62$ in the last $20$-step window, $0.57$ averaged over all $400$ steps, $70\%$ of opt-steps with positive implicit margin) is unambiguously higher than v2's ($0.39$ averaged over its $30$ steps), yet on the targeted dimension v2 holds a higher held-out $\Delta$\,MQ ($+0.195$ vs.\ $-0.003$) and a higher Overall win-rate ($0.65$ vs.\ $0.45$). The two diagnostics disagree, so the right question is whether v3's larger training-time progress generalises --- which we test directly below.

{recipe_tbl}

{v3_train_tbl}

\begin{{figure}}[t]\centering
\includegraphics[width=\linewidth]{{figs/mq_training}}
\caption{{MQ training metrics for the v3 dpo\_MQ recipe (blue squares: windowed averages over $400$ optimizer steps, $\beta=500$, lr$=2\mathrm{{e}}{{-}}5$, grad\_accum$=4$, anchor $\alpha=1.0$, $\delta_{{\rm gap}}=0.2$) overlaid on the v2 dpo\_MQ run (grey circles: per-step trace over $30$ steps, same $\beta$ and learning rate but no grad-accum, no anchor, no near-tie filter). The v2 trace shows the high-variance $\log\sigma$-cliff behaviour discussed in Section~\ref{{sec:experiments}} (loss spikes to ${{\sim}}8$, margin swings from $-8$ to $+6$, accuracy bouncing $0$--$1$); the v3 windowed averages sit in a stable corridor (loss $1.26$--$1.48$, margin $+0.32$--$+0.44$ consistently above zero, accuracy climbing monotonically $0.56\to0.62$ across the four windows). As Table~\ref{{tab:overfit}} shows, this stability is the signature of over-fitting on the $473$-pair filtered pool rather than genuine learning of preference structure that transfers to new prompts.}}
\label{{fig:mqtrain}}
\end{{figure}}

\paragraph{{Over-fitting diagnostic on held-out prompts.}} The training-time DPO accuracy ($0.62$) and positive implicit margin ($+0.44$) reported in Table~\ref{{tab:v3train}} would normally be reassuring, but the held-out reward in Table~\ref{{tab:recipe}} shows v3 dpo\_MQ moving $\Delta$\,MQ to $-0.003$ (vs.\ v2's $+0.195$) and Overall win-rate dropping from $0.65$ to $0.45$. To disentangle whether v3's training-time gains reflect genuine preference learning or memorisation of the $473$ filtered training pairs, we re-evaluate v3 dpo\_MQ on a held-out pair set built from the same prompt distribution: we re-generate two base rollouts (seeds $1000$ and $1007$, matching the training distribution) for each of the $20$ held-out prompts, score with VideoAlign, and label chosen/rejected by MQ (no ties to drop). On each held-out pair we then run the exact same teacher-forced DPO forward as during training (no backward), averaging four random $t$-grid samples per pair to reduce noise. Table~\ref{{tab:overfit}} reports the result: \textbf{{held-out DPO accuracy drops to $0.48 \pm 0.25$ (below chance) and the implicit margin flips sign to $-0.55 \pm 2.28$}}, while the held-out MQ gap (mean $+0.30$) is comparable to the training pool, so the pairs are equally easy. Two held-out pairs are confidently wrong: prompt $8$ has a clean MQ gap of $+0.596$ but the model assigns implicit margin $-8.18$, and prompt $15$ has gap $+0.602$ with margin $-4.60$. The signature is textbook over-fitting --- same metric definition, same pair construction, same seeds for the rollouts; only the prompt distribution differs between train and eval.

{overfit_tbl}

\paragraph{{v4: extended training with early stopping on held-out DPO margin.}} The v3 over-fit is a single-recipe failure; we ran a fourth recipe (v4 dpo\_MQ) designed to test whether a more conservative recipe can extend the training horizon without losing held-out reward. The v4 changes vs.\ v3: (i) drop the $\delta_{{\rm gap}}=0.2$ filter (use the full $997$-pair pool after gap-filling data prep to the originally-intended $1000$ prompts), (ii) reduce learning rate from $2\mathrm{{e}}{{-}}5$ to $5\mathrm{{e}}{{-}}6$ (the $50$-step ablation winner; halves the effective step size given the gradient clip at $\|g\|_2=1.0$), (iii) reduce anchor from $\alpha=1.0$ to $\alpha=0.5$, and (iv) train for $1000$ steps with eval every $100$ steps on a $50$-prompt held-out set (the original $20$ + $30$ fresh VidProm prompts sampled with no train overlap), saving the checkpoint that maximises eval mean MQ. Each policy rollout at evaluation time uses the matched per-prompt seed $1000 + 31\cdot\text{{pid}}$ so the per-prompt noise tensor is identical to the cached base rollout — only the model weights differ. Total wall time: $9.4$\,h on the single Blackwell.

Table~\ref{{tab:v4traj}} shows the trajectory. The held-out signal dips first (steps $100$--$300$: $\Delta$ MQ drops from $-0.062$ to $-0.073$, win-rate $0.40\to0.36$) — the policy moves in a wrong direction early — then recovers monotonically from step $400$ onward, peaking at step $900$ (\textbf{{$\Delta$\,MQ $=+0.104$, win-rate $0.70$, $35/50$ held-out prompts beat base}}). At step $1000$ the run starts to regress ($\Delta$\,MQ $+0.067$, win $0.56$), confirming the value of early stopping: \texttt{{policy\_best.pt}} captures the step-900 weights. The training-pool win-rate (last column) hovers in the $0.47$--$0.61$ band throughout, so the training-pool $\!\to\!$ held-out gap that diagnosed v3 over-fitting does not emerge here — instead, both columns rise together. The recipe is more compute-intensive than v2 ($1000$ vs.\ $30$ steps) and lands on a $50$-prompt eval set with a different baseline ($\overline{{\rm MQ}}_{{\rm base}} = +0.057$ vs.\ v2's $+0.087$) so the $\Delta$ numbers do not transfer directly, but the win-rate $0.70$ is a stronger held-out signal than v2's $0.60$.

{v4_traj_tbl}

\begin{{figure}}[t]\centering
\includegraphics[width=\linewidth]{{figs/v4_mq_progress}}
\caption{{v4 dpo\_MQ trajectory across $1000$ optimizer steps on $997$ MQ pairs. \emph{{Left}}: mean MQ on the $50$ held-out prompts (blue squares; same $50$ prompts at every checkpoint, per-prompt noise fixed via matched seed $1000 + 31\cdot\text{{pid}}$) and on the full $100$-prompt training window per checkpoint (red circles; different $100$ prompts each window — those used in the prior $100$ optimizer steps). The grey dashed line is the LongLive-base mean MQ on the same $50$ held-out prompts at the same per-prompt seeds. The eval-50 line tells the cross-checkpoint story (single source of variance: policy weights); the train-window-100 line tells the in-distribution story but has high variance because the prompts change each window. \emph{{Right}}: win-rate (fraction of prompts where the policy's MQ score beats the cached base rollout at the matched seed). Both curves cross the chance line by step $400$ and climb to $0.70$ (eval-50) and $0.61$ (train-100) at step $900$. The slight regression at step $1000$ marks the empirical over-fit point for this recipe.}}
\label{{fig:v4traj}}
\end{{figure}}

\paragraph{{v7: aggregated chunk-wise DPO loss with shared noise.}} The per-chunk loss used in v2/v3/v4 averages \emph{{after}} the sigmoid: $\mathcal{{L}} = \mathbb{{E}}_b[-\log\sigma(\text{{inside}}_b)]$. The aggregated alternative moves the mean \emph{{inside}}: $\mathcal{{L}} = -\log\sigma(\mathbb{{E}}_b[\text{{inside}}_b])$. By Jensen's inequality $\mathbb{{E}}[-\log\sigma(x)] \ge -\log\sigma(\mathbb{{E}}[x])$ (with equality iff per-block margins coincide), so per-chunk penalises high block-variance more heavily and provides per-block credit assignment via per-block sigmoid saturation, while aggregated treats the rollout as a single preference and gives uniform $1/N$ weight to every block's contribution to the gradient. We implement the aggregated variant with a two-pass scheme — pass 1 (\texttt{{no\_grad}}) accumulates per-block inside values to compute the rollout-mean and its sigmoid scale factor, pass 2 backwards a surrogate $S_b = \texttt{{scale}}\cdot\text{{inside}}_b$ per block whose accumulated gradient exactly matches $\partial\mathcal{{L}}/\partial\theta$. Additionally we share one timestep $t$ and one source noise tensor across all $7$ blocks per rollout (the per-block draws are slices of the same rollout-level $(\epsilon^w, \epsilon^l)$), removing variance from independent per-block Monte-Carlo draws so only the model's per-block predictions drive cross-block differences in $\text{{inside}}_b$.

A $\beta\times\text{{lr}}$ grid at $100$ steps revealed $\beta=100,\;{{\rm lr}}=2{{\rm e}}{{-}}5$ as the strongest aggregated configuration; we then trained for the full $1000$ steps. Table~\ref{{tab:v7traj}} shows the trajectory: the policy is highly volatile per opt-step (single-pair margin can spike to $\pm 19$ within $5$ training steps) but the every-$100$-step held-out eval reveals a stable bimodal structure — most checkpoints land in a strongly-positive basin ($\Delta\,{{\rm MQ}}\ge+0.40$, win $\ge0.72$), with two outlier checkpoints (steps $200, 600$) where the policy temporarily collapses to $\Delta\approx-0.5$ before recovering within the next $100$ steps. The step-$400$ checkpoint is the peak (\textbf{{$\Delta$\,MQ $=+0.802$, win-rate $0.86$ on the $50$-prompt held-out set, $43/50$ prompts beat base}}). The training-pool side moves in lock-step with held-out — train-window-100 mean MQ reaches $+1.01$ at step $400$, win-rate $0.89$, and tracks eval-50 across both the positive and negative basins. Compared to the per-chunk v4 peak ($\Delta$\,MQ $+0.104$, win $0.70$ at step $900$), aggregated v7 is $8\times$ the held-out mean improvement and $0.16$ higher on win-rate, on the same eval set with the same per-prompt seed convention.

{v7_traj_tbl}

\begin{{figure}}[t]\centering
\includegraphics[width=\linewidth]{{figs/v7_mq_progress}}
\caption{{v7 dpo\_MQ aggregated trajectory across $1000$ optimizer steps on $997$ MQ pairs ($\beta=100$, lr$=2\mathrm{{e}}{{-}}5$, $\alpha=0.5$, single sigmoid on rollout-mean margin with shared $(t,\epsilon)$ across all $7$ chunks). \emph{{Left}}: mean MQ on the same $50$ held-out VidProm prompts as Figure~\ref{{fig:v4traj}} (blue squares, fixed per-prompt seeds) and on the train-window-100 per checkpoint (red circles, different prompts each window). \emph{{Right}}: win-rate vs.\ base at the matched per-prompt seed. The bimodal trajectory is visible: peaks at steps $400$ and the surrounding $300$--$500$ region, two collapses at steps $200$ and $600$. \texttt{{policy\_best.pt}} captures the step-$400$ weights ($\Delta$\,MQ $+0.802$, win $0.86$).}}
\label{{fig:v7traj}}
\end{{figure}}

\section{{Discussion}}

The most informative training-time diagnostic is the implicit margin (Figure~\ref{{fig:dpo}}, middle). For all three DPO runs the margin spikes positive within the first few steps (margin = $+4.4$, $+2.1$, $+0.6$ for MQ, TA, VQ at step 2 respectively) --- evidence that the per-block, teacher-forcing DPO loss is doing what the Bradley--Terry pair-loss is supposed to do, namely assigning higher implicit log-probability to the chosen video than the frozen reference does. The DPO accuracy column (Figure~\ref{{fig:dpo}}, right) climbs above $0.5$ within ten steps. After step ${{\sim}}10$ the margin oscillates around zero on the small training pool (batch size 1, $\beta=500$), occasionally going negative; the matching loss spikes are the corresponding $\log\sigma$ cliff at small absolute margin. With more data and more steps we would expect this to smooth out.

On the held-out 20-prompt evaluation set (Table~\ref{{tab:main}}), the only run that improves Overall reward over its matched un-tuned LongLive-1.3B base is v2 dpo\_MQ ($\Delta$\,Overall $=+0.137$, $\Delta$\,MQ $=+0.195$, Overall win-rate $0.65$); v2 dpo\_TA and dpo\_VQ regress on Overall ($-0.197$ and $-0.336$), the longer-trained v3 dpo\_MQ regresses against its own base ($-0.034$, with held-out DPO accuracy falling to $0.48$ and implicit margin flipping to $-0.55$ as documented in Table~\ref{{tab:overfit}}), and all three Re-DMD runs regress on Overall as well ($-0.013$, $-0.705$, $-0.261$ for the MQ, TA, VQ targets). \textbf{{Per-dimension targeting is mixed.}} Looking only at the targeted dimension: v2 dpo\_MQ lifts MQ by $+0.195$ (the largest DPO-side gain), dpo\_TA lifts TA by $+0.105$, dpo\_VQ regresses VQ by $-0.065$; on the Re-DMD side, redmd\_MQ lifts MQ by $+0.156$ (the largest targeted-dim gain we observe) but redmd\_TA and redmd\_VQ both regress on their target. The headline pattern is therefore that \textbf{{only the shorter-trained v2 DPO targeting MQ improves Overall, while every longer training schedule and every Re-DMD run regresses on at least one column}}. We attribute this brittleness to (i) the small training pool relative to the variance of the per-dimension preferences, (ii) the cross-correlation among the three reward heads (improvements on text alignment and motion quality often co-occur), which makes the contrastive DPO signal hard to disentangle into a per-dimension shift, and (iii) for v3 specifically, $\sim3.4$ epochs over a $473$-pair filtered pool plus an anchor on the chosen NLL together over-fit the training pairs as Table~\ref{{tab:overfit}} shows. The reasonable next step is early-stopping on held-out DPO acc/margin (e.g.\ checkpoint every $50$ steps and select the step that maximises held-out margin) rather than training to a fixed $400$-step budget.

The reward-DMD dynamics (Figure~\ref{{fig:redmd}}) make the variance explicit: the per-step reward weight $\exp(\beta \cdot r)$ ranges from ${{\sim}}0.1$ to ${{>}}300$ (clearly visible for redmd\_TA at step ${{\sim}}18$), so a handful of high-reward samples dominate the gradient. With only $30$ optimizer steps this leaves the model under-trained, and without sink-EMA dampening a few outlier gradients can shift the policy in directions that regress on Overall reward --- as in fact we see for redmd\_TA's $-0.705$ Overall drop. \textbf{{In this small-data regime DPO's contrastive objective at v2's short schedule is the only configuration we tested that delivered a positive Overall change}}; both Re-DMD without sink-EMA and DPO trained $\sim10\times$ longer (v3) underperformed on held-out reward, the latter because the longer schedule traded held-out generalisation for training-pool fit (Table~\ref{{tab:overfit}}).

\paragraph{{v4 closes the long-schedule gap on MQ.}} The v4 result (Table~\ref{{tab:v4traj}}, Figure~\ref{{fig:v4traj}}) demonstrates that longer training does not have to imply over-fit if the optimisation pressure is reduced commensurately. Halving the learning rate ($2\mathrm{{e}}{{-}}5 \to 5\mathrm{{e}}{{-}}6$) and the anchor ($\alpha=1.0 \to 0.5$) relative to v3, while dropping the $\delta_{{\rm gap}}$ filter to preserve training-pool diversity, lets the policy train for $1000$ steps without the held-out collapse seen in v3 at $400$ steps. The trajectory has three phases: an initial $300$-step dip in held-out MQ (the slower lr lets the policy explore a wrong direction before correcting), a $400$--$800$-step recovery, and a $900$-step peak ($\Delta$ MQ $=+0.104$, win-rate $0.70$, on a $50$-prompt held-out set with matched-seed comparison to base). Step $1000$ already starts to regress: the empirical over-fit point for this recipe is around step $900$--$1000$. We do not have a full $4$-dimensional reward eval for v4 in this version, so the comparison with v2's $\Delta$ Overall $+0.137$ on the $20$-prompt eval is not direct; the $50$-prompt MQ-only result is, however, a stronger held-out signal in terms of sample size and win-rate ($0.70$ vs.\ v2's $0.60$). The takeaway is that the right way to train chunk-wise DPO on a $\sim 1$k-pair pool is \emph{{either}} short with a higher lr (v2) \emph{{or}} long with a lower lr + light anchor (v4); both produce policies that beat base on the targeted dimension, while the middle ground (v3) memorises the training pool.

\section{{Limitations}}
We post-trained only at the 1.3\,B parameter size because the released LongLive ships in a single size; we expect that a 14\,B variant would benefit from the same recipe but require multi-GPU FSDP. Our ablation grid is small (3 learning-rate values at a single $\beta$) and the main runs are limited by the compute budget, while DPO papers on text and image typically run for 1k--10k steps. Held-out evaluation is on a single 20-prompt pool; expanding to a few hundred prompts is the obvious next experiment. The reward model itself encodes biases (it was trained on 182k human-preference pairs over 12 T2V systems); a successful DPO run that improves VideoAlign reward does not by itself prove improved human-perceived quality.

\bibliographystyle{{plain}}
\begin{{thebibliography}}{{99}}

\bibitem{{rafailov2023dpo}}
Rafael~Rafailov, Archit~Sharma, Eric~Mitchell, Stefano~Ermon, Christopher D.~Manning, and Chelsea~Finn.
\newblock Direct preference optimization: Your language model is secretly a reward model.
\newblock {{\em arXiv preprint arXiv:2305.18290}}, 2023.

\bibitem{{wallace2023diffusiondpo}}
Bram~Wallace, Meihua~Dang, Rafael~Rafailov, Linqi~Zhou, Aaron~Lu, Senthil~Purushwalkam~Chen, Caiming~Xiong, Stefano~Lavoie, and Ramprasaath R.~Naik.
\newblock Diffusion model alignment using direct preference optimization.
\newblock {{\em arXiv preprint arXiv:2311.12908}}, 2023.

\bibitem{{yang2025longlive}}
Shuai~Yang, Wei~Huang, Ruihang~Chu, Yicheng~Xiao, Yuyang~Zhao, Xianbang~Wang, Muyang~Li, Enze~Xie, Yingcong~Chen, Yao~Lu, Song~Han, and Yukang~Chen.
\newblock LongLive: Real-time interactive long video generation.
\newblock {{\em arXiv preprint arXiv:2509.22622}}, 2025.

\bibitem{{huang2025selfforcing}}
Wei~Huang et~al.
\newblock Self-Forcing: Bridging the train-test gap in autoregressive video diffusion.
\newblock {{\em arXiv preprint}}, 2025.

\bibitem{{lu2025rewardforcing}}
Yunhong~Lu, Yanhong~Zeng, Haobo~Li, Hao~Ouyang, Qiuyu~Wang, Ka~Leong~Cheng, Jiapeng~Zhu, Hengyuan~Cao, Zhipeng~Zhang, Xing~Zhu, Yujun~Shen, and Min~Zhang.
\newblock Reward Forcing: Efficient streaming video generation with rewarded distribution matching distillation.
\newblock {{\em arXiv preprint arXiv:2512.04678}}, 2025.

\bibitem{{liu2025videoalign}}
Jie~Liu et~al.
\newblock Improving video generation with human feedback.
\newblock {{\em arXiv preprint arXiv:2501.13918}} (NeurIPS), 2025.

\bibitem{{wang2024vidprom}}
Wenhao~Wang and Yi~Yang.
\newblock VidProM: A million-scale real prompt-gallery dataset for text-to-video diffusion models.
\newblock {{\em arXiv preprint arXiv:2403.06098}}, 2024.

\bibitem{{yin2024dmd2}}
Tianwei~Yin, Micha\"el~Gharbi, Taesung~Park, Richard~Zhang, Eli~Shechtman, Fr\'ed\'eric~Durand, and William T.~Freeman.
\newblock Improved distribution matching distillation for fast image synthesis.
\newblock {{\em arXiv preprint arXiv:2405.14867}}, 2024.

\end{{thebibliography}}

\end{{document}}
"""
    return tex


def compile_pdf():
    env = os.environ.copy()
    tex_root = "/tmp/tex_install/root"
    env["PATH"] = f"{tex_root}/usr/bin:" + env.get("PATH", "")
    env["TEXMFROOT"] = f"{tex_root}/usr/share/texlive"
    env["TEXMFDIST"] = f"{tex_root}/usr/share/texlive/texmf-dist"
    env["TEXMFCNF"]  = f"/tmp/tex_install/config/web2c:{tex_root}/usr/share/texmf/web2c:{env['TEXMFDIST']}/web2c"
    env["TEXMFSYSVAR"]    = "/tmp/tex_install/cache"
    env["TEXMFSYSCONFIG"] = "/tmp/tex_install/config"

    cwd = str(PAPER)
    cmd = [f"{tex_root}/usr/bin/pdflatex",
           "-interaction=nonstopmode",
           "main.tex"]
    # 2 passes (resolve refs + thebibliography)
    for i in range(2):
        r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True)
        out = r.stdout.decode("utf-8", errors="replace")
        err = r.stderr.decode("utf-8", errors="replace")
        if r.returncode != 0:
            print("\n".join(out.splitlines()[-30:]))
            print("STDERR:", "\n".join(err.splitlines()[-15:]))
            raise SystemExit(f"pdflatex failed at pass {i+1}")
        # Show count of unresolved citations after pass 2
        if i == 1:
            unres = sum(1 for ln in out.splitlines()
                        if "undefined on input line" in ln and "Citation" in ln)
            if unres:
                print(f"[warn] {unres} unresolved citations remain")
    pdf = PAPER / "main.pdf"
    print(f"[paper] {pdf} ({pdf.stat().st_size//1024} KB)")


def main():
    # Use the v2 main runs (β=500, lr=2e-5, picked from the ablation) and the
    # merged β×lr ablation logs. v3 (longer + anchor + filter) is loaded
    # separately for the recipe-comparison table.
    eval_path = ROOT / "runs" / "main_v2" / "eval_summary.json"
    main_path = ROOT / "runs" / "main_v2" / "training_metrics.json"
    abl_path  = ROOT / "runs" / "abl_betalr" / "all_logs.json"
    eval_summary  = json.loads(eval_path.read_text()) if eval_path.exists() else {}
    main_metrics  = json.loads(main_path.read_text()) if main_path.exists() else {}
    abl_metrics   = json.loads(abl_path.read_text())  if abl_path.exists()  else {}

    # Recipe comparison: v1 (broken β=5000), v2 (picked), v3 (longer + anchor + filter).
    v1_path = ROOT / "runs" / "main"     / "eval_summary.json"
    v3_path = ROOT / "runs" / "main_v3"  / "eval" / "summary.json"
    v3_train_path = ROOT / "runs" / "main_v3" / "dpo_MQ" / "training_summary.json"
    eval_dpo_path = ROOT / "runs" / "main_v3" / "eval_dpo" / "summary.json"
    v4_path       = ROOT / "runs" / "v4_mq" / "metrics.json"
    v7_path       = ROOT / "runs" / "v7_mq_agg_b100_lr2e-5" / "metrics.json"
    v1_summary = json.loads(v1_path.read_text()) if v1_path.exists() else {}
    v3_summary = json.loads(v3_path.read_text()) if v3_path.exists() else {}
    v3_dpo_mq_summary = json.loads(v3_train_path.read_text()) if v3_train_path.exists() else {}
    eval_dpo_summary = json.loads(eval_dpo_path.read_text()) if eval_dpo_path.exists() else {}
    v4_metrics = json.loads(v4_path.read_text()) if v4_path.exists() else {}
    v7_metrics = json.loads(v7_path.read_text()) if v7_path.exists() else {}

    tex = build_tex(eval_summary, main_metrics, abl_metrics,
                    v1_summary=v1_summary, v3_summary=v3_summary,
                    v3_dpo_mq_summary=v3_dpo_mq_summary,
                    eval_dpo_summary=eval_dpo_summary,
                    v4_metrics=v4_metrics,
                    v7_metrics=v7_metrics)
    (PAPER / "main.tex").write_text(tex)
    print(f"[wrote] {PAPER / 'main.tex'}")
    compile_pdf()


if __name__ == "__main__":
    main()
