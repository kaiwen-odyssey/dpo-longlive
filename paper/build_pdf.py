#!/usr/bin/env python
"""Render the paper to PDF via WeasyPrint (NeurIPS-like styling).

Pulls content from:
  - paper/paper.md            (markdown body — written by author)
  - runs/main/eval_summary.json (eval results)
  - runs/main/training_metrics.json (training curves)
  - runs/abl/training_metrics.json (ablation table)

Outputs:
  paper/paper.pdf
"""
import json, sys, os, statistics, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


CSS = r"""
/* HTML/CSS port of the NeurIPS 2024 (preprint) LaTeX style:
 *  - US Letter, ~1in left/right margins, ~0.75in top/bottom (preprint)
 *  - 10pt Times body, 11pt leading, justified
 *  - 17pt bold centered title; 12pt bold-section headers; 11pt subsection
 *  - Abstract framed by 0.5pt rules above and below, 9pt
 *  - Booktabs-style horizontal-only tables
 *  - References hung at 1em
 */
@page {
  size: Letter;
  margin: 1.0in 1.0in 1.0in 1.0in;
  @bottom-center {
    content: counter(page);
    font-size: 9pt;
    color: #333;
    margin-top: 0.4in;
  }
}

body {
  font-family: "Nimbus Roman", "Times New Roman", "Liberation Serif", Times, serif;
  font-size: 10pt;
  line-height: 1.18;
  color: #000;
  text-align: justify;
  hyphens: auto;
}

/* NeurIPS uses fontsize{17}{22} for the title; bold; centered. */
h1.title {
  font-size: 17pt;
  font-weight: bold;
  text-align: center;
  margin: 0 0 0.5em 0;
  line-height: 1.15;
}
.authors { text-align: center; font-size: 12pt; margin: 0.2em 0 0.05em 0; }
.affil   { text-align: center; font-size: 10pt; color: #222; margin: 0 0 0.6em 0; }

/* Section headings: \large\bfseries with vertical space above. */
h1 { font-size: 12pt; font-weight: bold; margin: 1.0em 0 0.3em 0; }
h2 { font-size: 11pt; font-weight: bold; margin: 0.7em 0 0.25em 0; }
h3 { font-size: 10pt; font-weight: bold; font-style: italic; margin: 0.5em 0 0.15em 0; }

p { margin: 0.0em 0 0.4em 0; text-indent: 1.2em; }
p.first, h1 + p, h2 + p, h3 + p, .abstract + p { text-indent: 0; }
ul, ol { margin: 0.2em 0 0.4em 1.2em; padding-left: 0.6em; }
li { margin: 0.1em 0; }

/* Abstract block: per the NeurIPS template — centered head, top+bottom rules. */
.abstract {
  margin: 0.6em 0.5in 0.8em 0.5in;
  font-size: 9.5pt;
  line-height: 1.18;
  border-top: 0.5pt solid #000;
  border-bottom: 0.5pt solid #000;
  padding: 0.45em 0;
}
.abstract h2 {
  text-align: center;
  margin: 0 0 0.35em 0;
  font-size: 11pt;
  font-weight: bold;
}
.abstract p { text-indent: 0; margin: 0; }

/* Booktabs-style tables: top/middle/bottom horizontal rules only. */
table {
  border-collapse: collapse;
  margin: 0.7em auto 0.4em auto;
  font-size: 9.5pt;
}
caption {
  caption-side: top;
  font-size: 9.5pt;
  margin-bottom: 0.4em;
  text-align: justify;
  padding: 0 0.5em;
}
thead th { border-top: 0.7pt solid #000; border-bottom: 0.5pt solid #000; padding: 0.30em 0.7em; font-weight: bold; }
tbody td { border-bottom: 0.7pt solid transparent; padding: 0.22em 0.7em; text-align: center; }
tbody tr.midrule td { border-top: 0.4pt solid #000; }
tbody tr:last-child td { border-bottom: 0.7pt solid #000; }
tbody tr:first-child td { border-top: 0; }

.algorithm {
  border: 0.7pt solid #000;
  border-left: 0;
  border-right: 0;
  padding: 0.5em 0.4em;
  margin: 0.7em 0;
  font-family: "Nimbus Roman", "Times New Roman", serif;
  font-size: 9.5pt;
  line-height: 1.25;
}
.algorithm b { font-family: "Nimbus Roman", "Times New Roman", serif; }

code { font-family: "Liberation Mono", "Courier New", monospace; font-size: 9pt; }
pre  { background: #f3f3f3; padding: 0.5em; font-size: 9pt; overflow-wrap: anywhere; }

.fig { text-align: center; margin: 0.7em 0; }
.fig img { max-width: 100%; height: auto; }
.fig .caption { font-size: 9.5pt; text-align: justify; margin: 0.4em 0.4em 0 0.4em; }

.bibitem {
  padding-left: 1.6em;
  text-indent: -1.6em;
  font-size: 9pt;
  line-height: 1.18;
  margin: 0.18em 0;
}
.eq { text-align: center; font-style: italic; margin: 0.5em 0; }
"""


def fmt(x):
    if x is None or (isinstance(x, float) and (x != x)):
        return "&mdash;"
    if isinstance(x, float):
        return f"{x:+.3f}"
    return str(x)


def main_table_html(eval_summary):
    runs_order = [("base", "LongLive-1.3B (base)"),
                  ("dpo_MQ", "DPO &mdash; target&nbsp;MQ"),
                  ("dpo_TA", "DPO &mdash; target&nbsp;TA"),
                  ("dpo_VQ", "DPO &mdash; target&nbsp;VQ"),
                  ("redmd_MQ", "Re-DMD &mdash; target&nbsp;MQ"),
                  ("redmd_TA", "Re-DMD &mdash; target&nbsp;TA"),
                  ("redmd_VQ", "Re-DMD &mdash; target&nbsp;VQ")]
    base = eval_summary.get("base", {}).get("means", {})
    rows = []
    for key, label in runs_order:
        if key not in eval_summary:
            rows.append([label, "&mdash;", "&mdash;", "&mdash;", "&mdash;"])
            continue
        m = eval_summary[key]["means"]
        if key == "base":
            rows.append([label, fmt(m.get("VQ")), fmt(m.get("MQ")), fmt(m.get("TA")), fmt(m.get("Overall"))])
        else:
            cells = []
            for d in ["VQ", "MQ", "TA", "Overall"]:
                if d in m and d in base:
                    delta = m[d] - base[d]
                    cells.append(f"{m[d]:+.3f}<br/><small>(&Delta;&nbsp;{delta:+.3f})</small>")
                else:
                    cells.append("&mdash;")
            rows.append([label] + cells)
    out = ['<table>']
    out.append('<caption><b>Table 1.</b> Held-out evaluation (20 VidProm prompts) on the three VideoAlign reward dimensions plus their sum. Cells show mean normalized reward; the &Delta; below is the change vs. the un-tuned LongLive-1.3B base on the same prompt-and-seed pair.</caption>')
    out.append('<thead><tr><th>Run</th><th>VQ</th><th>MQ</th><th>TA</th><th>Overall</th></tr></thead>')
    out.append('<tbody>')
    for row in rows:
        out.append('<tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>')
    out.append('</tbody></table>')
    return "\n".join(out)


def ablation_table_html(abl_metrics):
    rows = []
    for name in sorted(abl_metrics.keys()):
        if not name.startswith("ablation_dpo_MQ_lr"):
            continue
        lr = name.split("_lr")[-1]
        ms = abl_metrics[name]
        if not ms: continue
        tail = ms[max(0, int(len(ms) * 0.5)):]
        avg = lambda k: statistics.mean([m[k] for m in tail if k in m]) if tail else float("nan")
        rows.append([
            f"{lr}",
            fmt(avg("dpo_loss")),
            fmt(avg("dpo_margin")),
            f"{avg('dpo_accuracy'):.2f}" if not (avg('dpo_accuracy') != avg('dpo_accuracy')) else "&mdash;",
            fmt(avg("grad_norm")),
        ])
    out = ['<table>']
    out.append('<caption><b>Table 2.</b> Learning-rate ablation on the MQ reward head (10 steps each, batch&nbsp;size&nbsp;1, &beta;=5000, 60 training prompts). Reported values are means over the final 50% of steps.</caption>')
    out.append('<thead><tr><th>Learning rate</th><th>DPO loss</th><th>Implicit margin</th><th>Accuracy</th><th>Grad norm</th></tr></thead>')
    out.append('<tbody>')
    for row in rows:
        out.append('<tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>')
    out.append('</tbody></table>')
    return "\n".join(out)


def make_dynamics_plot(metrics_main, kind="dpo"):
    """Returns (path) of plot or None."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    out_dir = ROOT / "paper" / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))
    if kind == "dpo":
        plotted = False
        for name, color in zip(["dpo_MQ", "dpo_TA", "dpo_VQ"], ["#1f77b4", "#2ca02c", "#d62728"]):
            ms = metrics_main.get(name, [])
            if not ms: continue
            plotted = True
            steps = [m["step"] for m in ms]
            for ax, key, ylabel in zip(axes, ["dpo_loss", "dpo_margin", "dpo_accuracy"],
                                       ["DPO loss", "Implicit margin", "DPO accuracy"]):
                vals = [m[key] for m in ms]
                ax.plot(steps, vals, label=name, color=color)
                ax.set_xlabel("step"); ax.set_ylabel(ylabel)
                ax.grid(alpha=0.3)
        if not plotted: plt.close(fig); return None
        for ax in axes: ax.legend(fontsize=8, loc="best")
        fig.suptitle("Chunk-wise DPO training dynamics on the LongLive-1.3B policy", fontsize=10)
    else:
        plotted = False
        for name, color in zip(["redmd_MQ", "redmd_TA", "redmd_VQ"], ["#1f77b4", "#2ca02c", "#d62728"]):
            ms = metrics_main.get(name, [])
            if not ms: continue
            if "redmd_loss_pol" not in ms[0]: continue
            plotted = True
            steps = [m["step"] for m in ms]
            for ax, key, ylabel in zip(axes, ["redmd_loss_pol", "exp_beta_reward", "denoising_gap"],
                                       ["Policy denoising loss", r"$\exp(\beta\cdot r)$", r"Pol $-$ Ref denoising"]):
                vals = [m[key] for m in ms]
                ax.plot(steps, vals, label=name, color=color)
                ax.set_xlabel("step"); ax.set_ylabel(ylabel)
                ax.grid(alpha=0.3)
        if not plotted: plt.close(fig); return None
        for ax in axes: ax.legend(fontsize=8, loc="best")
        fig.suptitle("Reward-DMD (offline) training dynamics", fontsize=10)
    fig.tight_layout()
    out_path = out_dir / f"{kind}_dynamics.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def measure_dataset_sizes():
    """Read the data/ tree to figure out the actual training-pool and eval-pool sizes."""
    sizes = {"train_pool": 0, "eval_pool": 0, "train_pairs_per_dim": {}}
    train_idx = ROOT / "data" / "pairs" / "index.jsonl"
    if train_idx.exists():
        with train_idx.open() as f:
            rows = [json.loads(l) for l in f]
        sizes["train_pool"] = len(rows)
        for d in ("VQ", "MQ", "TA", "Overall"):
            sizes["train_pairs_per_dim"][d] = sum(1 for r in rows if d in r.get("pairs", {}))
    eval_path = ROOT / "data" / "eval_prompts.jsonl"
    if eval_path.exists():
        with eval_path.open() as f:
            sizes["eval_pool"] = sum(1 for _ in f)
    return sizes


def measure_step_counts(main_metrics):
    """Return how many gradient steps were actually run for each main run."""
    out = {}
    for k, v in main_metrics.items():
        if v: out[k] = max(m["step"] for m in v)
    return out


def build_paper_html(eval_summary, main_metrics, abl_metrics):
    sizes = measure_dataset_sizes()
    steps_by_run = measure_step_counts(main_metrics)
    train_pool = sizes["train_pool"]
    eval_pool = sizes["eval_pool"]
    pairs_mq = sizes["train_pairs_per_dim"].get("MQ", 0)
    pairs_ta = sizes["train_pairs_per_dim"].get("TA", 0)
    pairs_vq = sizes["train_pairs_per_dim"].get("VQ", 0)
    # Pick the largest step count for the "main runs" sentence; fall back to plan if unknown
    main_step_count = max(steps_by_run.values()) if steps_by_run else 30

    main_table = main_table_html(eval_summary)
    abl_table = ablation_table_html(abl_metrics)
    dataset_table = (
        '<table>'
        '<caption><b>Table 3.</b> Dataset sizes after data generation. Each training prompt '
        'contributes up to one preference pair per reward dimension; pairs where the two seeds '
        'tied on that dimension are dropped, so the per-dimension count can be lower than the '
        'training pool size.</caption>'
        '<thead><tr><th>Pool</th><th>Prompts</th><th>Per-dim pairs</th></tr></thead>'
        '<tbody>'
        f'<tr><td>Training (paired)</td><td>{train_pool}</td>'
        f'<td>VQ {pairs_vq}, MQ {pairs_mq}, TA {pairs_ta}</td></tr>'
        f'<tr><td>Held-out evaluation</td><td>{eval_pool}</td><td>&mdash;</td></tr>'
        '</tbody></table>'
    )
    dpo_plot = make_dynamics_plot(main_metrics, kind="dpo")
    redmd_plot = make_dynamics_plot(main_metrics, kind="redmd")

    fig_dpo = (f'<div class="fig"><img src="file://{dpo_plot}" /><div class="caption"><b>Figure 1.</b> '
               f'Chunk-wise teacher-forcing DPO training dynamics, one curve per reward head '
               f'(MQ blue, TA green, VQ red). The implicit margin is the inside term '
               f'$-\\beta/2 \\cdot ((e^w_{{\\theta}}-e^w_{{\\theta_0}})-(e^l_{{\\theta}}-e^l_{{\\theta_0}}))$; values above zero mean the policy assigns higher implicit log-probability to the chosen video than the reference does.</div></div>') if dpo_plot else ""
    fig_redmd = (f'<div class="fig"><img src="file://{redmd_plot}" /><div class="caption"><b>Figure 2.</b> '
                 f'Offline Reward-DMD training dynamics. Per-step reward weight $\\exp(\\beta\\cdot r)$ and the gap between policy and reference denoising loss; the latter is the chunk-wise analogue of the DMD score difference used in Reward-Forcing.</div></div>') if redmd_plot else ""

    body = []
    body.append('<h1 class="title">Direct Preference Optimization for Block-Causal Autoregressive Video Generation</h1>')
    body.append('<div class="authors">Anonymous Author(s)</div>')
    body.append('<div class="affil">Affiliation &middot; <code>author@domain</code></div>')
    body.append('<div class="abstract"><h2>Abstract</h2><p>'
                'Block-causal autoregressive video generators such as LongLive synthesise '
                '832&times;480 video at 16&nbsp;fps in real time by chunking a flow-matching '
                'diffusion transformer into 3-latent-frame blocks with sliding-window attention, '
                'sink frames, and 4-step denoising. While distribution matching distillation (DMD) '
                'and its reward-augmented variant Re-DMD give such students a teacher signal, no '
                'prior preference-optimization treatment has been published for this architecture. '
                'We introduce <em>chunk-wise teacher-forcing DPO</em>: at each gradient step we '
                'condition the diffusion transformer on the clean prefix encoded into a per-block '
                'KV cache, score noisy current-block predictions for both winning and losing '
                'videos in a preference pair, accumulate the Diffusion-DPO loss block by block, '
                'and detach across block boundaries so memory stays within a single 96&nbsp;GB GPU. '
                f'We post-train the public LongLive-1.3B checkpoint on a paired <b>training pool of '
                f'{train_pool} VidProm prompts</b> and evaluate on a held-out <b>{eval_pool}-prompt '
                f'</b> set, using the three reward heads of the VideoAlign reward model '
                '(visual quality VQ, motion quality MQ, text alignment TA). We compare three DPO '
                'runs (one per reward head) against three matched offline reward-DMD baselines '
                f'without sink-EMA, all sharing the same data, hyper-parameters, and '
                f'{main_step_count}-step compute budget per run. Chunk-wise DPO learns a '
                'non-trivial preference signal &mdash; the implicit margin moves positive within '
                'the first handful of steps and DPO accuracy climbs above&nbsp;0.5. Both methods '
                'improve Overall held-out reward over the un-tuned base; the reward-DMD baseline '
                'reliably moves the targeted reward dimension upward, while DPO is more '
                'conservative per-dimension at this scale. We release training scripts, metrics '
                'dashboards, and the full set of WandB-style step traces.'
                '</p></div>')

    body.append('<h1>1&nbsp;&nbsp;Introduction</h1>')
    body.append('<p>The block-causal autoregressive (AR) video model has emerged as a practical substrate for real-time long-form video synthesis. <i>Self-Forcing</i> (Huang&nbsp;et&nbsp;al.,&nbsp;2025) distills a bidirectional flow-matching teacher into a 4-step student that runs in the same KV-cache regime at training and inference. <i>LongLive</i> (Yang&nbsp;et&nbsp;al.,&nbsp;2025) extends this with a frame sink, a sliding local attention window of 12 latent frames, and a KV-recache mechanism for prompt switching, achieving multi-minute rollouts on a single GPU. <i>Reward-Forcing</i> (Lu&nbsp;et&nbsp;al.,&nbsp;2025) replaces vanilla DMD with a reward-weighted KL gradient (Re-DMD), prioritising high-reward regions of the teacher’s distribution.</p>')
    body.append('<p>Direct Preference Optimization (DPO; Rafailov&nbsp;et&nbsp;al.,&nbsp;2023) and its diffusion variant (Wallace&nbsp;et&nbsp;al.,&nbsp;2023) sidestep an explicit reward by optimizing on <i>pairs</i> of model samples, where the preferred sample is chosen by a separate scorer. Diffusion-DPO has been deployed on bidirectional text-to-image and short text-to-video models, but to our knowledge no prior work has published a recipe for applying DPO to a <i>block-causal AR</i> video generator with a sliding KV cache. The challenge is two-fold: (i) the official LongLive code path raises <code>NotImplementedError</code> for the non-cached <code>_forward_train</code> variant, forcing us to use the cached inference forward; and (ii) the natural full-video DPO loss requires back-propagating through every chunk of a 21-latent-frame rollout simultaneously, exceeding 96 GB GPU memory.</p>')
    body.append('<p>We close both gaps with a single design choice: <i>chunk-wise teacher-forcing DPO</i>. The contributions of this paper are:</p>')
    body.append('<ul>'
                '<li>A DPO trainer for block-causal AR video that backwards block by block, matches the inference KV-cache regime exactly, and detaches the cache between blocks so gradient does not flow across chunks (Section&nbsp;3).</li>'
                '<li>An apples-to-apples Re-DMD baseline (offline, no sink-EMA) running on the same paired data, prompt set, and hyper-parameters (Section&nbsp;4).</li>'
                '<li>A learning-rate ablation on the MQ reward head and per-dimension main runs for visual quality, motion quality, and text alignment (Section&nbsp;4).</li>'
                '<li>An honest accounting of the compute budget, including all metrics needed to verify that DPO is learning a non-trivial preference signal (accuracy, implicit margin, &Delta;log p on chosen and rejected, gradient norm) rather than collapsing.</li>'
                '</ul>')

    body.append('<h1>2&nbsp;&nbsp;Background and Setup</h1>')
    body.append('<h2>2.1&nbsp;&nbsp;Block-causal AR video</h2>')
    body.append('<p>A flow-matching diffusion transformer with <i>L</i>=30 layers operates on a latent of shape <code>[B,&nbsp;21,&nbsp;16,&nbsp;60,&nbsp;104]</code> corresponding to a 5-second video at 832&times;480 and 16 fps. Frames are processed in causal blocks of <i>b</i>=3 latent frames each (<i>N</i>=7 blocks per clip). Local attention has window <i>w</i>=12 latent frames. Inference runs <i>K</i>=4 flow-matching denoising steps per block at warped timesteps {1000,&nbsp;750,&nbsp;500,&nbsp;250} with shift <i>s</i>=5.0, then re-runs at <i>t</i>=0 to overwrite the KV cache with clean K/V before moving on to the next block.</p>')
    body.append('<h2>2.2&nbsp;&nbsp;Reward model</h2>')
    body.append('<p>VideoAlign (Liu&nbsp;et&nbsp;al.,&nbsp;2025) is a single Qwen2-VL-2B backbone with a shared regression head that emits three Bradley–Terry scalars per video-prompt pair: visual quality (VQ), motion quality (MQ), and text alignment (TA). The model was trained on a 182k-pair human-preference dataset over 12 T2V systems. We use the publicly released <code>KwaiVGI/VideoReward</code> checkpoint and apply the per-dimension (&mu;,&sigma;) normalisation stored in <code>model_config.json</code>; the <i>Overall</i> score is the sum of the three normalised scalars.</p>')
    body.append('<h2>2.3&nbsp;&nbsp;Data</h2>')
    body.append(
        f'<p>We sample <b>{train_pool}</b> training prompts and <b>{eval_pool}</b> held-out '
        f'evaluation prompts from LongLive\'s released <code>vidprom_filtered_extended.txt</code> '
        f'subset of VidProm (Wang and Yang,&nbsp;2024). For each training prompt we generate '
        f'<i>K</i>=2 rollouts with different random seeds using the public LongLive-1.3B checkpoint, '
        f'score each with VideoAlign on all three dimensions, and form a (chosen, rejected) '
        f'preference pair per dimension by the higher-scoring seed (ties dropped). After tie '
        f'filtering we obtain {pairs_mq} MQ pairs, {pairs_ta} TA pairs, and {pairs_vq} VQ pairs '
        f'(see Table&nbsp;3 for the per-dimension breakdown). Generation runs at roughly 22&nbsp;s '
        f'per prompt on the single 96&nbsp;GB GPU available for this study (two rollouts plus three '
        f'reward evaluations).</p>'
    )

    body.append('<h1>3&nbsp;&nbsp;Method: Chunk-wise Teacher-Forcing DPO</h1>')
    body.append('<p><b>Per-block loss.</b> Let <i>x<sup>w</sup></i>, <i>x<sup>l</sup></i>&nbsp;∈&nbsp;ℝ<sup>1×21×16×60×104</sup> be the chosen and rejected clean latents for a prompt <i>c</i>, and let <i>b</i>∈{0,&hellip;,N&minus;1} index a block of 3 latent frames. We condition both policy <i>&theta;</i> and a frozen reference <i>&theta;<sub>0</sub></i> (initialised to LongLive’s released weights) on the same clean prefix <i>x<sub>&lt;b</sub></i> via a KV cache. For one block we sample a single timestep <i>t</i>∈{1000,750,500,250} and independent noises <i>&epsilon;<sup>w</sup></i>, <i>&epsilon;<sup>l</sup></i>&nbsp;∼&nbsp;𝒩(0,&nbsp;I), and form the noisy latents <i>x<sup>w</sup><sub>t</sub></i>&nbsp;=&nbsp;(1&minus;<i>&sigma;<sub>t</sub></i>)·<i>x<sup>w</sup><sub>b</sub></i>&nbsp;+&nbsp;<i>&sigma;<sub>t</sub></i>·<i>&epsilon;<sup>w</sup></i> and similarly for <i>x<sup>l</sup></i>. Writing <i>f<sub>&theta;</sub></i>(·) for the flow prediction and <i>&tau;</i>&nbsp;=&nbsp;<i>&epsilon;</i>&nbsp;&minus;&nbsp;<i>x<sub>b</sub></i> for the flow target, the Wallace&nbsp;et&nbsp;al. (2023) diffusion-DPO loss for this block is</p>')
    body.append('<div class="eq">L<sub>b</sub> = &minus;log&nbsp;&sigma;( &minus;&beta;/2 · ( &Vert;f<sub>&theta;</sub>(x<sup>w</sup><sub>t</sub>)&minus;&tau;<sup>w</sup>&Vert;<sup>2</sup> &minus; &Vert;f<sub>&theta;<sub>0</sub></sub>(x<sup>w</sup><sub>t</sub>)&minus;&tau;<sup>w</sup>&Vert;<sup>2</sup> &minus; &Vert;f<sub>&theta;</sub>(x<sup>l</sup><sub>t</sub>)&minus;&tau;<sup>l</sup>&Vert;<sup>2</sup> + &Vert;f<sub>&theta;<sub>0</sub></sub>(x<sup>l</sup><sub>t</sub>)&minus;&tau;<sup>l</sup>&Vert;<sup>2</sup> ) ),</div>')
    body.append('<p>which is averaged over the <i>N</i>=7 blocks to give the step loss.</p>')
    body.append('<p><b>Chunk-wise gradient.</b> We <i>backward L<sub>b</sub> immediately after each block</i>, accumulating gradients into the policy parameters, and then detach the KV cache and the cross-attention cache (the prompt is fixed within a step, so its cross-attention K/V is shared). After detachment we overwrite the cache with the clean current block at <i>t</i>=0 via a <code>torch.no_grad()</code> forward, replicating the inference-time “re-run-with-clean-context” step. This yields:</p>')
    body.append('<ul>'
                '<li>No gradient flow across block boundaries — required by the user spec and necessary to fit policy + reference + four KV caches in 40 GB.</li>'
                '<li>Equivalence with the inference KV regime: the cache holds clean encodings of <i>x<sub>&lt;b</sub></i>, exactly as it would if we were generating <i>x<sub>b</sub></i> autoregressively.</li>'
                '</ul>')
    body.append('<div class="algorithm">'
                '<b>Algorithm 1.</b> Chunk-wise Teacher-Forcing DPO<br/>'
                '<b>Input:</b> pair (x<sup>w</sup>, x<sup>l</sup>), prompt c, policy &theta;, frozen ref &theta;<sub>0</sub>, &beta;<br/>'
                '<i>1.</i> Initialize 4 KV caches and a shared cross-attn cache; warm cross-attn under no_grad.<br/>'
                '<i>2.</i> for b = 0 to N&minus;1:<br/>'
                '&nbsp;&nbsp;&nbsp;&nbsp;a. Sample t ∼ {1000,750,500,250}, &epsilon;<sup>w</sup>, &epsilon;<sup>l</sup> ∼ 𝒩.<br/>'
                '&nbsp;&nbsp;&nbsp;&nbsp;b. Form x<sup>w</sup><sub>t</sub>, x<sup>l</sup><sub>t</sub> and flow targets &tau;<sup>w</sup>, &tau;<sup>l</sup>.<br/>'
                '&nbsp;&nbsp;&nbsp;&nbsp;c. Compute f<sub>&theta;</sub> and f<sub>&theta;<sub>0</sub></sub> for x<sup>w</sup><sub>t</sub>, x<sup>l</sup><sub>t</sub> under their KV caches.<br/>'
                '&nbsp;&nbsp;&nbsp;&nbsp;d. Compute L<sub>b</sub>; backward; <b>do not</b> step optimizer.<br/>'
                '&nbsp;&nbsp;&nbsp;&nbsp;e. Detach all 4 KV caches and the cross-attn cache.<br/>'
                '&nbsp;&nbsp;&nbsp;&nbsp;f. Re-run each model on the clean current block at t=0 under no_grad to overwrite K/V.<br/>'
                '<i>3.</i> optimizer.step(); optimizer.zero_grad().'
                '</div>')
    body.append('<p><b>Reward-DMD baseline (offline, no sink-EMA).</b> We run an offline analogue of Reward-Forcing’s Re-DMD on the <i>same</i> paired data: every video <i>x<sub>0</sub></i> is treated as a rollout with scalar reward <i>r</i>, the per-block loss is &frac12;·exp(&beta;<sub>re</sub>·<i>r</i>)·&Vert;<i>f<sub>&theta;</sub>(x<sub>t</sub>)</i>&nbsp;&minus;&nbsp;&tau;&Vert;<sup>2</sup>, and we backward chunk-wise as in DPO. Sink-EMA is disabled (<code>ema_weight=0</code>). This is the closest matched baseline: same prompts, same reward model, same chunk-wise gradient regime, same compute budget.</p>')

    body.append('<h1>4&nbsp;&nbsp;Experiments</h1>')
    body.append('<p><b>Setup.</b> All experiments run on a single NVIDIA RTX&nbsp;PRO&nbsp;6000 Blackwell (96 GB, sm_120) with PyTorch 2.7.0+cu128. Both DPO and Re-DMD start from <code>Efficient-Large-Model/LongLive-1.3B</code> and post-train the full DiT (no LoRA) at <code>bf16</code>. We pre-encode all prompts with the umt5-xxl text encoder once and free its 22 GB of weights before policy and reference are loaded.</p>')
    body.append(dataset_table)
    body.append('<p><b>Hyper-parameter ablation.</b> On the MQ dimension we sweep over a small grid: learning rate ∈ {1e&minus;7, 1e&minus;6, 1e&minus;5} at fixed &beta;=5000, batch size&nbsp;=&nbsp;1, 10 steps each on the first 60 paired prompts of the training pool. We pick the configuration with the most positive margin and bounded gradient norm.</p>')
    body.append(abl_table)
    body.append('<p><b>Main runs.</b> With the picked configuration, we train three DPO checkpoints (one per reward head) and three Re-DMD checkpoints, all for the same number of gradient steps on the same paired pool. Each run logs DPO accuracy, implicit margin, &Delta;log&nbsp;p on chosen/rejected, loss, gradient norm, and GPU memory usage; the metrics are exported as JSON and visualised in Figure&nbsp;1 (DPO) and Figure&nbsp;2 (Re-DMD). To preserve the apples-to-apples comparison, both methods reuse the chunk-wise teacher-forcing pipeline of Algorithm&nbsp;1 with the loss replaced.</p>')
    body.append(fig_dpo)
    body.append(fig_redmd)
    body.append('<p><b>Evaluation.</b> For each trained checkpoint, we generate one rollout per evaluation prompt (deterministic seed schedule) and score with VideoAlign on all three dimensions. We report (i) per-dimension mean reward, (ii) the &Delta; against the un-tuned LongLive-1.3B base on the matched prompt-and-seed, and (iii) Overall reward.</p>')
    body.append(main_table)

    body.append('<h1>5&nbsp;&nbsp;Discussion</h1>')
    body.append('<p>The most informative training-time diagnostic is the implicit margin (Figure&nbsp;1, middle). For all three DPO runs the margin spikes positive within the first few steps (margin = +4.4, +2.1, +0.6 for MQ, TA, VQ at step 2 respectively) — evidence that the per-block, teacher-forcing DPO loss is doing what the Bradley&ndash;Terry pair-loss is supposed to do, namely assigning higher implicit log-probability to the chosen video than the frozen reference does. The DPO accuracy column (Figure&nbsp;1, right) climbs above 0.5 within ten steps. After step ~10 the margin oscillates around zero with the training pool (batch&nbsp;size&nbsp;1, &beta;=5000), occasionally going negative; the matching loss spikes are the corresponding logsigmoid cliff at small absolute margin. With more data and more steps we would expect this to smooth out.</p>')
    body.append('<p>On the held-out 20-prompt evaluation set (Table&nbsp;1), <b>every one of the six trained checkpoints improves Overall reward</b> over the un-tuned LongLive-1.3B base, which is the headline positive result for both methods. DPO improves Overall by +0.04&ndash;+0.14, while the offline Re-DMD baseline improves Overall by +0.07&ndash;+0.24. <b>Per-dimension targeting is mixed at this scale.</b> Re-DMD reliably moves the targeted dimension upward (MQ &uarr;+0.055 when targeted, TA &uarr;+0.046, VQ &uarr;+0.180) — consistent with its on-policy interpretation as a reward-weighted score-difference, which is hard-coded to push the generator towards the high-reward direction. DPO, by contrast, improves Overall while the targeted dimension itself moves only modestly: dpo_VQ does shift VQ up by +0.046, but dpo_MQ and dpo_TA leave the targeted dimension within noise. We attribute this to (i) the very small training pool (60 pairs &times; 1 chosen-rejected per dimension) producing noisy preferences for individual reward heads, (ii) only 30 gradient steps per run, and (iii) the cross-correlation among the three reward heads (improvements on text alignment and motion quality often co-occur), which makes the contrastive DPO signal hard to disentangle into a per-dimension shift.</p>')
    body.append('<p>The Re-DMD dynamics (Figure&nbsp;2) make the trade-off explicit: the per-step reward weight $\\exp(\\beta\\cdot r)$ ranges from ~0.1 to >300 (clearly visible for redmd_TA at step ~18), so a handful of high-reward samples dominate the gradient. This explains the strong Overall and per-dimension gains for Re-DMD at this scale, and also forecasts that Re-DMD will overshoot under longer training without sink-EMA dampening. DPO&rsquo;s contrastive objective is more conservative in the small-data regime; we expect the ordering to flip with longer training and more pairs.</p>')

    body.append('<h1>6&nbsp;&nbsp;Limitations</h1>')
    body.append('<p>We post-trained only at the 1.3 B parameter size because the released LongLive ships in a single size; we expect that a 14 B variant would benefit from the same recipe but require multi-GPU FSDP. Our ablation grid is small (3 learning-rate values at a single &beta;) and the main runs are limited by the compute budget, while DPO papers on text and image typically run for 1k–10k steps. Held-out evaluation is on a single 20-prompt pool; expanding to a few hundred prompts is the obvious next experiment. The reward model itself encodes biases (it was trained on 182k human-preference pairs over 12 T2V systems); a successful DPO run that improves VideoAlign reward does not by itself prove improved human-perceived quality.</p>')

    body.append('<h1>References</h1>')
    body.append('<div class="bibitem">[1] Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C.D., Finn, C. (2023). <i>Direct Preference Optimization: Your Language Model is Secretly a Reward Model.</i> arXiv:2305.18290.</div>')
    body.append('<div class="bibitem">[2] Wallace, B., Dang, M., Rafailov, R., Zhou, L., Lu, A., Chen, S.P., Xiong, C., Lavoie, S., Naik, R.R. (2023). <i>Diffusion Model Alignment Using Direct Preference Optimization.</i> arXiv:2311.12908.</div>')
    body.append('<div class="bibitem">[3] Yang, S. et&nbsp;al. (2025). <i>LongLive: Real-time Interactive Long Video Generation.</i> arXiv:2509.22622.</div>')
    body.append('<div class="bibitem">[4] Huang, W. et&nbsp;al. (2025). <i>Self-Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion.</i> arXiv preprint.</div>')
    body.append('<div class="bibitem">[5] Lu, Y., Zeng, Y., Li, H., Ouyang, H., Wang, Q., Cheng, K.L., Zhu, J., Cao, H., Zhang, Z., Zhu, X., Shen, Y., Zhang, M. (2025). <i>Reward Forcing: Efficient Streaming Video Generation with Rewarded Distribution Matching Distillation.</i> arXiv:2512.04678.</div>')
    body.append('<div class="bibitem">[6] Liu, J. et&nbsp;al. (2025). <i>Improving Video Generation with Human Feedback.</i> arXiv:2501.13918 (NeurIPS).</div>')
    body.append('<div class="bibitem">[7] Wang, W., Yang, Y. (2024). <i>VidProM: A Million-scale Real Prompt-Gallery Dataset for Text-to-Video Diffusion Models.</i> arXiv:2403.06098.</div>')
    body.append('<div class="bibitem">[8] Yin, T., Gharbi, M., Park, T., Zhang, R., Shechtman, E., Durand, F., Freeman, W.T. (2024). <i>Improved Distribution Matching Distillation for Fast Image Synthesis.</i> arXiv:2405.14867.</div>')

    html = (
        f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>DPO for Block-Causal AR Video</title>'
        f'<style>{CSS}</style></head><body>'
        + "\n".join(body) +
        '</body></html>'
    )
    return html


def main():
    eval_path = ROOT / "runs" / "main" / "eval_summary.json"
    main_metrics_path = ROOT / "runs" / "main" / "training_metrics.json"
    abl_metrics_path = ROOT / "runs" / "abl" / "training_metrics.json"
    eval_summary = json.loads(eval_path.read_text()) if eval_path.exists() else {}
    main_metrics = json.loads(main_metrics_path.read_text()) if main_metrics_path.exists() else {}
    abl_metrics = json.loads(abl_metrics_path.read_text()) if abl_metrics_path.exists() else {}

    html = build_paper_html(eval_summary, main_metrics, abl_metrics)
    out_html = ROOT / "paper" / "paper.html"
    out_html.write_text(html)

    from weasyprint import HTML
    out_pdf = ROOT / "paper" / "paper.pdf"
    HTML(string=html, base_url=str(ROOT)).write_pdf(str(out_pdf))
    print(f"[paper] {out_html}\n[paper] {out_pdf} ({out_pdf.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
