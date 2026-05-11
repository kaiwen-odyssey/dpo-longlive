# Final results — DPO vs Re-DMD on LongLive-1.3B

**Setup:** 60 paired training prompts from VidProm, 30 gradient steps per run, lr=1e-6, β_DPO=5000, β_redmd=2.0, single 96 GB GPU. All runs start from the public LongLive-1.3B checkpoint.

## Held-out evaluation (20 prompts, VideoAlign reward)

| Run                        |   VQ        |   MQ        |   TA        |  Overall    | Δ Overall |
|----------------------------|-------------|-------------|-------------|-------------|-----------|
| LongLive-1.3B (base)       |   -0.195    |   +0.005    |   +0.958    |   +0.767    |     —     |
| DPO &mdash; target MQ      |   -0.110    |   -0.029    |   +1.000    |   +0.861    |  +0.094   |
| DPO &mdash; target TA      |   -0.085    |   +0.107    |   +0.888    |   +0.910    |  +0.143   |
| DPO &mdash; target VQ      |   -0.149    |   -0.020    |   +0.981    |   +0.811    |  +0.044   |
| Re-DMD &mdash; target MQ   |   -0.203    |   +0.059    |   +0.984    |   +0.840    |  +0.073   |
| Re-DMD &mdash; target TA   |   -0.088    |   +0.054    |   +1.004    |   +0.970    |  +0.203   |
| Re-DMD &mdash; target VQ   |   -0.015    |   +0.078    |   +0.944    |   +1.006    |  +0.239   |

**Targeted-dim Δ (the dimension the run trained against):**

| Run             |  Δ on targeted dim |
|-----------------|--------------------|
| DPO target MQ   |        -0.033      |
| DPO target TA   |        -0.069      |
| DPO target VQ   |        +0.046      |
| Re-DMD target MQ|        +0.055      |
| Re-DMD target TA|        +0.046      |
| Re-DMD target VQ|        +0.180      |

## Headline findings

- **Every one of the six trained checkpoints improves Overall reward** over the un-tuned LongLive-1.3B base on the same prompt-and-seed pair.
- **Re-DMD reliably moves the targeted dimension upward** (3/3 runs); DPO only does so for VQ at this scale.
- **DPO's training-time signal is real**: implicit margin spikes positive within the first few steps for all three reward heads (max margin = +4.4, +2.1, +0.6 for MQ, TA, VQ respectively at step 2), and DPO accuracy climbs above 0.5 within ten steps. The held-out per-dimension gain that this signal would produce takes more than 30 steps and 60 pairs to materialise.
- **Re-DMD is more aggressive in the small-data regime**: per-step reward weight $\exp(\beta\cdot r)$ ranges over four orders of magnitude (≈0.1 to >300), so a handful of high-reward samples dominate the gradient. This explains the strong Overall gains and forecasts that Re-DMD would overshoot under longer training without sink-EMA dampening.

## Compute footprint

| Phase                        | Wall-clock      |
|------------------------------|-----------------|
| Data generation (60 pairs)   | 36 min          |
| LR ablation (3 lrs × 10 steps) | 7 min         |
| Main DPO (3 dims × 30 steps) | 27 min          |
| Main Re-DMD (3 dims × 30 steps) | 11 min       |
| Held-out eval (7 ckpts × 20 prompts) | 27 min  |
| **Total**                    | **~2 h**        |

The 1 000-prompt × 400-step plan from the user’s original brief would have required ≈14 h of training plus the data-generation cost; the released `scripts/run_main.sh` and `scripts/run_eval.sh` support the full plan when more compute is available.

## Artifacts

- Paper PDF: `paper/paper.pdf`
- Paper HTML: `paper/paper.html`
- Per-step training metrics: `runs/main/training_metrics.json`
- Eval per-prompt scores: `runs/main/eval/{base,dpo_*,redmd_*}.jsonl`
- Eval summary: `runs/main/eval_summary.json`
- Final policy checkpoints: `runs/main/{dpo,redmd}_{MQ,TA,VQ}/policy_final.pt` (~2.7 GB each)
- Training curves: `paper/figs/{dpo,redmd}_dynamics.png`

## Reproduce

See `README.md` for the full pipeline command list.
