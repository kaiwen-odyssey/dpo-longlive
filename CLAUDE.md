# dpo-longlive — Claude project notes

This file is the self-contained snapshot of project memory. It is auto-loaded by Claude Code when a session is opened in this directory. Mirrors of these notes also exist under `~/.claude/projects/-home-kaiwen-dev-dpo-longlive/memory/`.

---

## 1. Project goal

Apply **Direct Preference Optimization (DPO)** to **NVlabs/LongLive 1.3B** (block-causal AR video, Wan2.1-T2V-1.3B student, 4-step denoising, 21 latent frames × 3-frame blocks, sliding-window attn 12, sink size 3) and compare against an **offline reward-DMD baseline** (the Re-DMD half of Reward-Forcing, **without sink-EMA**). Reward signal: KwaiVGI/VideoReward (Qwen2-VL-2B with shared rm_head; output `[B,3]` for VQ/MQ/TA, Overall = sum after per-dim normalisation; reader-side `fps=2.0` so a 5-sec video → ~10 sampled frames per scoring).

Training data: **1000 paired VidProm prompts** (gap-filled 2026-05-11 from the initial 851), 2 rollouts/prompt at per-prompt seeds `1000 + 31·pid + 7·k` (k=0,1), scored with VideoAlign, (chosen, rejected) per dimension by higher score (ties dropped → MQ **997** / TA 986 / VQ 987 pairs). Held-out: **20 prompts** (and an extended 50-prompt eval set = those 20 + 30 fresh VidProm prompts sampled no-train-overlap, used by v4+). Each rollout: 5 s @ 16 fps, latent shape `[1, 21, 16, 60, 104]`.

**Why this project:** there is no published preference-optimisation treatment of block-causal AR video models; the LongLive paper itself does not cite DPO. This is the gap.

---

## 2. Hardware & toolchain

- Single **NVIDIA RTX PRO 6000 Blackwell Max-Q** (96 GB, sm_120). No multi-GPU. **Disk-tight workspace** — typical free 15–50 GB; the 2.7 GB DiT checkpoints fill it fast. Run `--no_save_policy` for grid-style runs; manually delete `policy_final.pt` from runs you don't need to keep.
- 125 GB RAM, 16 GB swap.
- **PyTorch 2.7.0+cu128** in `.venv/`. cu124 builds will NOT load on sm_120.
- bf16 throughout, full-DiT post-training (no LoRA).
- TeXLive hand-installed under `/tmp/tex_install/root` (no sudo). NeurIPS 2024 `.sty` is in `paper/`.
- umt5-xxl text encoder is ~22 GB on GPU; encode all prompts once at init, then `del` (or use the `CachedTextEncoder` lookup wrapper from `scripts/train_v4_mq.py` to keep `pipe.text_encoder` callable with cached embeddings).

---

## 3. Design constraints (non-negotiable)

**Chunk-wise teacher-forcing DPO**: per-block backward, gradient does NOT flow across the 7 chunks of a 21-frame rollout.

1. Teacher-forcing, not on-policy rollout. Given a (chosen, rejected) latent pair, condition each block on the *clean* preceding blocks of that video, not on the model's own previous-block predictions.
2. Each block backward immediately after its loss; detach all 4 KV caches (policy/ref × chosen/rejected) plus the shared cross-attn cache after each block.
3. Re-run each model on the clean current block at `t=0` under `torch.no_grad()` to overwrite the noisy K/V with clean K/V before moving to the next block — mirrors the inference KV regime exactly.

**Two loss variants implemented in `src/dpo_longlive/dpo_trainer.py`:**

- **`chunkwise_dpo_step`** (per-chunk): `L = mean_b(−log σ(inside_b)) + α · mean_b(e_pol_w_b)`. One sigmoid per block, gradient saturates per block. Used by v2/v3/v4.
- **`chunkwise_dpo_step_aggregated`** (rollout-level, added 2026-05-12): `L = −log σ(mean_b(inside_b)) + α · mean_b(e_pol_w_b)`. Single sigmoid on rollout-mean margin. Two-pass: pass 1 (no_grad) computes per-block margins; pass 2 backwards surrogate `scale·inside_b + (α/N)·e_pol_w_b` per block where `scale = −(1−σ(mean_inside))/N`. Same `(t, ε_w, ε_l)` shared across all 7 chunks (variance reduction). Used by v4_agg/v6/v7. ~2× compute per step.

---

## 4. Picked hyper-parameters (across recipes)

| recipe | loss | β | lr | α | filter | steps | best Δ\,MQ | best win | eval set |
|--------|------|---|----|---|--------|------:|-----------:|---------:|----------|
| v1 | per-chunk | 5000 | 1e-6 | 0 | – | 30 | −0.033 | 0.55 | 20-prompt |
| **v2** | per-chunk | 500 | 2e-5 | 0 | – | 30 | **+0.195** | 0.60 | 20-prompt |
| v3 | per-chunk | 500 | 2e-5 | 1.0 | δ_gap=0.2 | 400 | −0.003 | 0.55 | 20-prompt (over-fits) |
| v4 | per-chunk | 500 | 5e-6 | 0.5 | – | 1000 (peak@900) | +0.104 | 0.70 | 50-prompt |
| v4_agg | aggregated | 500 | 5e-6 | 0.5 | – | 1000 (peak@800) | +0.020 | 0.48 | 50-prompt |
| v5/v5b/v5c | aggregated | various | 5e-5–1e-4 | 0.5 | – | killed early | – | – | – (all diverged/drifted) |
| v6 | aggregated | 100 | 5e-5 | 0.5 | – | killed @ step 200 (Δ −0.27) | – | – | 50-prompt |
| **v7** | **aggregated** | **100** | **2e-5** | **0.5** | – | **1000 (peak@400)** | **+0.802** | **0.86** | **50-prompt** |

50-step β×lr ablation (held-out DPO acc/margin only) winner: β=500, lr=5e-6 (eval margin +0.059, win 0.75). 100-step aggregated β×lr grid winner: β=100, lr=2e-5 (eval Δ +0.170, win 0.56) — selected as v7 launch config.

**Stability watch-outs:**
- β≥500 with aggregated + lr≥1e-4 → step-10 catastrophic divergence (margin spikes to ±33, gradient pre-clip > 1000).
- aggregated + lr=2e-5 (v7) is bimodal: ~75% of checkpoints in strongly-positive basin (Δ +0.4 to +0.8), occasional 100-step windows drop to Δ ≈ −0.5 before recovering. Save `policy_best.pt` (already automatic in `train_v4_mq.py`); don't use `policy_final.pt`.
- v3-style anchor α=1.0 + min_gap filter + 400 steps over-fits the 473-pair filtered pool — keep α≤0.5 and don't filter.

---

## 5. Headline results

### v4_mq (per-chunk, 1000 steps, 50-prompt eval, baseline eval-50 mean MQ +0.057)

| step | eval-50 Δ | eval-50 win | train-100 mean | train-100 win |
|------|----------:|------------:|---------------:|--------------:|
| 100  | −0.062 | 0.40 | +0.047 | 0.52 |
| 300  | −0.073 | 0.36 | +0.080 | 0.47 |
| 600  | +0.013 | 0.52 | +0.107 | 0.53 |
| **900** | **+0.104** | **0.70** | +0.069 | 0.61 |
| 1000 | +0.067 | 0.56 | +0.196 | 0.58 |

Smooth arc with dip → steady climb → peak at step 900 → slight regression. `policy_best.pt` = step 900.

### v7_mq_agg_b100_lr2e-5 (aggregated, β=100, lr=2e-5, 1000 steps)

| step | eval-50 Δ | eval-50 win | train-100 mean | train-100 win |
|------|----------:|------------:|---------------:|--------------:|
| 100  | +0.099 | 0.56 | +0.230 | 0.63 |
| 200  | −0.510 | 0.30 | −0.253 | 0.16 |
| 300  | +0.536 | 0.86 | +0.569 | 0.81 |
| **400** | **+0.802** | **0.86** | **+1.010** | **0.89** |
| 500  | +0.569 | 0.82 | +0.691 | 0.77 |
| 600  | −0.451 | 0.28 | −0.315 | 0.27 |
| 700–1000 | +0.40 to +0.48 | 0.70–0.76 | +0.39 to +0.63 | 0.64–0.75 |

Peak at step 400: **8× v4 per-chunk peak Δ, same eval set, same per-prompt seeds**. `policy_best.pt` = step 400 (also backed up at `policy_step300_backup.pt`). Both held-out and training-pool sides move together → broad generalisation, not eval-specific. Bimodal volatility: occasional 100-step windows collapse to Δ ≈ −0.5 then recover.

### Per-prompt seed convention (all v4+ runs)

For prompt id `pid`:
- Cached base rollout seed: `1000 + 31·pid` (k=0, the "seed_a" from `gen_pairs.py`). MQ score in `data/pairs/<id>/scores.json` under that seed key.
- Policy rollout (held-out eval, train-window eval) uses the **same** matched seed `1000 + 31·pid` so the comparison `MQ_policy > MQ_base` only varies model weights.
- Eval-50 extended set: 20 original held-out (ids 0–19) + 30 new VidProm prompts (ids 1000–1029 sampled with `vidprom_seed=7`).

---

## 6. Layout

```
src/dpo_longlive/
  dpo_trainer.py                   # chunkwise_dpo_step + chunkwise_dpo_step_aggregated + run()
  longlive_pipeline.py             # load_pipeline + generate_one
  redmd_trainer.py                 # Re-DMD half (offline, no sink-EMA)
  reward_lib.py                    # VideoAlign wrapper

scripts/
  gen_pairs.py                     # data prep: 2 rollouts/prompt → score → pairs (resumable)
  train_v4_mq.py                   # v4/v6/v7 trainer: 1000-step + periodic eval + CachedTextEncoder
                                   #   --aggregate_chunks, --no_save_policy flags
  run_all.py                       # v2/v3 main entrypoint (per-chunk only)
  run_ablation_betalr.py           # 20-step β×lr sweep (per-chunk)
  run_ablation_betalr_50step.py    # 50-step β×lr ablation + held-out DPO eval
  eval_specific.py                 # eval a list of checkpoints on the 20-prompt held-out
  eval_dpo_holdout.py              # held-out DPO acc/margin diagnostic (over-fit detector)
  plot_v4_mq.py, plot_v7_mq.py     # 2-panel mean-MQ + win-rate plots
  plot_mq_training.py              # v3 windowed vs v2 per-step
  post_v7.sh                       # waits for v7 process, then plot+paper+commit+push

paper/
  build_tex.py                     # SINGLE SOURCE for paper/main.tex; reads runs/ + paper/figs
  neurips_2024.sty                 # NeurIPS 2024 template

runs/main_v2/                      # v2 30-step batch (full 4-dim eval)
  eval_summary.json                # base + 6 runs × 4 dims
  training_metrics.json            # per-step training for 6 runs
runs/main_v3/                      # v3 400-step batch; only dpo_MQ
  dpo_MQ/{policy_final.pt, training_summary.json}
  eval/summary.json                # base + dpo_MQ on 20-prompt held-out
  eval_dpo/{summary,metrics,pairs}.json/jsonl + latents/  # over-fit DPO diagnostic
runs/v4_mq/                        # per-chunk, lr=5e-6, 1000 steps
  metrics.json (per-step + 10 ckpts), policy_best.pt
runs/v4_mq_agg/                    # aggregated, β=500 lr=5e-6, 1000 steps
  metrics.json, base_eval.jsonl, eval_prompts_extra.jsonl, policy_best.pt
runs/v7_mq_agg_b100_lr2e-5/        # aggregated, β=100 lr=2e-5, 1000 steps (best)
  metrics.json, policy_best.pt, policy_step300_backup.pt, policy_final.pt
runs/grid_v5/                      # 100-step aggregated β×lr grid (4 configs, metrics only)
runs/abl_betalr/all_logs.json      # 20-step β×lr ablation (per-chunk)
runs/abl_betalr_50/all_logs.json   # 50-step β×lr ablation (per-chunk)

data/pairs/index.jsonl             # 1000 paired prompts (gap-filled 851→1000)
data/train_prompts.jsonl           # 1000 train prompts (VidProm)
data/eval_prompts.jsonl            # 20 original held-out prompts
assets/longlive/models/longlive_base.pt  # 5.3 GB base checkpoint
```

---

## 7. Paper compile

`python3 paper/build_tex.py` writes `paper/main.tex` and recompiles `paper/main.pdf` via two pdflatex passes from TeXLive at `/tmp/tex_install/root`. Reads:

- `runs/main_v2/eval_summary.json` — Table 3 main eval (v2 base + 5 v2 runs on 20-prompt)
- `runs/main_v3/eval/summary.json` — v3 dpo_MQ recipe-comparison row
- `runs/main_v3/dpo_MQ/training_summary.json` — Table 5 v3 training windowed metrics
- `runs/main_v3/eval_dpo/summary.json` — Table 6 over-fit diagnostic (train vs held-out DPO acc/margin)
- `runs/main_v2/training_metrics.json` — Figures 1, 2 (v2 DPO + Re-DMD dynamics)
- `runs/v4_mq/metrics.json` — Table 7 + Figure 4 v4 trajectory (50-prompt eval)
- `runs/v7_mq_agg_b100_lr2e-5/metrics.json` — Table 8 + Figure 5 v7 aggregated trajectory
- `runs/abl_betalr/all_logs.json` — Table 2 β×lr ablation
- `paper/figs/mq_training.png`, `v4_mq_progress.png`, `v7_mq_progress.png` — overlay plots

Edit `build_tex.py`, NOT the generated `main.tex`. Discussion paragraph honestly reflects the eval table (only v2 dpo_MQ is Overall-positive on 20-prompt; v7 aggregated peak is the headline on 50-prompt MQ-only).

---

## 8. Drive upload — MCP limit

A single `mcp__claude_ai_Google_Drive__create_file` call inlines at most **~25–30 KB of `base64Content` / `textContent`** before the assistant output-token budget truncates. Truncated string still uploads, producing a corrupted file. `paper/main.pdf` (180–290 KB depending on figure set) cannot be uploaded in one call. Upload `paper/main.tex` instead (32 KB, just fits). User must delete partial PDFs via the Drive UI — MCP exposes no delete operation. Auto-mode classifier also blocks chunked-base64 uploads as a "size-limit bypass" pattern; if needed, user runs the chunking themselves with `!` prefix.

---

## 9. Things to NOT do

- **No LoRA** — full-DiT post-training is the design.
- **No sink-EMA on Re-DMD** — `ema_weight=0` is what makes the apples-to-apples claim hold.
- **No cu124 PyTorch** — Blackwell sm_120 is unsupported there.
- **No 400+ steps on the v3 recipe without `--save_every 50`** — over-fits the 473-pair filtered pool. Avoid the filter entirely; use unfiltered ~997-pair MQ pool.
- **No aggregated DPO with β=500 lr≥1e-4** — step-10 catastrophic divergence (loss 33, margin −33, grad pre-clip 1k+). Even β=100 lr=1e-4 is too volatile.
- **No long training without `PYTHONUNBUFFERED=1`** — Python pipe-buffer hides progress for 10+ minutes.
- **No git config edits from the assistant** — user runs `git config ...` themselves.
- **No `git push` to `main` without user confirmation** — auto-mode classifier blocks it by default. User can `!git push --set-upstream origin main` themselves, or add a `Bash(git push:*)` permission rule.
- **No retry of full-PDF Drive upload via inline base64** — see §8.
- **No deletion of `runs/*/policy_best.pt`** for kept runs (v2, v3, v4, v7). Final and intermediate checkpoints from killed/superseded runs are safe to delete to recover disk.

---

## 10. Style preferences

- The user prefers honest paper text that matches the eval table (don't claim "every run improved" when only one did; v7 peak is on MQ-only on 50-prompt, not Overall on 20-prompt — keep that distinction).
- For training jobs that show held-out regression for several consecutive checkpoints, the user prefers killing the job and iterating rather than waiting for completion (cf. v3 continuation, v5, v5b, v5c, v6 — all killed for visible regression).
- Drive uploads are for finished artefacts only; in-progress work stays local.
- Terse responses for routine progress updates (e.g. "Continuing." or "ok"). Detailed table whenever the user asks for a snapshot of eval metrics across checkpoints.
- For multi-hour training runs, set up a chained bash watcher (`while pgrep -f ... > /dev/null; do sleep 60; done && next_job`) to auto-launch follow-up work rather than asking the user to wake up.
- After 1000-step runs, the standard wrap-up is: regen plot → rebuild paper → commit → push, ideally chained behind the trainer's process exit.
