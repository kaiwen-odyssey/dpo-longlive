# dpo-longlive — Claude project notes

This file is the self-contained snapshot of project memory. It is auto-loaded by Claude Code when a session is opened in this directory. Mirrors of these notes also exist under `~/.claude/projects/-home-kaiwen-dev-dpo-longlive/memory/`.

---

## 1. Project goal

Apply **Direct Preference Optimization (DPO)** to **NVlabs/LongLive 1.3B** (block-causal AR video, Wan2.1-T2V-1.3B student, 4-step denoising, 21 latent frames × 3-frame blocks, sliding-window attn 12, sink size 3) and compare against an **offline reward-DMD baseline** (the Re-DMD half of Reward-Forcing, **without sink-EMA**). Reward signal: KwaiVGI/VideoReward (Qwen2-VL-2B with shared rm_head; output `[B,3]` for VQ/MQ/TA, Overall = sum after per-dim normalisation).

Training data: **851 paired VidProm prompts**, 2 rollouts/prompt scored with VideoAlign, (chosen, rejected) per dimension by higher score (ties dropped → MQ 849 / TA 838 / VQ 839 pairs). Held-out: **20 prompts**. Each rollout: 5 s @ 16 fps, latent shape `[1, 21, 16, 60, 104]`.

**Why this project:** there is no published preference-optimisation treatment of block-causal AR video models; the LongLive paper itself does not cite DPO. This is the gap.

---

## 2. Hardware & toolchain

- Single **NVIDIA RTX PRO 6000 Blackwell Max-Q** (96 GB, sm_120). No multi-GPU. Disk-tight workspace (~50 GB free).
- 125 GB RAM, 16 GB swap.
- **PyTorch 2.7.0+cu128** in `.venv/`. cu124 builds will NOT load on sm_120.
- bf16 throughout, full-DiT post-training (no LoRA).
- TeXLive hand-installed under `/tmp/tex_install/root` (no sudo). NeurIPS 2024 `.sty` is in `paper/`.
- umt5-xxl text encoder is ~23 GB; reuse the cached copy at `/home/kaiwen/dev/FastWan2.1-T2V-1.3B-Diffusers/text_encoder/`.

---

## 3. Design constraint (non-negotiable)

**Chunk-wise teacher-forcing DPO**: per-block backward, gradient does NOT flow across the 7 chunks of a 21-frame rollout.

1. Teacher-forcing, not on-policy rollout. Given a (chosen, rejected) latent pair, condition each block on the *clean* preceding blocks of that video, not on the model's own previous-block predictions.
2. Each block backward immediately after its loss; detach all 4 KV caches (policy/ref × chosen/rejected) plus the shared cross-attn cache after each block.
3. Re-run each model on the clean current block at `t=0` under `torch.no_grad()` to overwrite the noisy K/V with clean K/V before moving to the next block — mirrors the inference KV regime exactly.

Implementation: `src/dpo_longlive/dpo_trainer.py:chunkwise_dpo_step`. The Re-DMD baseline reuses the same chunk loop and swaps the loss.

---

## 4. Picked hyper-parameters

From the β×lr ablation on MQ (20 steps, 58 prompts, batch 1):

- **β = 500, lr = 2e-5** for DPO. β=1 trivial (loss stuck at ln 2). β=2000 lr=2e-5 diverges at grad ~9k. β=500 lr=2e-5 is the only row with positive margin (+0.50), low final loss (0.606), bounded gradient (avg 65).
- **β_redmd = 2.0** for Re-DMD (Reward-Forcing default).
- v1's β=5000 is over-aggressive and only appears in the recipe-comparison table for historical reference; **do not start new runs with β=5000**.

| β | lr | final loss | avg margin | avg acc | avg grad |
|---|---|---|---|---|---|
| 1 | 1e-6 | 0.693 | +0.0001 | 0.66 | 0.5 |
| 1 | 5e-6 | 0.693 | −0.0000 | 0.54 | 0.6 |
| 1 | 2e-5 | 0.695 | +0.0051 | 0.43 | 0.6 |
| 50 | 1e-6 | 0.693 | −0.0034 | 0.46 | 38.6 |
| 50 | 5e-6 | 0.692 | +0.0214 | 0.54 | 13.1 |
| 50 | 2e-5 | 0.710 | −0.1325 | 0.29 | 7.7 |
| 500 | 1e-6 | 0.694 | −0.0204 | 0.29 | 152.6 |
| 500 | 5e-6 | 0.641 | −0.0465 | 0.60 | 185.2 |
| **500** | **2e-5** | **0.606** | **+0.5023** | **0.60** | **64.8** |
| 2000 | 1e-6 | 0.680 | +0.1042 | 0.54 | 709.6 |
| 2000 | 5e-6 | 0.753 | +0.0485 | 0.57 | 335.7 |
| 2000 | 2e-5 | 7.602 | +12.1330 | 0.54 | 8783.2 (diverges) |

---

## 5. Current results (2026-05-11)

### v2 — β=500, lr=2e-5, 30 steps, batch 1, no anchor, no filter

All 6 runs (3 DPO + 3 Re-DMD) finished and evaluated. Base eval means: VQ −0.063, MQ +0.087, TA +0.900, **Overall +0.924**.

Held-out Δ vs base (and win-rate / 20):

| run | Δ VQ | Δ MQ | Δ TA | Δ Overall | win-MQ | win-O |
|---|---|---|---|---|---|---|
| dpo_MQ | −0.021 | **+0.195** | −0.037 | **+0.137** | 12/20 (0.60) | **13/20 (0.65)** |
| dpo_TA | −0.163 | −0.138 | +0.105 | −0.197 | 8/20 | 9/20 (0.45) |
| dpo_VQ | −0.065 | −0.229 | −0.042 | −0.336 | 5/20 | 8/20 (0.40) |
| redmd_MQ | +0.071 | +0.156 | −0.240 | −0.013 | 12/20 (0.60) | 8/20 (0.40) |
| redmd_TA | −0.142 | −0.057 | −0.507 | −0.705 | 8/20 | 3/20 (0.15) |
| redmd_VQ | −0.005 | +0.282 | −0.537 | −0.261 | 14/20 (0.70) | 5/20 (0.25) |

**Headline:** dpo_MQ v2 is the only Overall-positive run. Re-DMD reliably moves the targeted dim (redmd_MQ +0.156 on MQ, redmd_VQ +0.282 on MQ — weird cross-effect; only TA gain is dpo_TA +0.105) but trashes the other two dims, so all three Re-DMD runs regress on Overall.

### v3 — same recipe + grad_accum=4 + anchor α=1.0 + min_gap=0.2 + 400 steps (dpo_MQ only)

Filtered training pool after min_gap=0.2: **473 pairs** (1{,}600 pair-uses / 3.4 epochs).

Training windowed averages (training_summary.json, 83 logged opt-steps):

| window | loss | margin | acc |
|---|---|---|---|
| 50–150 | 1.351 | +0.317 | 0.56 |
| 150–250 | 1.481 | +0.353 | 0.57 |
| 250–350 | 1.258 | +0.323 | 0.61 |
| **final 20** | **1.316** | **+0.438** | **0.62** |

Overall avg acc 0.57; **positive margin on 58/82 = 71%** of logged steps.

Held-out eval (v3 base differs from v2 base by inference noise: VQ −0.065, MQ +0.118, TA +0.995, Overall +1.048):

| run | Δ VQ | Δ MQ | Δ TA | Δ Overall | win-MQ | win-O |
|---|---|---|---|---|---|---|
| v3 dpo_MQ | +0.075 | **−0.003** | −0.106 | **−0.034** | 11/20 (0.55) | **9/20 (0.45)** |

**Overfitting verdict:** training acc 0.39→0.62, training margin oscillating→consistently +0.4, but held-out Δ MQ +0.195→−0.003 and win-Overall 0.65→0.45. Textbook over-fit on the 473-pair filtered pool. Per-prompt Δ MQ σ=0.382 so the −0.003 mean is within noise — i.e. the v2 → v3 regression destroyed the gain rather than producing one.

### v3 continuation — CANCELLED 2026-05-11

`scripts/run_all.py` was relaunched for dpo_TA, dpo_VQ, redmd_MQ, redmd_TA, redmd_VQ at the v3 recipe; **killed at dpo_TA step ~115/400** once the dpo_MQ overfitting was diagnosed. Restart should add `--save_every 50` so the held-out trajectory can be plotted across the run.

---

## 6. Layout

```
src/dpo_longlive/                # trainer; dpo + redmd loops
scripts/run_all.py               # main entrypoint; one batch of 3 DPO + 3 Re-DMD runs
scripts/run_ablation_betalr.py   # β × lr sweep on MQ
scripts/eval_specific.py         # eval a list of checkpoints on the 20-prompt held-out
scripts/plot_mq_training.py      # v3 dpo_MQ windowed vs v2 per-step training plot
scripts/plot_ta_live.py          # parse the in-progress dpo_TA log into a 4-panel plot
paper/build_tex.py               # SINGLE SOURCE for paper/main.tex; reads runs/ + paper/figs
paper/neurips_2024.sty           # official NeurIPS 2024 template
runs/main_v2/                    # v2 30-step batch (full eval)
  eval_summary.json              # base + 6 runs × 4 dims
  training_metrics.json          # per-step training for 6 runs
runs/main_v3/                    # v3 400-step batch; only dpo_MQ + base eval
  dpo_MQ/policy_final.pt         # 5.3 GB
  dpo_MQ/training_summary.json   # windowed averages only (no per-step)
  eval/summary.json              # base + dpo_MQ
runs/abl_betalr/all_logs.json    # β×lr ablation (12 configs)
data/pairs/index.jsonl           # 851 paired prompts
data/eval_prompts.jsonl          # 20 held-out prompts
assets/longlive/models/longlive_base.pt  # 5.3 GB base checkpoint
logs/main_v3.log                 # most recent training log
```

---

## 7. Paper compile

`python3 paper/build_tex.py` writes `paper/main.tex` and recompiles `paper/main.pdf` (148–168 KB depending on figure set). Two pdflatex passes resolve refs + bibliography. Reads:

- `runs/main_v2/eval_summary.json` — Table 3 main eval (v2 base + 5 v2 runs)
- `runs/main_v3/eval/summary.json` — v3 dpo_MQ row overlaid on the main table (deltas recomputed vs v2 base so columns are consistent)
- `runs/main_v3/dpo_MQ/training_summary.json` — Table 5 v3 dpo_MQ windowed training metrics
- `runs/main_v2/training_metrics.json` — Figures 1, 2 (v2 30-step DPO + Re-DMD dynamics)
- `paper/figs/mq_training.png` — Figure 3 (v3 dpo_MQ windowed vs v2 dpo_MQ per-step)
- `runs/abl_betalr/all_logs.json` — Table 2 β×lr ablation

Edit `build_tex.py`, NOT the generated `main.tex`. The build pulls TeXLive from `/tmp/tex_install/root` via env vars (`TEXMFROOT`, `TEXMFCNF`, `TEXMFSYSVAR`, `TEXMFSYSCONFIG`).

**Discussion paragraph** reflects actual numbers: dpo_MQ v2 is the only Overall-positive DPO run; the v3 picked recipe over-fits; all Re-DMD baselines regress on Overall.

---

## 8. Drive upload — MCP limit

A single `mcp__claude_ai_Google_Drive__create_file` call inlines at most **~25–30 KB of `base64Content` / `textContent`** before the assistant output-token budget truncates the string. The truncated string still uploads, producing a corrupted file. Fingerprint from 2026-05-11: 152 KB PDF (200 KB b64) became a 19,499-byte Drive file (~13% of the original).

- `paper/main.pdf` cannot be uploaded in one call. Upload the LaTeX source (`paper/main.tex` is 25 KB) instead, or split base64 into multiple Drive text files for the user to `cat | base64 -d`.
- Current Drive state: `dpo-longlive-paper.tex` is the full source (good); `dpo-longlive-paper.pdf` is the 19.5 KB corrupted partial — user must delete via the Drive UI (MCP exposes no delete).
- GhostScript compression does NOT help — the paper PDF is already well-compressed and `/screen` and friends make it bigger.

---

## 9. Things to NOT do

- **No LoRA** — full-DiT post-training is the design.
- **No sink-EMA on Re-DMD** — `ema_weight=0` is what makes the apples-to-apples claim hold.
- **No cu124 PyTorch** — Blackwell sm_120 is unsupported there.
- **No 400+ steps on the v3 recipe without `--save_every 50`** — over-fits the 473-pair filtered pool.
- **No long training without `PYTHONUNBUFFERED=1`** — Python pipe-buffer hides progress for 10+ minutes.
- **No retry of full-PDF Drive upload via inline base64** — it always truncates; see §8.
- **No git config edits from the assistant** — user runs `git config ...` themselves.

---

## 10. Style preferences

- The user prefers honest paper text that matches the eval table (don't claim "every run improved" when only one did).
- For training jobs that show held-out regression, the user prefers killing the job and iterating rather than waiting for completion.
- Drive uploads are for finished artefacts only; in-progress work stays local.
