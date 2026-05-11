# DPO for Block-Causal Autoregressive Video Generation

This repository contains the code and artifacts for a feasibility study of
**chunk-wise teacher-forcing DPO** applied to NVlabs/LongLive-1.3B,
compared against a chunk-wise offline reward-DMD baseline (no sink-EMA).
All experiments use the [VideoAlign](https://github.com/KwaiVGI/VideoAlign)
reward model on three reward heads (Visual Quality, Motion Quality, Text Alignment).

The output paper is at [`paper/paper.pdf`](paper/paper.pdf).

## Layout

```
src/dpo_longlive/
  longlive_pipeline.py     # thin wrapper around LongLive's CausalInferencePipeline
  reward_lib.py            # VideoAlign loader + score-from-tensor helper
  dpo_trainer.py           # chunk-wise teacher-forcing DPO trainer
  redmd_trainer.py         # offline reward-DMD trainer (no sink-EMA)
  path_setup.py            # adds LongLive + Reward-Forcing dirs to sys.path

scripts/
  prepare_prompts.py       # sample 1000 train + 20 eval prompts from VidProm
  gen_pairs.py             # 2 rollouts per prompt → score → build (chosen, rejected) pairs
  run_all.py               # single-process runner: ablation OR main training
  eval_all.py              # eval base + 6 ckpts on 20 prompts, write JSONL + summary
  parse_log_metrics.py     # backup parser if training_metrics.json gets clobbered
  run_ablation.sh          # bash wrapper for the lr ablation
  run_main.sh              # bash wrapper for the 3 DPO + 3 Re-DMD main runs
  run_eval.sh              # bash wrapper for the eval pipeline

paper/
  build_pdf.py             # render the paper from runs/* metrics → paper.pdf via WeasyPrint
  paper.pdf                # the final paper
  paper.html               # HTML source
  figs/                    # dynamics plots
  references.bib           # BibTeX

assets/
  longlive/                # LongLive base ckpt + VidProm prompts (downloaded from HF)
  wan_models/Wan2.1-T2V-1.3B/  # Wan2.1 base diffusion model + text encoder + VAE
  videoreward/             # VideoAlign reward model

LongLive/                  # cloned from https://github.com/NVlabs/LongLive (Apache-2.0)
data/                      # train_prompts.jsonl, eval_prompts.jsonl, pairs/
runs/                      # ablation + main runs + eval results
logs/                      # per-run logs
```

## How to reproduce (single 96 GB GPU)

```bash
# env
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install transformers==4.49.0 diffusers==0.31.0 accelerate peft \
  omegaconf einops imageio imageio-ffmpeg av==13.1.0 opencv-python decord \
  safetensors huggingface_hub[cli] wandb sentencepiece pandas scipy scikit-image \
  Pillow "numpy<2" tqdm easydict matplotlib qwen-vl-utils datasets timm \
  trl==0.12.2 weasyprint
uv pip install flash-attn --no-build-isolation

# data
python scripts/prepare_prompts.py
python scripts/gen_pairs.py --max_prompts 60 --keep_mp4

# ablation (lr sweep on MQ)
PYTHONUNBUFFERED=1 PYTHONPATH=src:$PYTHONPATH python scripts/run_all.py \
  --pairs_index data/pairs/index.jsonl --max_pair_count 60 \
  --steps 10 --ablation --no_eval --out_root runs/abl

# main (3 DPO + 3 Re-DMD)
PYTHONUNBUFFERED=1 PYTHONPATH=src:$PYTHONPATH python scripts/run_all.py \
  --pairs_index data/pairs/index.jsonl --max_pair_count 60 \
  --steps 30 --lr 1e-6 --beta_dpo 5000 --beta_redmd 2.0 \
  --reward_dims MQ,TA,VQ --methods dpo,redmd \
  --no_eval --out_root runs/main

# eval
PYTHONUNBUFFERED=1 PYTHONPATH=src:$PYTHONPATH python scripts/eval_all.py \
  --runs_dir runs/main --out_dir runs/main/eval

# build paper
python paper/build_pdf.py
```

## Hardware constraints

These results were produced on a single NVIDIA RTX PRO 6000 Blackwell (96 GB,
sm_120). Owing to the wall-clock budget of the report, we cap data generation
at 60 paired prompts and training at 30 steps per run (vs. the 1 000-prompt /
400-step plan that the released scripts support). This is a feasibility study;
see Section 6 of the paper for the full set of suggested follow-ups.

## License

The training and inference code is MIT-licensed. The model weights inherit
their upstream licenses:

- LongLive-1.3B: CC-BY-NC-SA 4.0 (non-commercial)
- Wan2.1-T2V-1.3B: Apache 2.0
- VideoReward (KwaiVGI): Apache 2.0
