#!/usr/bin/env python
"""Sanity check: load LongLive + VideoReward, generate one short video, score it."""
import sys, time, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

def main():
    from dpo_longlive.longlive_pipeline import load_pipeline, generate_one
    from dpo_longlive.reward_lib import load_reward, score_one

    print("[load] LongLive...")
    t0 = time.time()
    pipe = load_pipeline(device="cuda", dtype=torch.bfloat16, seed=0)
    print(f"[load] LongLive {time.time()-t0:.1f}s")

    print("[load] VideoReward...")
    t0 = time.time()
    rwd = load_reward(device="cuda", dtype=torch.bfloat16)
    print(f"[load] VideoReward {time.time()-t0:.1f}s")

    prompt = "A close-up of a steaming cup of coffee on a wooden table next to a window with morning light streaming in."
    print(f"[gen] prompt={prompt[:60]}...")
    t0 = time.time()
    pixels, latent = generate_one(pipe, prompt, seed=0)
    torch.cuda.synchronize()
    print(f"[gen] pixels={tuple(pixels.shape)} latent={tuple(latent.shape)} in {time.time()-t0:.1f}s")

    print("[score] reward...")
    t0 = time.time()
    sc = score_one(rwd, pixels, prompt, use_norm=True)
    print(f"[score] {sc} in {time.time()-t0:.1f}s")

    free, total = torch.cuda.mem_get_info(0)
    print(f"[mem] {(total-free)/1e9:.1f}/{total/1e9:.1f} GB used")

if __name__ == "__main__":
    main()
