#!/usr/bin/env python
"""Evaluate a (possibly DPO-tuned) LongLive checkpoint on the eval prompt set.

For each eval prompt, generate one video (single seed), score with VideoAlign on all
three reward dims (VQ, MQ, TA, plus Overall). Compare against a per-prompt baseline
score (the base model's score on the same prompt+seed) and emit win-rate plus mean
deltas.
"""
import argparse, json, sys, time, os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_checkpoint_into_pipeline(pipe, ckpt_path: str):
    """Load DiT weights from a DPO/Re-DMD-trained checkpoint into pipe.generator."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "generator" in state: state = state["generator"]
    elif "model" in state: state = state["model"]
    # state could be wrapper.state_dict() (so contains 'model.X') or just inner.
    # Try both.
    g = pipe.generator
    try:
        g.load_state_dict(state, strict=True)
    except RuntimeError:
        # try stripping a 'model.' prefix
        cleaned = {}
        for k, v in state.items():
            if k.startswith("model."):
                cleaned[k[len("model."):]] = v
            else:
                cleaned[k] = v
        g.model.load_state_dict(cleaned, strict=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/eval_prompts.jsonl")
    ap.add_argument("--ckpt", default=None, help="Path to DPO/Re-DMD policy state dict; if omitted, use base LongLive.")
    ap.add_argument("--out", default="runs/eval/base.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default="base")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from dpo_longlive.longlive_pipeline import load_pipeline, generate_one
    from dpo_longlive.reward_lib import load_reward, score_one

    print("[load] LongLive ...")
    pipe = load_pipeline(device="cuda", dtype=torch.bfloat16, seed=args.seed)
    if args.ckpt:
        print(f"[load] override generator from {args.ckpt}")
        load_checkpoint_into_pipeline(pipe, args.ckpt)
        pipe = pipe.to(torch.bfloat16)
        for p in pipe.parameters():
            p.requires_grad_(False)
        pipe.eval()

    print("[load] VideoReward ...")
    rwd = load_reward(device="cuda", dtype=torch.bfloat16)

    prompts = []
    with open(args.prompts) as f:
        for line in f:
            prompts.append(json.loads(line))

    fout = out_path.open("w")
    rows = []
    for prow in prompts:
        pid, prompt = prow["id"], prow["prompt"]
        seed = args.seed + 17 * pid
        t0 = time.time()
        pixels, _ = generate_one(pipe, prompt, seed=seed)
        sc = score_one(rwd, pixels, prompt, use_norm=True)
        row = {"name": args.name, "id": pid, "seed": seed, "scores": sc, "elapsed_s": round(time.time() - t0, 1)}
        rows.append(row)
        fout.write(json.dumps(row) + "\n")
        fout.flush()
        print(f"  [{pid:3d}] VQ={sc['VQ']:+.3f} MQ={sc['MQ']:+.3f} TA={sc['TA']:+.3f} Overall={sc['Overall']:+.3f}")

    fout.close()

    # Aggregate
    means = {k: sum(r["scores"][k] for r in rows) / len(rows) for k in ["VQ", "MQ", "TA", "Overall"]}
    print(f"[agg] {args.name}: mean reward = {means}")


if __name__ == "__main__":
    main()
