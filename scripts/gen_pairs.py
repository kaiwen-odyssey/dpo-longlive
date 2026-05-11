#!/usr/bin/env python
"""Generate K rollouts per prompt and reward-score each, then build (chosen, rejected) pairs per dim.

Each video: 5 sec @ 16 fps = 81 frames, 832x480, latent shape [1, 21, 16, 60, 104].
Saved per prompt:
  data/pairs/<id>/seed<k>.latent.pt        (BF16 latent of shape [1, 21, 16, 60, 104])
  data/pairs/<id>/seed<k>.mp4              (only if --keep_mp4)
  data/pairs/<id>/scores.json              {seed: {VQ, MQ, TA, Overall}}
And after all prompts:
  data/pairs/index.jsonl                   one row per prompt with chosen/rejected per reward dim
"""
import argparse, json, os, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/train_prompts.jsonl")
    ap.add_argument("--out_dir", default="data/pairs")
    ap.add_argument("--n_per_prompt", type=int, default=2)
    ap.add_argument("--seed_base", type=int, default=1000)
    ap.add_argument("--max_prompts", type=int, default=None)
    ap.add_argument("--keep_mp4", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"

    # Track which prompts have already been processed (resumable)
    processed = set()
    if index_path.exists():
        with index_path.open() as f:
            for line in f:
                row = json.loads(line)
                processed.add(row["id"])
        print(f"resume: {len(processed)} prompts already processed")

    # Load prompts
    prompts = []
    with open(args.prompts) as f:
        for line in f:
            prompts.append(json.loads(line))
    if args.max_prompts:
        prompts = prompts[: args.max_prompts]

    todo = [p for p in prompts if p["id"] not in processed]
    print(f"to do: {len(todo)} / {len(prompts)} prompts; n_per_prompt={args.n_per_prompt}")
    if not todo:
        return

    from dpo_longlive.longlive_pipeline import load_pipeline, generate_one
    from dpo_longlive.reward_lib import load_reward, score_one
    import imageio

    print("[load] LongLive pipeline...")
    t0 = time.time()
    pipe = load_pipeline(device=args.device, dtype=torch.bfloat16)
    print(f"[load] LongLive ready in {time.time()-t0:.1f}s")

    t0 = time.time()
    reward_inf = load_reward(device=args.device, dtype=torch.bfloat16)
    print(f"[load] VideoReward ready in {time.time()-t0:.1f}s")

    fout = index_path.open("a")

    for prow in todo:
        pid, prompt = prow["id"], prow["prompt"]
        pdir = out_dir / f"{pid:06d}"
        pdir.mkdir(parents=True, exist_ok=True)
        scores = {}
        latents = {}
        t_prompt = time.time()
        for k in range(args.n_per_prompt):
            seed = args.seed_base + 31 * pid + 7 * k
            t_gen = time.time()
            pixels, latent = generate_one(pipe, prompt, seed=seed)
            t_decode = time.time() - t_gen
            t_score = time.time()
            sc = score_one(reward_inf, pixels, prompt, use_norm=True)
            t_sc = time.time() - t_score
            scores[seed] = sc
            # save latent
            torch.save(latent.cpu().to(torch.bfloat16), pdir / f"seed{seed}.latent.pt")
            latents[seed] = latent.cpu()
            if args.keep_mp4 and k == 0:
                vid_uint8 = (pixels.clamp(0, 1).permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()
                imageio.mimwrite(pdir / f"seed{seed}.mp4", vid_uint8, fps=16, quality=6)
            print(f"  [{pid:5d}] seed={seed} {sc} | gen+decode {t_decode:.1f}s, score {t_sc:.1f}s")

        # save score json
        with (pdir / "scores.json").open("w") as f:
            json.dump({"prompt": prompt, "scores": scores}, f, indent=2)

        # build pairs per reward dim (need at least 2 rollouts)
        if len(scores) >= 2:
            seeds = list(scores.keys())
            # All pairwise combinations -> for each dim, choose argmax/argmin
            row = {"id": pid, "prompt": prompt, "seeds": seeds, "scores": scores, "pairs": {}}
            for dim in ["VQ", "MQ", "TA", "Overall"]:
                # rank by dim
                ranked = sorted(seeds, key=lambda s: scores[s][dim], reverse=True)
                chosen, rejected = ranked[0], ranked[-1]
                if scores[chosen][dim] == scores[rejected][dim]:
                    # tie -> skip this dim
                    continue
                row["pairs"][dim] = {
                    "chosen_seed": chosen,
                    "rejected_seed": rejected,
                    "chosen_score": scores[chosen][dim],
                    "rejected_score": scores[rejected][dim],
                    "chosen_latent": str(pdir / f"seed{chosen}.latent.pt"),
                    "rejected_latent": str(pdir / f"seed{rejected}.latent.pt"),
                }
            fout.write(json.dumps(row) + "\n")
            fout.flush()
        print(f"  [{pid:5d}] done in {time.time()-t_prompt:.1f}s")

    fout.close()
    print("done")


if __name__ == "__main__":
    main()
