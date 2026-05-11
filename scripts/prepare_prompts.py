#!/usr/bin/env python
"""Sample 1000 train + 20 eval prompts from VidProm-filtered (LongLive's distribution)."""
import argparse, json, random, os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="assets/longlive/prompts/vidprom_filtered_extended.txt")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--n_eval", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20251010)
    args = ap.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with src.open() as f:
        prompts = [line.strip() for line in f if line.strip()]
    print(f"loaded {len(prompts)} prompts from {src}")

    rng = random.Random(args.seed)
    sample = rng.sample(prompts, args.n_train + args.n_eval)
    train, evalp = sample[: args.n_train], sample[args.n_train :]

    def dump(rows, path):
        with open(path, "w") as fo:
            for i, p in enumerate(rows):
                fo.write(json.dumps({"id": i, "prompt": p}) + "\n")

    dump(train, out_dir / "train_prompts.jsonl")
    dump(evalp, out_dir / "eval_prompts.jsonl")
    print(f"wrote {len(train)} train -> {out_dir/'train_prompts.jsonl'}")
    print(f"wrote {len(evalp)} eval  -> {out_dir/'eval_prompts.jsonl'}")


if __name__ == "__main__":
    main()
