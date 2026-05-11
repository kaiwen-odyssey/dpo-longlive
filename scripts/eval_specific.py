#!/usr/bin/env python
"""Eval a specific list of checkpoints on the held-out 20-prompt set."""
import json, os, sys, time, gc, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/eval_prompts.jsonl")
    ap.add_argument("--ckpts", required=True, help="Comma-separated list name=path,name2=path2 (use 'base' for the un-tuned LongLive)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    spec = []
    for kv in args.ckpts.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1); spec.append((k.strip(), v.strip() if v.strip() != 'base' else None))
        else:
            spec.append((kv.strip(), None))

    import torch
    from dpo_longlive.longlive_pipeline import load_pipeline, generate_one
    from dpo_longlive.reward_lib import load_reward, score_one

    eval_prompts = []
    with open(args.prompts) as f:
        for line in f:
            eval_prompts.append(json.loads(line))

    print("[load] pipeline + reward ...")
    t0 = time.time()
    pipe = load_pipeline(device="cuda", dtype=torch.bfloat16, seed=0)
    rwd = load_reward(device="cuda", dtype=torch.bfloat16)
    print(f"[load] ready in {time.time()-t0:.1f}s")

    base_state_cpu = {k: v.detach().to("cpu") for k, v in pipe.generator.state_dict().items()}

    summary = {}
    for name, ckpt in spec:
        out_path = out_dir / f"{name}.jsonl"
        if out_path.exists():
            print(f"[skip] {name}")
        else:
            print(f"[eval] {name} from {ckpt or 'base'}")
            if ckpt:
                state = torch.load(ckpt, map_location="cpu", weights_only=False)
                if "generator" in state: state = state["generator"]
                elif "model" in state: state = state["model"]
                pipe.generator.load_state_dict(state, strict=True)
            else:
                pipe.generator.load_state_dict(
                    {k: v.to("cuda") for k, v in base_state_cpu.items()}, strict=True,
                )
            pipe.generator.to(device="cuda", dtype=torch.bfloat16)
            with out_path.open("w") as fout:
                for prow in eval_prompts:
                    pid, prompt = prow["id"], prow["prompt"]
                    seed = 42 + 17 * pid
                    pix, _ = generate_one(pipe, prompt, seed=seed)
                    sc = score_one(rwd, pix, prompt, use_norm=True)
                    fout.write(json.dumps({"name": name, "id": pid, "seed": seed, "scores": sc}) + "\n")
                    fout.flush()
                    print(f"    [{pid:3d}] VQ={sc['VQ']:+.3f} MQ={sc['MQ']:+.3f} TA={sc['TA']:+.3f} O={sc['Overall']:+.3f}")

        rows = [json.loads(l) for l in open(out_path)]
        means = {k: sum(r["scores"][k] for r in rows) / max(1, len(rows)) for k in ["VQ","MQ","TA","Overall"]}
        summary[name] = {"means": means, "n": len(rows)}
        print(f"  [agg] {name}: {means}")
        gc.collect(); torch.cuda.empty_cache()

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[saved] {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
