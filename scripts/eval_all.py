#!/usr/bin/env python
"""Single-process eval of all checkpoints in runs/main/{dpo,redmd}_* + base."""
import json, os, sys, time, gc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/eval_prompts.jsonl")
    ap.add_argument("--runs_dir", default="runs/main")
    ap.add_argument("--out_dir", default="runs/main/eval")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = Path(args.runs_dir)

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

    # Snapshot the base policy state
    base_state_cpu = {k: v.detach().to("cpu") for k, v in pipe.generator.state_dict().items()}

    runs = [("base", None)]
    for sub in sorted(runs_dir.iterdir()):
        if not sub.is_dir(): continue
        if sub.name in ("eval",): continue
        ckpt = sub / "policy_final.pt"
        if not ckpt.exists():
            ckpt = next(sub.glob("policy_step*.pt"), None)
        if ckpt is None:
            print(f"[skip] {sub.name}: no ckpt")
            continue
        runs.append((sub.name, str(ckpt)))

    summary = {}
    for name, ckpt in runs:
        out_path = out_dir / f"{name}.jsonl"
        if out_path.exists():
            print(f"[skip] {name} (already evaluated)")
        else:
            print(f"[eval] {name} from {ckpt or 'base'}")
            if ckpt:
                state = torch.load(ckpt, map_location="cpu", weights_only=False)
                if "generator" in state: state = state["generator"]
                elif "model" in state: state = state["model"]
                # state is the full WanDiffusionWrapper state_dict; load
                pipe.generator.load_state_dict(state, strict=True)
            else:
                pipe.generator.load_state_dict(
                    {k: v.to("cuda") for k, v in base_state_cpu.items()}, strict=True
                )
            pipe.generator.to(device="cuda", dtype=torch.bfloat16)
            with out_path.open("w") as fout:
                for prow in eval_prompts:
                    pid, prompt = prow["id"], prow["prompt"]
                    seed = 42 + 17 * pid
                    pix, _ = generate_one(pipe, prompt, seed=seed)
                    sc = score_one(rwd, pix, prompt, use_norm=True)
                    row = {"name": name, "id": pid, "seed": seed, "scores": sc}
                    fout.write(json.dumps(row) + "\n")
                    fout.flush()
                    print(f"    [{pid:3d}] VQ={sc['VQ']:+.3f} MQ={sc['MQ']:+.3f} TA={sc['TA']:+.3f} O={sc['Overall']:+.3f}")

        rows = []
        with out_path.open() as fin:
            for line in fin: rows.append(json.loads(line))
        means = {k: sum(r["scores"][k] for r in rows) / max(1, len(rows)) for k in ["VQ", "MQ", "TA", "Overall"]}
        summary[name] = {"means": means, "n": len(rows)}
        print(f"  [agg] {name}: {means}")
        gc.collect(); torch.cuda.empty_cache()

    # Add per-prompt comparisons (delta vs base)
    base_rows = {}
    base_path = out_dir / "base.jsonl"
    if base_path.exists():
        with base_path.open() as f:
            for line in f:
                row = json.loads(line)
                base_rows[row["id"]] = row

    for name, rows_path in [(name, out_dir / f"{name}.jsonl") for name, _ in runs]:
        if name == "base": continue
        with open(rows_path) as f:
            rows = [json.loads(line) for line in f]
        base_means = summary["base"]["means"]
        deltas = {k: summary[name]["means"][k] - base_means[k] for k in ["VQ", "MQ", "TA", "Overall"]}
        wins = {k: 0 for k in ["VQ", "MQ", "TA", "Overall"]}
        n = 0
        for r in rows:
            if r["id"] not in base_rows: continue
            n += 1
            for k in ["VQ", "MQ", "TA", "Overall"]:
                if r["scores"][k] > base_rows[r["id"]]["scores"][k]:
                    wins[k] += 1
        wins = {k: v / max(1, n) for k, v in wins.items()}
        summary[name]["deltas_vs_base"] = deltas
        summary[name]["winrate_vs_base"] = wins
    out_summary = runs_dir / "eval_summary.json"
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[saved] {out_summary}")
    print("\n=== SUMMARY ===")
    for name, info in summary.items():
        print(f"  {name}: means={info['means']} | wr={info.get('winrate_vs_base', {})}")


if __name__ == "__main__":
    main()
