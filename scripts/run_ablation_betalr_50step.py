#!/usr/bin/env python
"""50-step (β, lr) ablation for chunk-wise DPO on MQ.

Goal (per user 2026-05-11): find the (β, lr) combination that gives the highest
held-out DPO margin at a v2-style clean recipe (no anchor, no min_gap filter,
batch 1, grad_accum 1) and a 50-step horizon. Includes β=10 to test the
user's hypothesis that the prior β=500 picked at 20 steps is too aggressive.

Each config:
  1. reset policy to LongLive base
  2. train 50 steps on shuffled MQ pairs
  3. eval held-out DPO acc/margin on the 20 cached held-out pairs from
     runs/main_v3/eval_dpo/ (no fresh generation needed; ~3 min/config)
  4. save train log + held-out metrics; delete in-memory policy state

Output: runs/abl_betalr_50/{beta}_{lr}.json per config + all_logs.json aggregate.
"""
from __future__ import annotations
import os, sys, time, json, gc, random, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_index", default="data/pairs/index.jsonl")
    ap.add_argument("--eval_pairs", default="runs/main_v3/eval_dpo/pairs.jsonl")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out_root", default="runs/abl_betalr_50")
    ap.add_argument("--betas", default="10,100,500,2000")
    ap.add_argument("--lrs", default="5e-6,2e-5")
    ap.add_argument("--eval_t_samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda"; dtype = torch.bfloat16
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    from dpo_longlive.path_setup import add_longlive_to_path, ensure_wan_models_symlink
    add_longlive_to_path(); ensure_wan_models_symlink()

    from dpo_longlive.dpo_trainer import (
        load_policy_and_ref, load_text_encoder_and_vae, chunkwise_dpo_step,
    )
    from eval_dpo_holdout import chunkwise_dpo_eval

    # ----- Training-pair rows (MQ only) -----
    train_rows = []
    with open(args.pairs_index) as f:
        for line in f:
            row = json.loads(line)
            if "MQ" in row.get("pairs", {}):
                p = row["pairs"]["MQ"]
                train_rows.append({
                    "prompt": row["prompt"],
                    "chosen":  p["chosen_latent"],
                    "rejected": p["rejected_latent"],
                })
    print(f"[data] {len(train_rows)} MQ training pairs (no min_gap filter)")

    # ----- Held-out pair rows (cached from this morning's eval_dpo run) -----
    eval_pairs = []
    with open(args.eval_pairs) as f:
        for line in f:
            eval_pairs.append(json.loads(line))
    print(f"[data] {len(eval_pairs)} held-out pairs cached at {args.eval_pairs}")

    # ----- Encode all unique prompts (train + eval) once -----
    train_prompts = sorted(set(r["prompt"] for r in train_rows))
    eval_prompts  = sorted(set(p["prompt"] for p in eval_pairs))
    all_prompts = sorted(set(train_prompts) | set(eval_prompts))
    print(f"[init] loading TE; encoding {len(all_prompts)} unique prompts ...")
    t0 = time.time()
    te, _ = load_text_encoder_and_vae(device=device, dtype=dtype)
    prompt_emb = {}
    with torch.no_grad():
        for pp in all_prompts:
            d = te(text_prompts=[pp])
            prompt_emb[pp] = {k: v.detach().to("cpu") for k, v in d.items()}
    del te; gc.collect(); torch.cuda.empty_cache()
    print(f"[init] TE done in {time.time()-t0:.1f}s")

    # ----- Load policy + ref once; save initial state -----
    print("[init] loading policy + ref ...")
    t0 = time.time()
    policy, ref = load_policy_and_ref(device=device, dtype=dtype)
    if hasattr(policy, "enable_gradient_checkpointing"):
        policy.enable_gradient_checkpointing()
    print(f"[init] policy+ref ready in {time.time()-t0:.1f}s")

    scheduler = policy.get_scheduler()
    print("[init] caching initial policy state on CPU ...")
    initial_state = {k: v.detach().to("cpu", dtype=dtype) for k, v in policy.state_dict().items()}

    def reset_policy():
        policy.load_state_dict(
            {k: v.to(device=device, dtype=dtype) for k, v in initial_state.items()},
            strict=True,
        )

    def cycle(xs):
        while True:
            for x in xs: yield x

    betas = [float(b) for b in args.betas.split(",")]
    lrs   = [float(s) for s in args.lrs.split(",")]
    grid = [(b, l) for b in betas for l in lrs]
    print(f"[grid] {len(grid)} configs: betas={betas}, lrs={lrs}, steps={args.steps}")

    all_results = {}
    rng = random.Random(args.seed)

    for idx, (beta, lr) in enumerate(grid):
        name = f"beta{beta:g}_lr{lr:g}"
        print(f"\n========== {idx+1}/{len(grid)}: {name} ==========")
        t_cfg = time.time()
        reset_policy(); gc.collect(); torch.cuda.empty_cache()
        policy.train()
        for p in policy.parameters():
            p.requires_grad_(True)

        params = [p for p in policy.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.01)

        local = train_rows.copy(); rng.shuffle(local); it = cycle(local)

        # ----- Training -----
        train_log = []
        torch.manual_seed(args.seed + idx); random.seed(args.seed + idx)
        opt.zero_grad(set_to_none=True)
        for step in range(1, args.steps + 1):
            pair = next(it)
            cond = {k: v.to(device=device, dtype=dtype) for k, v in prompt_emb[pair["prompt"]].items()}
            cw = torch.load(pair["chosen"], map_location=device, weights_only=False).to(dtype)
            rj = torch.load(pair["rejected"], map_location=device, weights_only=False).to(dtype)
            if cw.dim() == 4: cw = cw.unsqueeze(0)
            if rj.dim() == 4: rj = rj.unsqueeze(0)
            _, m = chunkwise_dpo_step(
                policy, ref, scheduler,
                chosen_lat=cw, rejected_lat=rj, cond=cond,
                num_frame_per_block=3, beta=beta,
                timesteps_grid=(1000, 750, 500, 250),
                dtype=dtype, loss_scale=1.0, anchor_alpha=0.0,
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            m["grad_norm"] = float(grad_norm.detach().item())
            m["step"] = step
            train_log.append(m)
            if step % 5 == 0 or step <= 3:
                print(f"  [train {step:3d}/{args.steps}] loss={m['dpo_loss']:.4f} margin={m['dpo_margin']:+.4f} acc={m['dpo_accuracy']:.2f} grad={m['grad_norm']:.1f}")

        # ----- Held-out DPO eval (no backward) -----
        print(f"  [eval] held-out DPO acc/margin on {len(eval_pairs)} pairs ...")
        policy.eval()
        for p in policy.parameters():
            p.requires_grad_(False)
        torch.manual_seed(0); random.seed(0)
        per_pair = []
        for ep in eval_pairs:
            cond = {k: v.to(device=device, dtype=dtype) for k, v in prompt_emb[ep["prompt"]].items()}
            cw = torch.load(ep["chosen_latent"], map_location=device, weights_only=False).to(dtype)
            rj = torch.load(ep["rejected_latent"], map_location=device, weights_only=False).to(dtype)
            if cw.dim() == 4: cw = cw.unsqueeze(0)
            if rj.dim() == 4: rj = rj.unsqueeze(0)
            samples = []
            for k in range(args.eval_t_samples):
                em = chunkwise_dpo_eval(
                    policy, ref, scheduler,
                    chosen_lat=cw, rejected_lat=rj, cond=cond,
                    num_frame_per_block=3, beta=beta,
                    timesteps_grid=(1000, 750, 500, 250), dtype=dtype,
                )
                samples.append(em)
            agg = {k: statistics.mean([s[k] for s in samples]) for k in samples[0]}
            agg["id"] = ep["id"]
            agg["gap"] = ep["chosen_score"] - ep["rejected_score"]
            per_pair.append(agg)
            del cw, rj
        accs = [r["dpo_accuracy"] for r in per_pair]
        margins = [r["dpo_margin"] for r in per_pair]
        losses = [r["dpo_loss"] for r in per_pair]
        eval_summary = {
            "n_pairs": len(per_pair),
            "dpo_accuracy_mean": statistics.mean(accs),
            "dpo_accuracy_std":  statistics.pstdev(accs),
            "dpo_margin_mean":   statistics.mean(margins),
            "dpo_margin_std":    statistics.pstdev(margins),
            "dpo_loss_mean":     statistics.mean(losses),
            "dpo_loss_std":      statistics.pstdev(losses),
            "win_rate_margin>0": sum(1 for m in margins if m > 0) / max(1, len(margins)),
        }
        # Train final-window (last 10 steps)
        tail = train_log[-10:]
        train_final = {
            "loss":   statistics.mean(m["dpo_loss"] for m in tail),
            "margin": statistics.mean(m["dpo_margin"] for m in tail),
            "acc":    statistics.mean(m["dpo_accuracy"] for m in tail),
            "grad":   statistics.mean(m["grad_norm"] for m in tail),
        }

        cfg_result = {
            "beta": beta, "lr": lr,
            "train_log": train_log,
            "train_final_window_10": train_final,
            "eval_summary": eval_summary,
            "eval_per_pair": per_pair,
            "wall_time_sec": time.time() - t_cfg,
        }
        with (out_root / f"{name}.json").open("w") as f:
            json.dump(cfg_result, f, indent=2)
        all_results[name] = {"beta": beta, "lr": lr,
                             "train_final_window_10": train_final,
                             "eval_summary": eval_summary,
                             "wall_time_sec": cfg_result["wall_time_sec"]}
        print(f"  [done] train final acc={train_final['acc']:.2f} margin={train_final['margin']:+.3f} "
              f"| eval acc={eval_summary['dpo_accuracy_mean']:.2f}±{eval_summary['dpo_accuracy_std']:.2f} "
              f"margin={eval_summary['dpo_margin_mean']:+.3f}±{eval_summary['dpo_margin_std']:.2f} "
              f"win={eval_summary['win_rate_margin>0']:.2f}  ({cfg_result['wall_time_sec']/60:.1f}m)")

    # ----- Aggregate -----
    with (out_root / "all_logs.json").open("w") as f:
        json.dump(all_results, f, indent=2)
    print("\n========== ranked by held-out DPO margin ==========")
    ranked = sorted(all_results.items(),
                    key=lambda kv: kv[1]["eval_summary"]["dpo_margin_mean"], reverse=True)
    print(f"{'config':>22}  {'eval margin':>14}  {'eval acc':>10}  {'eval win':>10}  {'train margin':>14}  {'train acc':>10}")
    for name, r in ranked:
        es = r["eval_summary"]; ts = r["train_final_window_10"]
        print(f"{name:>22}  {es['dpo_margin_mean']:+.3f}±{es['dpo_margin_std']:.2f}  "
              f"{es['dpo_accuracy_mean']:.2f}±{es['dpo_accuracy_std']:.2f}  "
              f"{es['win_rate_margin>0']:.2f}  "
              f"{ts['margin']:+.3f}        {ts['acc']:.2f}")
    print(f"\n[saved] {out_root}")


if __name__ == "__main__":
    main()
