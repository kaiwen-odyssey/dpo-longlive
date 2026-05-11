#!/usr/bin/env python
"""Sweep over (β, lr) for chunk-wise DPO on the MQ reward head.

Goal: find a (β, lr) that gives positive margin AND bounded gradient norm AND
falling loss. The previous β=5000 grid (run_all.py --ablation) showed positive
margin at lr=1e-6 but with grad_norm ≈ 2k–9k; the user has asked us to reduce
β and bump lr to see whether the gradient blow-up softens while keeping margin
positive.
"""
import os, sys, time, json, gc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_index", default="data/pairs/index.jsonl")
    ap.add_argument("--max_pair_count", type=int, default=58)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out_root", default="runs/abl_betalr")
    ap.add_argument("--betas", default="50,500,5000")
    ap.add_argument("--lrs", default="1e-6,5e-6,2e-5")
    args = ap.parse_args()

    device = "cuda"; dtype = torch.bfloat16
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    from dpo_longlive.path_setup import LONGLIVE_MODELS_DIR, add_longlive_to_path, ensure_wan_models_symlink
    add_longlive_to_path(); ensure_wan_models_symlink()

    from dpo_longlive.dpo_trainer import (
        load_policy_and_ref, load_text_encoder_and_vae, chunkwise_dpo_step,
    )

    # Load training-pair rows
    rows = []
    with open(args.pairs_index) as f:
        for line in f:
            rows.append(json.loads(line))
    rows = rows[: args.max_pair_count]
    pairs = []
    for row in rows:
        if "MQ" in row.get("pairs", {}):
            p = row["pairs"]["MQ"]
            pairs.append({"prompt": row["prompt"],
                          "chosen":  p["chosen_latent"],
                          "rejected": p["rejected_latent"]})
    print(f"[data] {len(pairs)} pairs (MQ)")
    if not pairs:
        raise SystemExit("no MQ pairs")

    # Encode all unique prompts once.
    prompts = sorted(set(p["prompt"] for p in pairs))
    print(f"[init] loading text encoder; encoding {len(prompts)} prompts ...")
    t0 = time.time()
    te, _ = load_text_encoder_and_vae(device=device, dtype=dtype)
    prompt_emb = {}
    with torch.no_grad():
        for pp in prompts:
            d = te(text_prompts=[pp])
            prompt_emb[pp] = {k: v.detach().to("cpu") for k, v in d.items()}
    del te; gc.collect(); torch.cuda.empty_cache()
    print(f"[init] TE done in {time.time()-t0:.1f}s")

    # Load policy + ref once.
    print("[init] loading policy + ref ...")
    t0 = time.time()
    policy, ref = load_policy_and_ref(device=device, dtype=dtype)
    print(f"[init] policy+ref ready in {time.time()-t0:.1f}s")

    scheduler = policy.get_scheduler()
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
    print(f"[grid] betas={betas}  lrs={lrs}  steps={args.steps}")

    all_logs = {}
    import random
    rng = random.Random(0)

    for beta in betas:
        for lr in lrs:
            name = f"beta{beta:g}_lr{lr:g}"
            print(f"\n========== {name} ==========")
            reset_policy(); gc.collect(); torch.cuda.empty_cache()

            params = [p for p in policy.model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.01)

            local = pairs.copy(); rng.shuffle(local); it = cycle(local)
            log = []
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
                    dtype=dtype, loss_scale=1.0,
                )
                grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                m["grad_norm"] = float(grad_norm.detach().item())
                m["step"] = step
                free, total = torch.cuda.mem_get_info(0)
                m["mem_gb"] = (total - free) / 1e9
                log.append(m)
                if step % 5 == 0 or step <= 3:
                    print(f"  [{step:4d}/{args.steps}] loss={m['dpo_loss']:.4f} "
                          f"margin={m['dpo_margin']:+.4f} acc={m['dpo_accuracy']:.2f} "
                          f"grad={m['grad_norm']:.1f} mem={m['mem_gb']:.1f}GB")
            all_logs[name] = log
            with (out_root / f"{name}.json").open("w") as f:
                json.dump(log, f, indent=2)
            # Save the final policy checkpoint for downstream eval.
            ckpt_dir = out_root / name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt = ckpt_dir / "policy_final.pt"
            torch.save(policy.state_dict(), ckpt)
            print(f"  [save] {ckpt}")

    with (out_root / "all_logs.json").open("w") as f:
        json.dump(all_logs, f, indent=2)
    print(f"\n[saved] {out_root}")


if __name__ == "__main__":
    main()
