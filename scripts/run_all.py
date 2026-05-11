#!/usr/bin/env python
"""Single-process runner: load policy/ref/text-encoder once, then run all training/eval
sequentially to avoid the 50 s LongLive load + 9 s VideoReward load per run.
"""
import os, sys, time, json, argparse, gc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch


def encode_prompts_once(text_encoder, prompts, device, dtype):
    out = {}
    seen = set()
    with torch.no_grad():
        for p in prompts:
            if p in seen: continue
            seen.add(p)
            d = text_encoder(text_prompts=[p])
            out[p] = {k: v.detach().to("cpu") for k, v in d.items()}
    return out


def fresh_policy_from_state(make_policy_fn, state_dict, device, dtype):
    pol = make_policy_fn()
    pol.load_state_dict(state_dict, strict=True)
    return pol.to(device=device, dtype=dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_index", default="data/pairs/index.jsonl")
    ap.add_argument("--max_pair_count", type=int, default=None)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta_dpo", type=float, default=5000.0)
    ap.add_argument("--beta_redmd", type=float, default=2.0)
    ap.add_argument("--reward_dims", default="MQ,TA,VQ")
    ap.add_argument("--methods", default="dpo,redmd")
    ap.add_argument("--out_root", default="runs")
    ap.add_argument("--eval_prompts", default="data/eval_prompts.jsonl")
    ap.add_argument("--no_eval", action="store_true")
    ap.add_argument("--wandb_mode", default="disabled")
    ap.add_argument("--ablation", action="store_true", help="run small lr sweep on MQ instead of main runs")
    ap.add_argument("--grad_accum", type=int, default=1,
                    help="number of pairs (samples) accumulated per optimizer.step()")
    ap.add_argument("--anchor_alpha", type=float, default=0.0,
                    help="if >0, add α · ||f_pol(x_w)-τ_w||² to each block's DPO loss "
                         "to prevent the chosen log-prob from sliding (only used for DPO).")
    ap.add_argument("--min_gap", type=float, default=0.0,
                    help="drop pairs whose chosen-rejected reward gap is below this threshold "
                         "on the *targeted* dim. Applied to both DPO and Re-DMD so they "
                         "train on the same prompt subset.")
    ap.add_argument("--save_every", type=int, default=0,
                    help="if >0, save policy_step{N}.pt every N opt-steps during training (DPO only).")
    args = ap.parse_args()

    device = "cuda"; dtype = torch.bfloat16
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    # Load LongLive base ckpt once for sharing across ref + initial policies.
    from dpo_longlive.path_setup import LONGLIVE_MODELS_DIR, add_longlive_to_path, ensure_wan_models_symlink
    add_longlive_to_path(); ensure_wan_models_symlink()
    base_ckpt_path = str(LONGLIVE_MODELS_DIR / "models" / "longlive_base.pt")
    base_state = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    if "generator" in base_state: base_state = base_state["generator"]
    elif "model" in base_state: base_state = base_state["model"]

    # Build a policy/ref factory we can call repeatedly to get a fresh init.
    from dpo_longlive.dpo_trainer import (
        load_policy_and_ref, load_text_encoder_and_vae,
        chunkwise_dpo_step, init_kv_cache, init_crossattn_cache,
    )
    from dpo_longlive.redmd_trainer import chunkwise_redmd_step, _load_all_samples

    # Step 1: encode prompts (training + eval) using the text encoder.
    pairs_path = args.pairs_index
    eval_prompts_list = []
    with open(args.eval_prompts) as f:
        for line in f:
            eval_prompts_list.append(json.loads(line))

    train_pair_rows = []
    with open(pairs_path) as f:
        for line in f:
            train_pair_rows.append(json.loads(line))
    if args.max_pair_count:
        train_pair_rows = train_pair_rows[: args.max_pair_count]

    all_prompts = sorted(set([r["prompt"] for r in train_pair_rows] + [p["prompt"] for p in eval_prompts_list]))
    print(f"[init] loading text encoder; encoding {len(all_prompts)} unique prompts ...")
    t0 = time.time()
    te, _ = load_text_encoder_and_vae(device=device, dtype=dtype)
    prompt_emb = encode_prompts_once(te, all_prompts, device, dtype)
    del te; gc.collect(); torch.cuda.empty_cache()
    print(f"[init] encoded {len(prompt_emb)} prompts in {time.time()-t0:.1f}s")

    # Step 2: load policy + ref (the ref will not change).
    print("[init] loading policy + ref ...")
    t0 = time.time()
    policy, ref = load_policy_and_ref(device=device, dtype=dtype)
    print(f"[init] policy+ref ready in {time.time()-t0:.1f}s")

    scheduler = policy.get_scheduler()

    # Convenience: snapshot the initial policy state in CPU memory so we can quickly reset
    # between runs without going through the disk again.
    initial_policy_cpu = {k: v.detach().to("cpu", dtype=dtype) for k, v in policy.state_dict().items()}

    def reset_policy():
        policy.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in initial_policy_cpu.items()},
                               strict=True)

    def reset_caches_etc():
        gc.collect(); torch.cuda.empty_cache()

    # Determine reward dims and methods
    reward_dims = [d for d in args.reward_dims.split(",") if d]
    methods = [m for m in args.methods.split(",") if m]

    # Step 3: training loops
    runs_done = []  # list of (name, ckpt_path)
    metric_log = {}  # name -> list of metrics

    if args.ablation:
        # Quick lr sweep on MQ
        lrs = [1e-7, 1e-6, 1e-5]
        for lr in lrs:
            name = f"ablation_dpo_MQ_lr{lr}"
            print(f"\n========== {name} ==========")
            reset_policy(); reset_caches_etc()
            log = _train_dpo_inner(
                policy, ref, scheduler, prompt_emb, train_pair_rows,
                reward_dim="MQ", lr=lr, beta=args.beta_dpo, max_steps=args.steps,
                device=device, dtype=dtype,
            )
            metric_log[name] = log
            ckpt_path = out_root / name / f"policy_final.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(policy.state_dict(), ckpt_path)
            runs_done.append((name, str(ckpt_path)))
    else:
        # Main runs
        for method in methods:
            for dim in reward_dims:
                name = f"{method}_{dim}"
                ckpt_path = out_root / name / "policy_final.pt"
                if ckpt_path.exists():
                    print(f"\n========== {name} ==========")
                    print(f"  [skip] {ckpt_path} already exists; not retraining.")
                    runs_done.append((name, str(ckpt_path)))
                    continue
                print(f"\n========== {name} ==========")
                reset_policy(); reset_caches_etc()
                if method == "dpo":
                    log = _train_dpo_inner(
                        policy, ref, scheduler, prompt_emb, train_pair_rows,
                        reward_dim=dim, lr=args.lr, beta=args.beta_dpo, max_steps=args.steps,
                        device=device, dtype=dtype, grad_accum=args.grad_accum,
                        anchor_alpha=args.anchor_alpha,
                        min_gap=args.min_gap,
                        save_every=args.save_every,
                        save_dir=out_root / name,
                    )
                elif method == "redmd":
                    samples = _load_all_samples(pairs_path, dim)
                    if args.max_pair_count:
                        samples = [s for s in samples if int(s["id"]) < args.max_pair_count]
                    if args.min_gap > 0:
                        # Keep only samples whose prompt has a clear chosen-vs-rejected gap on this dim.
                        keep_ids = set()
                        for r in train_pair_rows:
                            if dim in r.get("pairs", {}):
                                p = r["pairs"][dim]
                                if (p["chosen_score"] - p["rejected_score"]) > args.min_gap:
                                    keep_ids.add(r["id"])
                        before = len(samples)
                        samples = [s for s in samples if s["id"] in keep_ids]
                        print(f"  [min_gap={args.min_gap}] Re-DMD samples: {before} -> {len(samples)}")
                    log = _train_redmd_inner(
                        policy, ref, scheduler, prompt_emb, samples,
                        lr=args.lr, beta=args.beta_redmd, max_steps=args.steps,
                        device=device, dtype=dtype, grad_accum=args.grad_accum,
                    )
                else:
                    raise ValueError(method)
                metric_log[name] = log
                ckpt_path = out_root / name / f"policy_final.pt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(policy.state_dict(), ckpt_path)
                runs_done.append((name, str(ckpt_path)))

    # Save metric logs
    with open(out_root / "training_metrics.json", "w") as f:
        json.dump(metric_log, f, indent=2)
    print(f"[saved] {out_root}/training_metrics.json")

    # Step 4: optional eval
    if args.no_eval:
        return

    print("\n========== EVAL ==========")
    # Load reward model + VAE in addition (vae is needed for inference pixel decode; re-load).
    # Easiest: reuse the existing pipe via dpo_longlive.longlive_pipeline.
    from dpo_longlive.longlive_pipeline import load_pipeline, generate_one
    from dpo_longlive.reward_lib import load_reward, score_one

    # We need a pipeline (which has VAE) to decode for eval. The pipeline embeds its own
    # generator, but we'll override it from each ckpt during eval.
    print("[eval] loading inference pipeline ...")
    pipe = load_pipeline(device=device, dtype=dtype, seed=0)
    print("[eval] loading reward model ...")
    rwd = load_reward(device=device, dtype=dtype)

    eval_dir = out_root / "eval"; eval_dir.mkdir(parents=True, exist_ok=True)
    runs_to_eval = [("base", None)] + runs_done
    eval_summary = {}
    for name, ckpt in runs_to_eval:
        out_path = eval_dir / f"{name}.jsonl"
        if out_path.exists():
            print(f"[eval-skip] {name}")
        else:
            print(f"[eval] {name}")
            if ckpt:
                state = torch.load(ckpt, map_location="cpu", weights_only=False)
                if "generator" in state: state = state["generator"]
                elif "model" in state: state = state["model"]
                pipe.generator.load_state_dict(state, strict=True)
                pipe.generator.to(device=device, dtype=dtype)
            else:
                # base — reload from initial cpu state
                pipe.generator.load_state_dict(
                    {k: v.to(device=device, dtype=dtype) for k, v in initial_policy_cpu.items()},
                    strict=True,
                )
            with out_path.open("w") as fout:
                rows = []
                for prow in eval_prompts_list:
                    pid, prompt = prow["id"], prow["prompt"]
                    seed = 42 + 17 * pid
                    pix, _ = generate_one(pipe, prompt, seed=seed)
                    sc = score_one(rwd, pix, prompt, use_norm=True)
                    row = {"name": name, "id": pid, "seed": seed, "scores": sc}
                    rows.append(row)
                    fout.write(json.dumps(row) + "\n")
                    fout.flush()
                    print(f"    [{pid:3d}] {sc}")
        # aggregate
        rows = []
        with out_path.open() as fin:
            for line in fin: rows.append(json.loads(line))
        means = {k: sum(r["scores"][k] for r in rows) / max(1, len(rows)) for k in ["VQ", "MQ", "TA", "Overall"]}
        eval_summary[name] = {"means": means, "n": len(rows)}
        print(f"  [agg] {name}: {means}")

    with open(out_root / "eval_summary.json", "w") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"[saved] {out_root}/eval_summary.json")


def _train_dpo_inner(policy, ref, scheduler, prompt_emb, train_pair_rows, *,
                     reward_dim, lr, beta, max_steps, device, dtype, grad_accum=1,
                     anchor_alpha=0.0, min_gap=0.0, save_every=0, save_dir=None):
    from torch.optim import AdamW
    from dpo_longlive.dpo_trainer import chunkwise_dpo_step
    import random, statistics

    pairs = []
    for row in train_pair_rows:
        if reward_dim in row.get("pairs", {}):
            p = row["pairs"][reward_dim]
            gap = p["chosen_score"] - p["rejected_score"]
            if gap <= min_gap:
                continue
            pairs.append({
                "prompt": row["prompt"],
                "chosen": p["chosen_latent"],
                "rejected": p["rejected_latent"],
                "gap": gap,
            })
    if not pairs:
        return []
    rng = random.Random(0); rng.shuffle(pairs)
    print(f"  [data] {len(pairs)} pairs after min_gap={min_gap} filter "
          f"(grad_accum={grad_accum} → {grad_accum} pairs/optstep, "
          f"anchor_alpha={anchor_alpha})")

    params = [p for p in policy.model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.01)

    def cycle_iter(xs):
        while True:
            for x in xs: yield x
    it = cycle_iter(pairs)

    log = []
    opt.zero_grad(set_to_none=True)
    inv = 1.0 / float(grad_accum)
    for step in range(1, max_steps + 1):
        # Accumulate `grad_accum` pair forward+backwards before opt.step()
        acc_metrics = {"dpo_loss": [], "dpo_margin": [], "dpo_accuracy": [],
                       "chosen_logp_diff": [], "rejected_logp_diff": []}
        for _ in range(grad_accum):
            pair = next(it)
            cond = {k: v.to(device=device, dtype=dtype) for k, v in prompt_emb[pair["prompt"]].items()}
            chosen_lat = torch.load(pair["chosen"], map_location=device, weights_only=False).to(dtype)
            rejected_lat = torch.load(pair["rejected"], map_location=device, weights_only=False).to(dtype)
            if chosen_lat.dim() == 4: chosen_lat = chosen_lat.unsqueeze(0)
            if rejected_lat.dim() == 4: rejected_lat = rejected_lat.unsqueeze(0)
            _, m_one = chunkwise_dpo_step(
                policy, ref, scheduler,
                chosen_lat=chosen_lat, rejected_lat=rejected_lat,
                cond=cond, num_frame_per_block=3,
                beta=beta, timesteps_grid=(1000, 750, 500, 250),
                dtype=dtype, loss_scale=inv,   # average across the grad_accum pairs
                anchor_alpha=anchor_alpha,
            )
            for k in list(acc_metrics.keys()):
                if k in m_one: acc_metrics[k].append(m_one[k])
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        metrics = {k: statistics.mean(v) for k, v in acc_metrics.items() if v}
        metrics["grad_norm"] = float(grad_norm.detach().item())
        metrics["step"] = step
        free, total = torch.cuda.mem_get_info(0)
        metrics["mem_gb"] = (total - free) / 1e9
        log.append(metrics)
        if step % 5 == 0 or step <= 3:
            print(f"  [{step:4d}/{max_steps}] loss={metrics['dpo_loss']:.4f} margin={metrics['dpo_margin']:+.4f} "
                  f"acc={metrics['dpo_accuracy']:.2f} grad={metrics['grad_norm']:.1f} mem={metrics['mem_gb']:.1f}GB")
        if save_every and save_dir is not None and (step % save_every == 0) and step != max_steps:
            sp = save_dir / f"policy_step{step:04d}.pt"
            sp.parent.mkdir(parents=True, exist_ok=True)
            torch.save(policy.state_dict(), sp)
            print(f"  [save] {sp}")
    return log


def _train_redmd_inner(policy, ref, scheduler, prompt_emb, samples, *,
                       lr, beta, max_steps, device, dtype, grad_accum=1):
    from torch.optim import AdamW
    from dpo_longlive.redmd_trainer import chunkwise_redmd_step
    import random, statistics

    rng = random.Random(0); rng.shuffle(samples)
    print(f"  [data] {len(samples)} samples (grad_accum={grad_accum} → {grad_accum} samples/optstep)")

    params = [p for p in policy.model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.01)

    def cycle_iter(xs):
        while True:
            for x in xs: yield x
    it = cycle_iter(samples)

    log = []
    opt.zero_grad(set_to_none=True)
    inv = 1.0 / float(grad_accum)
    for step in range(1, max_steps + 1):
        acc = {"redmd_loss_pol": [], "redmd_loss_ref": [], "reward": [],
               "exp_beta_reward": [], "denoising_gap": []}
        for _ in range(grad_accum):
            s = next(it)
            cond = {k: v.to(device=device, dtype=dtype) for k, v in prompt_emb[s["prompt"]].items()}
            x0_lat = torch.load(s["latent"], map_location=device, weights_only=False).to(dtype)
            if x0_lat.dim() == 4: x0_lat = x0_lat.unsqueeze(0)
            _, m_one = chunkwise_redmd_step(
                policy, ref, scheduler, x0_lat=x0_lat, reward=s["reward"],
                cond=cond, num_frame_per_block=3,
                beta=beta, timesteps_grid=(1000, 750, 500, 250),
                dtype=dtype, loss_scale=inv,
            )
            for k in list(acc.keys()):
                if k in m_one: acc[k].append(m_one[k])
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        metrics = {k: statistics.mean(v) for k, v in acc.items() if v}
        metrics["grad_norm"] = float(grad_norm.detach().item())
        metrics["step"] = step
        free, total = torch.cuda.mem_get_info(0)
        metrics["mem_gb"] = (total - free) / 1e9
        log.append(metrics)
        if step % 5 == 0 or step <= 3:
            print(f"  [{step:4d}/{max_steps}] loss_pol={metrics['redmd_loss_pol']:.4f} reward={metrics['reward']:+.3f} "
                  f"w={metrics['exp_beta_reward']:.2f} grad={metrics['grad_norm']:.1f} mem={metrics['mem_gb']:.1f}GB")
    return log


if __name__ == "__main__":
    main()
