#!/usr/bin/env python
"""Compute teacher-forced DPO accuracy and implicit margin on the 20 held-out prompts
under the v3 dpo_MQ policy (vs ref = LongLive base). This is the over-fitting diagnostic
for the v3 recipe — if training acc/margin >> held-out acc/margin, the policy has
memorised the training-pool pairs.

Phase 1: generate 2 base rollouts per held-out prompt (seeds 1000, 1007), score MQ.
Phase 2: form (chosen, rejected) pair per prompt by MQ; drop ties.
Phase 3: load policy (v3 dpo_MQ) + ref; for each pair, run K=4 random t-grid samples
         of the chunkwise teacher-forced forward (no backward) and average.
Phase 4: write per-pair metrics + aggregate mean/std to runs/main_v3/eval_dpo/.
"""
from __future__ import annotations
import os, sys, json, time, gc, math, random, argparse, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F

from dpo_longlive.dpo_trainer import (
    load_policy_and_ref,
    load_text_encoder_and_vae,
    init_kv_cache, init_crossattn_cache,
    add_noise_to_block,
    _FRAME_SEQ_LEN, _NUM_TRANSFORMER_BLOCKS,
)
from dpo_longlive.longlive_pipeline import load_pipeline, generate_one
from dpo_longlive.reward_lib import load_reward, score_one


@torch.no_grad()
def chunkwise_dpo_eval(
    policy, ref, scheduler, *,
    chosen_lat: torch.Tensor, rejected_lat: torch.Tensor,
    cond: dict, num_frame_per_block: int, beta: float,
    timesteps_grid: tuple, dtype, local_attn_size: int = 12,
    context_noise: int = 0,
):
    """No-backward variant of chunkwise_dpo_step. Returns metrics dict averaged across blocks.

    For each 3-frame block:
      - sample one t from the 4-step grid
      - add noise; forward policy and ref on noisy chosen + rejected
      - compute the DPO inside (= -0.5*beta*((e_pol_w-e_ref_w)-(e_pol_l-e_ref_l)))
    Then update KV caches with clean K/V for the next block (teacher forcing).
    """
    device = chosen_lat.device
    T_total = chosen_lat.shape[1]
    assert T_total % num_frame_per_block == 0
    num_blocks = T_total // num_frame_per_block

    kv_size = local_attn_size * _FRAME_SEQ_LEN
    kv = {
        ("pol", "w"): init_kv_cache(1, kv_size, dtype, device),
        ("pol", "l"): init_kv_cache(1, kv_size, dtype, device),
        ("ref", "w"): init_kv_cache(1, kv_size, dtype, device),
        ("ref", "l"): init_kv_cache(1, kv_size, dtype, device),
    }
    shared_cax = init_crossattn_cache(1, dtype, device)
    # Warm cross-attn cache once.
    warm_x = torch.zeros((1, num_frame_per_block, 16, chosen_lat.shape[-2], chosen_lat.shape[-1]),
                         dtype=dtype, device=device)
    warm_t = torch.zeros((1, num_frame_per_block), dtype=torch.long, device=device)
    ref(
        noisy_image_or_video=warm_x,
        conditional_dict=cond,
        timestep=warm_t,
        kv_cache=kv[("ref", "w")],
        crossattn_cache=shared_cax,
        current_start=0,
    )
    for layer in shared_cax:
        layer["is_init"] = True
    # Reset the dirty kv cache.
    for layer in kv[("ref", "w")]:
        layer["k"].zero_()
        layer["v"].zero_()
        layer["global_end_index"].zero_()
        layer["local_end_index"].zero_()
    cax = {"pol": shared_cax, "ref": shared_cax}

    losses, margins, accs, c_logp, r_logp = [], [], [], [], []

    for bi in range(num_blocks):
        s, e = bi * num_frame_per_block, (bi + 1) * num_frame_per_block
        x0_w = chosen_lat[:, s:e].contiguous()
        x0_l = rejected_lat[:, s:e].contiguous()

        t_idx = torch.randint(0, len(timesteps_grid), (1,)).item()
        t_val = int(timesteps_grid[t_idx])
        t = torch.tensor([t_val], device=device, dtype=torch.long)

        noise_w = torch.randn_like(x0_w)
        noise_l = torch.randn_like(x0_l)
        xt_w = add_noise_to_block(scheduler, x0_w, noise_w, t)
        xt_l = add_noise_to_block(scheduler, x0_l, noise_l, t)

        timestep_BT = torch.full((1, num_frame_per_block), t_val, device=device, dtype=torch.long)
        current_start = s * _FRAME_SEQ_LEN

        def kvfwd(model, xt_block, model_key, video_key):
            return model(
                noisy_image_or_video=xt_block,
                conditional_dict=cond,
                timestep=timestep_BT,
                kv_cache=kv[(model_key, video_key)],
                crossattn_cache=cax[model_key],
                current_start=current_start,
            )

        flow_pred_w_ref, _ = kvfwd(ref, xt_w, "ref", "w")
        flow_pred_l_ref, _ = kvfwd(ref, xt_l, "ref", "l")
        flow_pred_w_pol, _ = kvfwd(policy, xt_w, "pol", "w")
        flow_pred_l_pol, _ = kvfwd(policy, xt_l, "pol", "l")

        target_w = (noise_w - x0_w).float()
        target_l = (noise_l - x0_l).float()
        e_pol_w = (flow_pred_w_pol.float() - target_w).pow(2).mean()
        e_ref_w = (flow_pred_w_ref.float() - target_w).pow(2).mean()
        e_pol_l = (flow_pred_l_pol.float() - target_l).pow(2).mean()
        e_ref_l = (flow_pred_l_ref.float() - target_l).pow(2).mean()

        inside = -0.5 * beta * ((e_pol_w - e_ref_w) - (e_pol_l - e_ref_l))
        loss_b = -F.logsigmoid(inside)

        losses.append(float(loss_b.item()))
        margins.append(float(inside.item()))
        accs.append(float((inside > 0).float().item()))
        c_logp.append(float(-(e_pol_w - e_ref_w).item()))
        r_logp.append(float(-(e_pol_l - e_ref_l).item()))

        # Update KV caches with clean K/V (low timestep) for the next block.
        for c_list in kv.values():
            for layer_cache in c_list:
                layer_cache["global_end_index"] -= num_frame_per_block * _FRAME_SEQ_LEN
                layer_cache["local_end_index"] -= num_frame_per_block * _FRAME_SEQ_LEN
        ctx_t = torch.full((1, num_frame_per_block), context_noise, device=device, dtype=torch.long)
        for (m_key, v_key) in kv.keys():
            model = policy if m_key == "pol" else ref
            x0c = x0_w if v_key == "w" else x0_l
            model(
                noisy_image_or_video=x0c,
                conditional_dict=cond,
                timestep=ctx_t,
                kv_cache=kv[(m_key, v_key)],
                crossattn_cache=cax[m_key],
                current_start=current_start,
            )

    return {
        "dpo_loss": statistics.mean(losses),
        "dpo_margin": statistics.mean(margins),
        "dpo_accuracy": statistics.mean(accs),
        "chosen_logp_diff": statistics.mean(c_logp),
        "rejected_logp_diff": statistics.mean(r_logp),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/eval_prompts.jsonl")
    ap.add_argument("--policy", default="runs/main_v3/dpo_MQ/policy_final.pt")
    ap.add_argument("--out_dir", default="runs/main_v3/eval_dpo")
    ap.add_argument("--seeds", default="1000,1007", help="Two seeds for the chosen/rejected rollouts")
    ap.add_argument("--reward_dim", default="MQ")
    ap.add_argument("--beta", type=float, default=500.0)
    ap.add_argument("--num_t_samples", type=int, default=4, help="random t-grid samples per pair to average")
    ap.add_argument("--latents_dir", default="runs/main_v3/eval_dpo/latents")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    lat_dir = Path(args.latents_dir); lat_dir.mkdir(parents=True, exist_ok=True)
    seed_a, seed_b = [int(x) for x in args.seeds.split(",")]

    eval_prompts = []
    with open(args.prompts) as f:
        for line in f:
            eval_prompts.append(json.loads(line))
    print(f"[init] {len(eval_prompts)} held-out prompts; seeds=({seed_a},{seed_b})")

    device = "cuda"
    dtype = torch.bfloat16

    # ---------- Phase 1: generate base rollouts (latents + MQ scores) ----------
    rollouts_path = out_dir / "rollouts.jsonl"
    pairs_path = out_dir / "pairs.jsonl"

    if not rollouts_path.exists():
        print("[phase 1] generate 2 base rollouts per prompt + score with VideoAlign")
        t0 = time.time()
        pipe = load_pipeline(device=device, dtype=dtype, seed=0)
        rwd = load_reward(device=device, dtype=dtype)
        print(f"[phase 1] pipeline + reward loaded in {time.time()-t0:.1f}s")
        with rollouts_path.open("w") as fout:
            for prow in eval_prompts:
                pid, prompt = prow["id"], prow["prompt"]
                for seed in (seed_a, seed_b):
                    lat_path = lat_dir / f"{pid:03d}_seed{seed}.latent.pt"
                    g0 = time.time()
                    pix, latent = generate_one(pipe, prompt, seed=seed)
                    sc = score_one(rwd, pix, prompt, use_norm=True)
                    torch.save(latent.detach().to("cpu"), lat_path)
                    fout.write(json.dumps({
                        "id": pid, "seed": seed, "scores": sc,
                        "latent": str(lat_path),
                    }) + "\n"); fout.flush()
                    dt = time.time() - g0
                    print(f"  [{pid:3d} s={seed}] MQ={sc['MQ']:+.3f} VQ={sc['VQ']:+.3f} TA={sc['TA']:+.3f} O={sc['Overall']:+.3f}  [{dt:.1f}s]")
        del pipe, rwd
        gc.collect(); torch.cuda.empty_cache()
        print(f"[phase 1] done in {time.time()-t0:.1f}s")
    else:
        print(f"[phase 1] reusing {rollouts_path}")

    # ---------- Phase 2: form pairs by reward_dim ----------
    rows_by_pid = {}
    with open(rollouts_path) as f:
        for line in f:
            r = json.loads(line)
            rows_by_pid.setdefault(r["id"], {})[r["seed"]] = r

    pairs = []
    for pid in sorted(rows_by_pid):
        if seed_a not in rows_by_pid[pid] or seed_b not in rows_by_pid[pid]:
            print(f"[skip-pair {pid}] missing seed")
            continue
        ra = rows_by_pid[pid][seed_a]; rb = rows_by_pid[pid][seed_b]
        sa = ra["scores"][args.reward_dim]; sb = rb["scores"][args.reward_dim]
        if sa == sb:
            print(f"[skip-pair {pid}] tie on {args.reward_dim} = {sa:+.3f}")
            continue
        if sa > sb:
            chosen, rejected = ra, rb
        else:
            chosen, rejected = rb, ra
        # Find the prompt text by id
        prompt = next(p["prompt"] for p in eval_prompts if p["id"] == pid)
        pairs.append({
            "id": pid,
            "prompt": prompt,
            "chosen_seed": chosen["seed"], "rejected_seed": rejected["seed"],
            "chosen_score": chosen["scores"][args.reward_dim],
            "rejected_score": rejected["scores"][args.reward_dim],
            "chosen_latent": chosen["latent"],
            "rejected_latent": rejected["latent"],
        })
    with pairs_path.open("w") as fout:
        for p in pairs:
            fout.write(json.dumps(p) + "\n")
    print(f"[phase 2] {len(pairs)} pairs after tie-drop on dim={args.reward_dim}")

    # ---------- Phase 3: encode prompts, load policy+ref, compute teacher-forced DPO metrics ----------
    print("[phase 3] encode prompts")
    t0 = time.time()
    te, _ = load_text_encoder_and_vae(device=device, dtype=dtype)
    prompt_emb = {}
    with torch.no_grad():
        for p in pairs:
            if p["prompt"] in prompt_emb: continue
            d = te(text_prompts=[p["prompt"]])
            prompt_emb[p["prompt"]] = {k: v.detach().to(device=device, dtype=dtype).cpu() for k, v in d.items()}
    print(f"[phase 3] encoded {len(prompt_emb)} prompts in {time.time()-t0:.1f}s; freeing TE")
    del te
    gc.collect(); torch.cuda.empty_cache()

    print("[phase 3] load policy + ref")
    t0 = time.time()
    policy, ref = load_policy_and_ref(device=device, dtype=dtype)
    state = torch.load(args.policy, map_location="cpu", weights_only=False)
    if "generator" in state: state = state["generator"]
    elif "model" in state: state = state["model"]
    policy.load_state_dict(state, strict=True)
    policy = policy.to(device=device, dtype=dtype)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    print(f"[phase 3] policy+ref ready in {time.time()-t0:.1f}s")
    scheduler = policy.get_scheduler()

    # Fix the per-block t/noise seeds reproducibly across pairs.
    torch.manual_seed(0); random.seed(0)

    per_pair = []
    metrics_path = out_dir / "metrics.jsonl"
    with metrics_path.open("w") as fout:
        for pi, pair in enumerate(pairs):
            cond_cpu = prompt_emb[pair["prompt"]]
            cond = {k: v.to(device=device, dtype=dtype) for k, v in cond_cpu.items()}
            chosen_lat = torch.load(pair["chosen_latent"], map_location=device, weights_only=False).to(dtype)
            rejected_lat = torch.load(pair["rejected_latent"], map_location=device, weights_only=False).to(dtype)
            if chosen_lat.dim() == 4: chosen_lat = chosen_lat.unsqueeze(0)
            if rejected_lat.dim() == 4: rejected_lat = rejected_lat.unsqueeze(0)

            # Average K samples per pair (random t-grid + fresh noise per call).
            samples = []
            for k in range(args.num_t_samples):
                m = chunkwise_dpo_eval(
                    policy, ref, scheduler,
                    chosen_lat=chosen_lat, rejected_lat=rejected_lat,
                    cond=cond, num_frame_per_block=3, beta=args.beta,
                    timesteps_grid=(1000, 750, 500, 250), dtype=dtype,
                )
                samples.append(m)
            agg = {k: statistics.mean([s[k] for s in samples]) for k in samples[0]}
            agg["dpo_margin_std"] = statistics.pstdev([s["dpo_margin"] for s in samples])
            agg["dpo_accuracy_std"] = statistics.pstdev([s["dpo_accuracy"] for s in samples])
            row = {
                "id": pair["id"],
                "chosen_score": pair["chosen_score"], "rejected_score": pair["rejected_score"],
                "gap": pair["chosen_score"] - pair["rejected_score"],
                **agg,
            }
            per_pair.append(row)
            fout.write(json.dumps(row) + "\n"); fout.flush()
            print(f"  [{pair['id']:3d}] gap={row['gap']:+.3f} acc={row['dpo_accuracy']:.2f} margin={row['dpo_margin']:+.3f} loss={row['dpo_loss']:.3f}")

            del chosen_lat, rejected_lat
            gc.collect(); torch.cuda.empty_cache()

    # ---------- Phase 4: aggregate ----------
    accs = [r["dpo_accuracy"] for r in per_pair]
    margins = [r["dpo_margin"] for r in per_pair]
    losses = [r["dpo_loss"] for r in per_pair]
    gaps = [r["gap"] for r in per_pair]
    summary = {
        "n_pairs": len(per_pair),
        "policy": args.policy,
        "reward_dim": args.reward_dim,
        "beta": args.beta,
        "num_t_samples_per_pair": args.num_t_samples,
        "dpo_accuracy": {"mean": statistics.mean(accs), "std": statistics.pstdev(accs)},
        "dpo_margin":   {"mean": statistics.mean(margins), "std": statistics.pstdev(margins)},
        "dpo_loss":     {"mean": statistics.mean(losses), "std": statistics.pstdev(losses)},
        "MQ_gap":       {"mean": statistics.mean(gaps), "std": statistics.pstdev(gaps)},
        "win_rate_acc>0.5": sum(1 for a in accs if a > 0.5) / max(1, len(accs)),
        "win_rate_margin>0": sum(1 for m in margins if m > 0) / max(1, len(margins)),
    }
    with (out_dir / "summary.json").open("w") as fout:
        json.dump(summary, fout, indent=2)
    print("\n[summary]")
    print(json.dumps(summary, indent=2))
    print(f"\n[saved] {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
