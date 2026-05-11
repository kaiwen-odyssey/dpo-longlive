"""Offline Reward-DMD-style trainer for LongLive (block-causal AR video).

This is an *offline* analogue of Reward-Forcing's Re-DMD loss adapted to share the
exact same pre-generated paired data and chunk-wise teacher-forcing infrastructure as
the DPO trainer. Online Re-DMD (rollout per step + DMD score-difference) requires
keeping a separate `fake_score` critic and a per-step generator rollout, both of which
explode in memory on a single 96 GB GPU when paired with the reward model and VAE.

Per Reward-Forcing's `model/re_dmd.py`, the per-sample loss is:

    L = 0.5 · exp(β · reward) · MSE(generator_output, generator_output − KL_grad)

where the KL gradient (in latent space) is `(pred_fake − pred_real)`. Online, both
predictions are evaluated at the SAME noisy latent at the same timestep, and the
generator's `original_latent` carries gradient. Offline, we have no fresh rollout —
so we substitute `original_latent` with the dataset's clean latent x0 and treat the
reward-weighted denoising loss as the surrogate. Concretely:

    L_block = exp(β · r) · MSE( flow_pred_θ(x_t, t, ctx),  (noise − x0) )

evaluated chunk-wise with a teacher-forcing KV cache populated from the clean prefix.
This is the reward-RWR variant of Re-DMD without sink-EMA. We report results under the
"reward-DMD (offline, no sink-EMA)" name.
"""
from __future__ import annotations
import os, sys, time, json, math, random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
import wandb

from dpo_longlive.dpo_trainer import (
    load_policy_and_ref,
    load_text_encoder_and_vae,
    init_kv_cache,
    init_crossattn_cache,
    add_noise_to_block,
    cycle,
    _FRAME_SEQ_LEN,
    _NUM_TRANSFORMER_BLOCKS,
)


@dataclass
class ReDMDConfig:
    pairs_index: str = "data/pairs/index.jsonl"
    reward_dim: str = "MQ"
    out_dir: str = "runs/redmd_mq"
    lr: float = 1e-6
    beta: float = 2.0  # reward exponent (Reward-Forcing default)
    weight_decay: float = 0.01
    grad_accum: int = 1
    max_steps: int = 400
    log_every: int = 1
    save_every: int = 200
    seed: int = 0
    num_frame_per_block: int = 3
    timesteps: tuple = (1000, 750, 500, 250)
    grad_clip: float = 1.0
    wandb_project: Optional[str] = "dpo-longlive"
    wandb_name: Optional[str] = None
    wandb_mode: str = "online"
    disable_wandb: bool = False


def _load_all_samples(pairs_index_path: str, dim: str):
    """Both chosen and rejected videos become training samples; reward is the dim's score."""
    samples = []
    with open(pairs_index_path) as f:
        for line in f:
            row = json.loads(line)
            for seed, scoredict in row["scores"].items():
                latent_path = Path(row.get("pairs", {}).get(dim, {}).get("chosen_latent", "")).parent / f"seed{seed}.latent.pt"
                # Build path even if not in chosen/rejected
                if not latent_path.exists():
                    pid = row["id"]
                    latent_path = Path(f"data/pairs/{pid:06d}/seed{seed}.latent.pt")
                if not latent_path.exists():
                    continue
                samples.append({
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "latent": str(latent_path),
                    "reward": float(scoredict[dim]),
                })
    return samples


def chunkwise_redmd_step(
    policy, ref, scheduler, *,
    x0_lat: torch.Tensor, reward: float,
    cond: dict, num_frame_per_block: int, beta: float,
    timesteps_grid: tuple, dtype, local_attn_size: int = 12,
    context_noise: int = 0, loss_scale: float = 1.0,
):
    """
    For one rollout (clean latent x0_lat with scalar reward r), per block:
      - Forward policy (grad) and ref (no_grad) on the noisy current block, both
        conditioned on the same clean prefix via KV cache.
      - Denoising loss: ||flow_pol − (noise − x0)||² weighted by exp(β · r).
      - Per-block backward, then detach caches and write clean K/V for next block.
    Backwarding the policy's denoising loss alone (without the ref's score) is the
    reward-weighted-regression flavor of Re-DMD; the ref forward is kept here as an
    implicit anchor (its activations are released immediately).
    """
    device = x0_lat.device
    T_total = x0_lat.shape[1]
    assert T_total % num_frame_per_block == 0
    num_blocks = T_total // num_frame_per_block
    kv_size = local_attn_size * _FRAME_SEQ_LEN
    kv = {
        "pol": init_kv_cache(1, kv_size, dtype, device),
        "ref": init_kv_cache(1, kv_size, dtype, device),
    }
    cax = init_crossattn_cache(1, dtype, device)

    losses_pol, losses_ref, weights = [], [], []
    weight = float(math.exp(beta * reward))

    for bi in range(num_blocks):
        s, e = bi * num_frame_per_block, (bi + 1) * num_frame_per_block
        x0 = x0_lat[:, s:e].contiguous()
        t_idx = torch.randint(0, len(timesteps_grid), (1,)).item()
        t_val = int(timesteps_grid[t_idx])
        t = torch.tensor([t_val], device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        xt = add_noise_to_block(scheduler, x0, noise, t)
        timestep_BT = torch.full((1, num_frame_per_block), t_val, device=device, dtype=torch.long)
        current_start = s * _FRAME_SEQ_LEN

        with torch.no_grad():
            flow_pred_ref, _ = ref(
                noisy_image_or_video=xt, conditional_dict=cond, timestep=timestep_BT,
                kv_cache=kv["ref"], crossattn_cache=cax, current_start=current_start,
            )
        flow_pred_pol, _ = policy(
            noisy_image_or_video=xt, conditional_dict=cond, timestep=timestep_BT,
            kv_cache=kv["pol"], crossattn_cache=cax, current_start=current_start,
        )

        target = (noise - x0).float()
        e_pol = (flow_pred_pol.float() - target).pow(2).mean()
        e_ref = (flow_pred_ref.float() - target).pow(2).mean()

        # Reward-weighted denoising loss (offline Re-DMD surrogate)
        loss_b = weight * 0.5 * e_pol
        scaled = loss_b / num_blocks * loss_scale
        scaled.backward(retain_graph=False)

        with torch.no_grad():
            losses_pol.append(float(e_pol.detach().item()))
            losses_ref.append(float(e_ref.detach().item()))
            weights.append(weight)

        del flow_pred_pol, flow_pred_ref, e_pol, e_ref, loss_b, scaled
        del xt, noise, target, x0

        for c_list in kv.values():
            for layer in c_list:
                layer["k"] = layer["k"].detach()
                layer["v"] = layer["v"].detach()
        for layer in cax:
            layer["k"] = layer["k"].detach()
            layer["v"] = layer["v"].detach()
            layer["is_init"] = True

        with torch.no_grad():
            for c_list in kv.values():
                for layer in c_list:
                    layer["global_end_index"] -= num_frame_per_block * _FRAME_SEQ_LEN
                    layer["local_end_index"] -= num_frame_per_block * _FRAME_SEQ_LEN
            ctx_t = torch.full((1, num_frame_per_block), context_noise, device=device, dtype=torch.long)
            x0_clean = x0_lat[:, s:e].contiguous()
            for m_key, c_list in kv.items():
                model = policy if m_key == "pol" else ref
                model(
                    noisy_image_or_video=x0_clean, conditional_dict=cond, timestep=ctx_t,
                    kv_cache=c_list, crossattn_cache=cax, current_start=current_start,
                )

    del kv, cax
    import statistics
    return statistics.mean(losses_pol), {
        "redmd_loss_pol": statistics.mean(losses_pol),
        "redmd_loss_ref": statistics.mean(losses_ref),
        "reward": reward,
        "exp_beta_reward": float(weight),
        "denoising_gap": statistics.mean(losses_pol) - statistics.mean(losses_ref),
    }


def run(cfg: ReDMDConfig):
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    device = "cuda"; dtype = torch.bfloat16
    torch.manual_seed(cfg.seed); random.seed(cfg.seed)

    if not cfg.disable_wandb and cfg.wandb_mode != "disabled":
        wandb.init(
            project=cfg.wandb_project, name=cfg.wandb_name or Path(cfg.out_dir).name,
            mode=cfg.wandb_mode, config=cfg.__dict__, dir=str(Path(cfg.out_dir)),
        )

    samples = _load_all_samples(cfg.pairs_index, cfg.reward_dim)
    if not samples:
        raise RuntimeError(f"no samples for dim {cfg.reward_dim}")
    print(f"[data] {len(samples)} samples for dim={cfg.reward_dim}")
    rng = random.Random(cfg.seed); rng.shuffle(samples)

    print("[init] loading text encoder ...")
    t0 = time.time()
    te, _ = load_text_encoder_and_vae(device=device, dtype=dtype)
    prompt_emb = {}
    seen = set()
    with torch.no_grad():
        for s in samples:
            if s["prompt"] in seen: continue
            seen.add(s["prompt"])
            d = te(text_prompts=[s["prompt"]])
            prompt_emb[s["prompt"]] = {k: v.detach().cpu() for k, v in d.items()}
    del te
    import gc; gc.collect(); torch.cuda.empty_cache()
    print(f"[init] encoded {len(prompt_emb)} prompts in {time.time()-t0:.1f}s")

    print("[init] loading policy + ref ...")
    t0 = time.time()
    policy, ref = load_policy_and_ref(device=device, dtype=dtype)
    print(f"[init] policy+ref ready in {time.time()-t0:.1f}s")

    scheduler = policy.get_scheduler()
    params = [p for p in policy.model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=cfg.lr, betas=(0.9, 0.999), weight_decay=cfg.weight_decay)

    it = cycle(samples)
    step = 0; accum = 0
    opt.zero_grad(set_to_none=True)
    while step < cfg.max_steps:
        s = next(it)
        cond_cpu = prompt_emb[s["prompt"]]
        cond = {k: v.to(device=device, dtype=dtype) for k, v in cond_cpu.items()}
        x0_lat = torch.load(s["latent"], map_location=device, weights_only=False).to(dtype)
        if x0_lat.dim() == 4: x0_lat = x0_lat.unsqueeze(0)

        loss, metrics = chunkwise_redmd_step(
            policy, ref, scheduler, x0_lat=x0_lat, reward=s["reward"],
            cond=cond, num_frame_per_block=cfg.num_frame_per_block,
            beta=cfg.beta, timesteps_grid=cfg.timesteps,
            dtype=dtype, loss_scale=1.0 / cfg.grad_accum,
        )
        accum += 1
        if accum >= cfg.grad_accum:
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            accum = 0
            step += 1
            metrics["grad_norm"] = float(grad_norm.detach().item())
            metrics["step"] = step
            metrics["lr"] = opt.param_groups[0]["lr"]
            free, total = torch.cuda.mem_get_info(0)
            metrics["mem_gb"] = (total - free) / 1e9
            if step % cfg.log_every == 0:
                print(f"[step {step:4d}/{cfg.max_steps}] {metrics}")
                if not cfg.disable_wandb and cfg.wandb_mode != "disabled":
                    wandb.log(metrics, step=step)
            if step % cfg.save_every == 0 or step == cfg.max_steps:
                ck = Path(cfg.out_dir) / f"policy_step{step:06d}.pt"
                torch.save(policy.state_dict(), ck)
                print(f"[save] {ck}")

    if not cfg.disable_wandb and cfg.wandb_mode != "disabled":
        wandb.finish()


def parse_and_run():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_index", default="data/pairs/index.jsonl")
    ap.add_argument("--reward_dim", default="MQ")
    ap.add_argument("--out_dir", default="runs/redmd_mq")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb_project", default="dpo-longlive")
    ap.add_argument("--wandb_name", default=None)
    ap.add_argument("--wandb_mode", default="online")
    ap.add_argument("--disable_wandb", action="store_true")
    args = ap.parse_args()
    cfg = ReDMDConfig(**vars(args))
    run(cfg)


if __name__ == "__main__":
    parse_and_run()
