"""Chunk-wise teacher-forcing DPO trainer for LongLive (block-causal AR video).

Per the user's design:
  - Teacher forcing: feed clean previous blocks via `clean_x` (no rollout).
  - Per-block gradient: forward and backward block by block (3 latent frames at a time).
  - No gradient flow across chunks: clean prefix is detached.

Loss = -log σ( -β/2 · ( (e_pol^w - e_ref^w) - (e_pol^l - e_ref^l) ) )
where e_*^· = ||noise - flow_pred_*(noisy_block, t, clean_prefix, prompt)||²  (per-element mean).

This is the Wallace et al. (2023) Diffusion-DPO loss, applied chunk-wise to each
3-latent-frame block. The reference model is a frozen copy of the LongLive base.

Logged: dpo_loss, dpo_accuracy (P(margin>0)), implicit_margin (avg of inside),
chosen_logp_diff, rejected_logp_diff, grad_norm.
"""
from __future__ import annotations
import os, sys, time, json, math, random, copy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import wandb

from dpo_longlive.path_setup import (
    add_longlive_to_path,
    ensure_wan_models_symlink,
    LONGLIVE_DIR,
    LONGLIVE_MODELS_DIR,
)


@dataclass
class DPOConfig:
    pairs_index: str = "data/pairs/index.jsonl"
    reward_dim: str = "MQ"  # one of {VQ, MQ, TA, Overall}
    out_dir: str = "runs/dpo_mq"
    lr: float = 1e-6
    beta: float = 5000.0  # DPO temperature; diffusion-DPO uses very large betas
    weight_decay: float = 0.01
    grad_accum: int = 1
    max_steps: int = 400
    log_every: int = 1
    save_every: int = 200
    seed: int = 0
    num_frame_per_block: int = 3
    timesteps: tuple = (1000, 750, 500, 250)  # the 4-step inference grid
    grad_clip: float = 1.0
    wandb_project: Optional[str] = "dpo-longlive"
    wandb_name: Optional[str] = None
    wandb_mode: str = "online"  # or "offline" / "disabled"
    disable_wandb: bool = False
    sample_one_t_per_block: bool = True  # if True, draw a single t per block; else per video


def _build_pipeline(device: str, dtype):
    add_longlive_to_path()
    ensure_wan_models_symlink()
    prev = os.getcwd()
    os.chdir(str(LONGLIVE_DIR))
    try:
        from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper  # type: ignore
        from utils.scheduler import FlowMatchScheduler  # type: ignore
    finally:
        os.chdir(prev)
    return WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper, FlowMatchScheduler


def load_policy_and_ref(device: str = "cuda", dtype=torch.bfloat16):
    """Two copies of LongLive's WanDiffusionWrapper. Policy is trainable, ref is frozen."""
    add_longlive_to_path()
    ensure_wan_models_symlink()
    WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper, _ = _build_pipeline(device, dtype)

    prev = os.getcwd()
    os.chdir(str(LONGLIVE_DIR))
    try:
        policy = WanDiffusionWrapper(
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=5.0, is_causal=True,
            local_attn_size=12, sink_size=3, use_infinite_attention=False,
        )
        ref = WanDiffusionWrapper(
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=5.0, is_causal=True,
            local_attn_size=12, sink_size=3, use_infinite_attention=False,
        )
        ckpt_path = LONGLIVE_MODELS_DIR / "models" / "longlive_base.pt"
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if "generator" in state: state = state["generator"]
        elif "model" in state: state = state["model"]
        policy.load_state_dict(state, strict=True)
        ref.load_state_dict(state, strict=True)
    finally:
        os.chdir(prev)

    policy = policy.to(device=device, dtype=dtype)
    ref = ref.to(device=device, dtype=dtype)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()
    policy.train()
    return policy, ref


def load_text_encoder_and_vae(device: str = "cuda", dtype=torch.bfloat16):
    add_longlive_to_path()
    ensure_wan_models_symlink()
    WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper, _ = _build_pipeline(device, dtype)
    prev = os.getcwd()
    os.chdir(str(LONGLIVE_DIR))
    try:
        te = WanTextEncoder()
        vae = WanVAEWrapper()
    finally:
        os.chdir(prev)
    te = te.to(device=device, dtype=dtype)
    vae = vae.to(device=device, dtype=dtype)
    for m in (te, vae):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    return te, vae


def load_pair_index(path: str, dim: str):
    pairs = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if dim in row.get("pairs", {}):
                pairs.append({
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "chosen": row["pairs"][dim]["chosen_latent"],
                    "rejected": row["pairs"][dim]["rejected_latent"],
                    "chosen_score": row["pairs"][dim]["chosen_score"],
                    "rejected_score": row["pairs"][dim]["rejected_score"],
                })
    return pairs


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def add_noise_to_block(scheduler, x0_block: torch.Tensor, noise: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
    """Add noise to a [B, T, C, H, W] block at scalar timestep t. Returns x_t with same shape."""
    B, T, C, H, W = x0_block.shape
    t_full = t_scalar.view(1).expand(B * T).to(x0_block.device, dtype=torch.long)
    xt = scheduler.add_noise(
        x0_block.flatten(0, 1),
        noise.flatten(0, 1),
        t_full,
    ).unflatten(0, (B, T))
    return xt


_FRAME_SEQ_LEN = 1560
_NUM_TRANSFORMER_BLOCKS = 30


def init_kv_cache(batch_size, kv_size, dtype, device):
    return [
        {
            "k": torch.zeros([batch_size, kv_size, 12, 128], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, kv_size, 12, 128], dtype=dtype, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
        }
        for _ in range(_NUM_TRANSFORMER_BLOCKS)
    ]


def init_crossattn_cache(batch_size, dtype, device):
    return [
        {
            "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
            "is_init": False,
        }
        for _ in range(_NUM_TRANSFORMER_BLOCKS)
    ]


def chunkwise_dpo_step(
    policy, ref, scheduler, *,
    chosen_lat: torch.Tensor, rejected_lat: torch.Tensor,
    cond: dict, num_frame_per_block: int, beta: float,
    timesteps_grid: tuple, dtype, local_attn_size: int = 12,
    context_noise: int = 0, loss_scale: float = 1.0,
    anchor_alpha: float = 0.0,
):
    """
    Block-causal teacher-forcing DPO with PER-BLOCK BACKWARD.

    For each 3-latent-frame block (in order):
      1. Forward through policy and ref on the noisy current block; KV cache holds clean
         encodings of all previous blocks.
      2. Compute the DPO loss for this block.
      3. Backward immediately so the activations for that block can be released.
      4. Update each path's KV cache with the clean current block (no grad).

    The four KV caches store no_grad tensors (the slice-assignment into the zero-init cache
    detaches their grad), so gradient does NOT flow across blocks even though the loss
    accumulates per-block gradients into policy.parameters().

    Args:
        loss_scale: scalar by which to divide each block's loss before backward, to keep
            the effective per-step loss equal to the unscaled mean across blocks.
    Returns:
        scalar (mean loss across blocks, as Python float) and a metrics dict.
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
    # Pre-populate a SHARED cross-attn cache (prompt is constant; populating once with ref
    # under no_grad means subsequent forwards don't write into it, so no grad tensors get
    # stuck across block-by-block backwards).
    shared_cax = init_crossattn_cache(1, dtype, device)
    with torch.no_grad():
        # Run a single tiny forward through ref just to populate the cross-attn cache.
        # We feed a zero block at the lowest timestep — the result is discarded.
        warm_x = torch.zeros((1, num_frame_per_block, 16, chosen_lat.shape[-2], chosen_lat.shape[-1]),
                             dtype=dtype, device=device)
        warm_t = torch.zeros((1, num_frame_per_block), dtype=torch.long, device=device)
        ref(
            noisy_image_or_video=warm_x,
            conditional_dict=cond,
            timestep=warm_t,
            kv_cache=kv[("ref", "w")],  # we'll reset this below
            crossattn_cache=shared_cax,
            current_start=0,
        )
        # Detach all crossattn cache values and reset their write flag so they're treated as fixed.
        for layer in shared_cax:
            layer["k"] = layer["k"].detach()
            layer["v"] = layer["v"].detach()
            layer["is_init"] = True
        # Reset the kv cache we just dirtied.
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

        with torch.no_grad():
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
        dpo_term = -F.logsigmoid(inside)
        # Optional anchor: explicit chosen-NLL term to prevent the chosen video's
        # log-prob from sliding while DPO suppresses the rejected one.
        anchor_term = anchor_alpha * e_pol_w if anchor_alpha > 0 else torch.tensor(0.0, device=e_pol_w.device)
        loss_b = dpo_term + anchor_term

        # Per-block backward — accumulates grads into policy.parameters().
        scaled = loss_b / num_blocks * loss_scale
        scaled.backward(retain_graph=False)

        with torch.no_grad():
            losses.append(loss_b.detach().float().item())
            margins.append(float(inside.detach().item()))
            accs.append(float((inside > 0).float().item()))
            c_logp.append(float(-(e_pol_w - e_ref_w).detach().item()))
            r_logp.append(float(-(e_pol_l - e_ref_l).detach().item()))

        # Drop everything in the autograd graph for this block.
        del flow_pred_w_pol, flow_pred_l_pol, flow_pred_w_ref, flow_pred_l_ref
        del e_pol_w, e_pol_l, e_ref_w, e_ref_l, inside, loss_b
        del xt_w, xt_l, noise_w, noise_l, target_w, target_l, x0_w, x0_l

        # Detach all caches so the freed graph from this block's backward doesn't trip the
        # next block. After this point, cache contents are stale (noisy K/V from this block's
        # forward) — we'll overwrite with clean K/V right below.
        for c_list in kv.values():
            for layer in c_list:
                layer["k"] = layer["k"].detach()
                layer["v"] = layer["v"].detach()
        for layer in shared_cax:
            layer["k"] = layer["k"].detach()
            layer["v"] = layer["v"].detach()

        # Update KV caches with clean K/V (low timestep) to use as context for block bi+1.
        with torch.no_grad():
            for c_list in kv.values():
                for layer_cache in c_list:
                    layer_cache["global_end_index"] -= num_frame_per_block * _FRAME_SEQ_LEN
                    layer_cache["local_end_index"] -= num_frame_per_block * _FRAME_SEQ_LEN
            ctx_t = torch.full((1, num_frame_per_block), context_noise, device=device, dtype=torch.long)
            x0_w_clean = chosen_lat[:, s:e].contiguous()
            x0_l_clean = rejected_lat[:, s:e].contiguous()
            for (m_key, v_key) in kv.keys():
                model = policy if m_key == "pol" else ref
                x0c = x0_w_clean if v_key == "w" else x0_l_clean
                model(
                    noisy_image_or_video=x0c,
                    conditional_dict=cond,
                    timestep=ctx_t,
                    kv_cache=kv[(m_key, v_key)],
                    crossattn_cache=cax[m_key],
                    current_start=current_start,
                )

    del kv, cax
    import statistics
    metrics = {
        "dpo_loss": statistics.mean(losses),
        "dpo_margin": statistics.mean(margins),
        "dpo_accuracy": statistics.mean(accs),
        "chosen_logp_diff": statistics.mean(c_logp),
        "rejected_logp_diff": statistics.mean(r_logp),
    }
    return metrics["dpo_loss"], metrics


def run(cfg: DPOConfig):
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(cfg.seed); random.seed(cfg.seed)

    # WandB
    if not cfg.disable_wandb and cfg.wandb_mode != "disabled":
        wandb.init(
            project=cfg.wandb_project, name=cfg.wandb_name or Path(cfg.out_dir).name,
            mode=cfg.wandb_mode, config=cfg.__dict__,
            dir=str(Path(cfg.out_dir)),
        )

    pairs = load_pair_index(cfg.pairs_index, cfg.reward_dim)
    if not pairs:
        raise RuntimeError(f"no pairs for dim {cfg.reward_dim} in {cfg.pairs_index}")
    print(f"[data] {len(pairs)} pairs for dim={cfg.reward_dim}")
    rng = random.Random(cfg.seed)
    rng.shuffle(pairs)

    # Encode all prompts upfront with the text encoder, then free it (it's 22GB).
    print("[init] loading text encoder for one-time prompt encoding ...")
    t0 = time.time()
    text_encoder, _ = load_text_encoder_and_vae(device=device, dtype=dtype)
    print(f"[init] TE ready in {time.time()-t0:.1f}s")

    print(f"[init] encoding {len(pairs)} prompts ...")
    t0 = time.time()
    prompt_emb = {}
    seen = set()
    with torch.no_grad():
        for p in pairs:
            if p["prompt"] in seen:
                continue
            seen.add(p["prompt"])
            d = text_encoder(text_prompts=[p["prompt"]])
            prompt_emb[p["prompt"]] = {k: v.detach().to(device=device, dtype=dtype).cpu() for k, v in d.items()}
    print(f"[init] encoded {len(prompt_emb)} prompts in {time.time()-t0:.1f}s; freeing TE")
    del text_encoder
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(0)
    print(f"[mem] after freeing TE: {(total-free)/1e9:.1f}/{total/1e9:.1f} GB used")

    print("[init] loading policy + ref ...")
    t0 = time.time()
    policy, ref = load_policy_and_ref(device=device, dtype=dtype)
    # Enable grad checkpointing on policy (memory-saving during DPO)
    if hasattr(policy, "enable_gradient_checkpointing"):
        policy.enable_gradient_checkpointing()
    print(f"[init] policy+ref ready in {time.time()-t0:.1f}s")
    free, total = torch.cuda.mem_get_info(0)
    print(f"[mem] after policy+ref: {(total-free)/1e9:.1f}/{total/1e9:.1f} GB used")

    scheduler = policy.get_scheduler()

    # Optimizer: train ALL of policy.model parameters (the DiT).
    params = [p for p in policy.model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=cfg.lr, betas=(0.9, 0.999), weight_decay=cfg.weight_decay)

    it = cycle(pairs)
    step = 0
    accum = 0
    opt.zero_grad(set_to_none=True)
    while step < cfg.max_steps:
        pair = next(it)
        cond_cpu = prompt_emb[pair["prompt"]]
        cond = {k: v.to(device=device, dtype=dtype) for k, v in cond_cpu.items()}

        chosen_lat = torch.load(pair["chosen"], map_location=device, weights_only=False).to(dtype)
        rejected_lat = torch.load(pair["rejected"], map_location=device, weights_only=False).to(dtype)
        if chosen_lat.dim() == 4: chosen_lat = chosen_lat.unsqueeze(0)
        if rejected_lat.dim() == 4: rejected_lat = rejected_lat.unsqueeze(0)

        # Per-block backward inside chunkwise_dpo_step accumulates grads in policy.parameters().
        loss, metrics = chunkwise_dpo_step(
            policy, ref, scheduler,
            chosen_lat=chosen_lat, rejected_lat=rejected_lat,
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
    ap.add_argument("--out_dir", default="runs/dpo_mq")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=5000.0)
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
    cfg = DPOConfig(**vars(args))
    run(cfg)


if __name__ == "__main__":
    parse_and_run()
