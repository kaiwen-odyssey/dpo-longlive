"""Thin wrapper around LongLive's CausalInferencePipeline that we can call from our scripts.

Loads the Wan2.1-T2V-1.3B base + LongLive base ckpt + VAE + text encoder once.
Designed for single-rollout, no-prompt-transition inference (5s, 81 frames, 832x480).
"""
from __future__ import annotations
import os
import torch
from pathlib import Path
from omegaconf import OmegaConf

from dpo_longlive.path_setup import (
    add_longlive_to_path,
    ensure_wan_models_symlink,
    LONGLIVE_DIR,
    LONGLIVE_MODELS_DIR,
)


def build_inference_config(seed: int = 0) -> "OmegaConf":
    cfg = OmegaConf.create({
        "denoising_step_list": [1000, 750, 500, 250],
        "warp_denoising_step": True,
        "num_frame_per_block": 3,
        "model_name": "Wan2.1-T2V-1.3B",
        "model_kwargs": {
            "local_attn_size": 12,
            "timestep_shift": 5.0,
            "sink_size": 3,
        },
        # inference
        "data_path": str(LONGLIVE_MODELS_DIR / "prompts" / "vidprom_filtered_extended.txt"),
        "output_folder": "videos/short",
        "use_ema": False,
        "seed": int(seed),
        "num_samples": 1,
        "save_with_index": True,
        "global_sink": True,
        "context_noise": 0,
        "generator_ckpt": str(LONGLIVE_MODELS_DIR / "models" / "longlive_base.pt"),
    })
    return cfg


def load_pipeline(device: str | torch.device = "cuda", dtype=torch.bfloat16, seed: int = 0):
    """Load LongLive CausalInferencePipeline + load the longlive_base.pt weights into the generator."""
    add_longlive_to_path()
    ensure_wan_models_symlink()

    prev = os.getcwd()
    os.chdir(str(LONGLIVE_DIR))
    try:
        from pipeline.causal_inference import CausalInferencePipeline  # type: ignore
        cfg = build_inference_config(seed=seed)
        device = torch.device(device)
        pipe = CausalInferencePipeline(cfg, device=device)
        state = torch.load(cfg.generator_ckpt, map_location="cpu", weights_only=False)
        if "generator" in state:
            state = state["generator"]
        elif "model" in state:
            state = state["model"]
        pipe.generator.load_state_dict(state, strict=True)
        pipe = pipe.to(dtype=dtype)
        pipe.text_encoder.to(device)
        pipe.generator.to(device)
        pipe.vae.to(device)
    finally:
        os.chdir(prev)
    pipe.eval()
    for p in pipe.parameters():
        p.requires_grad_(False)
    pipe._cfg = cfg
    return pipe


@torch.no_grad()
def generate_one(pipe, prompt: str, *, seed: int, num_latent_frames: int = 21,
                 height: int = 60, width: int = 104, channels: int = 16, dtype=torch.bfloat16):
    """Generate a single 21-latent-frame (= 81 pixel-frame) video. Returns (pixels[T,C,H,W] in [0,1], latent[B,T,C,H,W])."""
    g = torch.Generator(device=pipe.generator.model.parameters().__next__().device)
    g.manual_seed(int(seed))
    sampled_noise = torch.randn(
        [1, num_latent_frames, channels, height, width],
        device=g.device,
        dtype=dtype,
        generator=g,
    )
    video, latents = pipe.inference(
        noise=sampled_noise,
        text_prompts=[prompt],
        return_latents=True,
        low_memory=False,
    )
    # video: [1, T, C, H, W] in [0, 1]
    pixels = video[0].clamp(0, 1).contiguous()
    return pixels, latents
