"""VideoAlign / VideoReward inference wrapper for in-memory (T,C,H,W) tensors."""
from __future__ import annotations
import torch
from pathlib import Path
from dpo_longlive.path_setup import (
    add_reward_forcing_to_path,
    VIDEOREWARD_DIR,
)


def load_reward(device: str | torch.device = "cuda", dtype=torch.bfloat16):
    add_reward_forcing_to_path()
    import os, json
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Monkey-patch the broken state-dict remapping in videoalign.utils.load_model_from_checkpoint
    # so it loads the checkpoint keys verbatim (the remap was written for a different
    # transformers Qwen2-VL layout than 4.49.0 ships).
    from videoalign import utils as va_utils  # type: ignore
    import torch as _torch
    import os.path as op

    def _load_no_remap(model, checkpoint_dir, checkpoint_step=-1):
        # find latest checkpoint
        import glob
        ckpt_dirs = sorted(glob.glob(op.join(checkpoint_dir, "checkpoint-*")))
        ckpt_path = ckpt_dirs[-1]
        full = op.join(ckpt_path, "model.pth")
        sd = _torch.load(full, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected:
            print(f"[reward_lib] {len(unexpected)} unexpected keys (first 3): {list(unexpected)[:3]}")
        if missing:
            # only warn for non-rm_head missing keys
            real_missing = [k for k in missing if "rm_head" not in k]
            if real_missing:
                print(f"[reward_lib] {len(real_missing)} missing keys (first 3): {real_missing[:3]}")
        return model, ckpt_path.split("checkpoint-")[-1]

    va_utils.load_model_from_checkpoint = _load_no_remap

    from videoalign.inference import VideoVLMRewardInference  # type: ignore

    inferencer = VideoVLMRewardInference(
        str(VIDEOREWARD_DIR),
        device=str(device),
        dtype=dtype,
    )
    inferencer.model.eval()
    for p in inferencer.model.parameters():
        p.requires_grad_(False)
    return inferencer


@torch.no_grad()
def score_one(inferencer, pixels_TCHW: torch.Tensor, prompt: str, use_norm: bool = True,
              tmp_path: str = "/tmp/_dpo_score.mp4", fps: int = 16) -> dict:
    """pixels_TCHW: [T,C,H,W] in [0,1]. Writes to a tmp mp4 and uses .reward(file_paths) which
    handles all the Qwen2-VL preprocessing reliably. Returns VQ/MQ/TA/Overall floats."""
    if pixels_TCHW.dim() != 4:
        raise ValueError(f"expected [T,C,H,W], got {tuple(pixels_TCHW.shape)}")
    import imageio
    vid = (pixels_TCHW.clamp(0, 1).permute(0, 2, 3, 1) * 255).round().to(torch.uint8).cpu().numpy()
    imageio.mimwrite(tmp_path, vid, fps=fps, quality=6, macro_block_size=1)
    rewards = inferencer.reward([tmp_path], [prompt], use_norm=use_norm)
    return rewards[0]
