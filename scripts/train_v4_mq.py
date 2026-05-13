#!/usr/bin/env python
"""v4 dpo_MQ training: 1000 steps with periodic eval + checkpointing.

Recipe (from 50-step β×lr ablation winner + design 2026-05-11):
  - β=500, lr=5e-6, anchor α=0.5, batch 1, no min_gap filter
  - MQ reward dim, training pool = full ~1000 MQ pairs in data/pairs/
  - 1000 optimizer steps

Every 100 steps:
  - Evaluate on 50 held-out prompts: mean MQ + win-rate vs cached base@seed1000
  - Evaluate on 100 most-recent train prompts: mean MQ + win-rate vs cached base@seed1000
  - Save best-by-eval50-mean-MQ policy (overwrite); save metrics history JSON

Implementation:
  - One CausalInferencePipeline shared between training (training-mode policy =
    pipe.generator) and eval (pipe.inference at seed=1000); ref = deepcopy of
    pipe.generator with frozen weights.
  - text_encoder pre-encodes all prompts upfront; then patched to a tiny lookup
    class so pipe.inference uses cached embeddings — keeps the 22 GB of TE off
    GPU throughout, freeing memory for training activations.
"""
from __future__ import annotations
import os, sys, json, time, gc, random, copy, statistics, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F


# ============================================================================
# Setup helpers
# ============================================================================

class CachedTextEncoder(torch.nn.Module):
    """Tiny lookup-table stand-in for pipe.text_encoder.

    After pre-encoding all prompts via the real TE, we replace pipe.text_encoder
    with this so subsequent pipe.inference() calls don't need 22 GB TE on GPU.
    Returns the cached conditional_dict moved to GPU at the right dtype.

    Subclasses nn.Module because pipe is itself an nn.Module and Python's
    __setattr__ check requires child attrs to be Modules.
    """
    def __init__(self, prompt_emb_cpu: dict, device: str = "cuda", dtype=torch.bfloat16):
        super().__init__()
        self.cache = prompt_emb_cpu  # {prompt_str: {k: cpu_tensor}} — kept off the parameter list
        self._device = device
        self._dtype = dtype

    def forward(self, text_prompts):
        prompt = text_prompts[0]
        d = self.cache[prompt]
        return {k: v.to(device=self._device, dtype=self._dtype) for k, v in d.items()}


def sample_extra_eval_prompts(n_extra: int, train_prompts: list, current_eval: list,
                              vidprom_path: Path, seed: int = 7) -> list:
    """Pick n_extra prompts from vidprom_filtered_extended.txt not in train/eval."""
    used = set(p["prompt"] for p in train_prompts) | set(p["prompt"] for p in current_eval)
    rng = random.Random(seed)
    all_lines = vidprom_path.read_text().splitlines()
    candidates = [ln.strip() for ln in all_lines if ln.strip() and ln.strip() not in used]
    rng.shuffle(candidates)
    return candidates[:n_extra]


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_index", default="data/pairs/index.jsonl")
    ap.add_argument("--train_prompts", default="data/train_prompts.jsonl")
    ap.add_argument("--eval_prompts", default="data/eval_prompts.jsonl")
    ap.add_argument("--vidprom_path", default="assets/longlive/prompts/vidprom_filtered_extended.txt")
    ap.add_argument("--reward_dim", default="MQ")
    ap.add_argument("--out_dir", default="runs/v4_mq")
    ap.add_argument("--beta", type=float, default=500.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--anchor_alpha", type=float, default=0.5)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--n_eval_prompts", type=int, default=50)
    ap.add_argument("--aggregate_chunks", action="store_true",
                    help="Aggregate per-block margins before sigmoid (single rollout-level DPO loss)")
    ap.add_argument("--no_save_policy", action="store_true",
                    help="Skip policy_best.pt and policy_final.pt writes (saves ~5 GB for grid runs)")
    ap.add_argument("--train_seed", type=int, default=0)
    ap.add_argument("--vidprom_seed", type=int, default=7)
    # Per-prompt seed = 1000 + 31*pid (matches data/pairs/ convention from gen_pairs.py).
    # Each prompt has its own deterministic noise that is identical across checkpoints
    # and identical to its cached base rollout in data/pairs/<id>/seed{1000+31*pid}.latent.pt.
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"

    device = "cuda"; dtype = torch.bfloat16
    torch.manual_seed(args.train_seed); random.seed(args.train_seed)

    from dpo_longlive.path_setup import add_longlive_to_path, ensure_wan_models_symlink
    add_longlive_to_path(); ensure_wan_models_symlink()

    from dpo_longlive.dpo_trainer import chunkwise_dpo_step, chunkwise_dpo_step_aggregated
    from dpo_longlive.longlive_pipeline import load_pipeline, generate_one
    from dpo_longlive.reward_lib import load_reward, score_one

    train_step_fn = chunkwise_dpo_step_aggregated if args.aggregate_chunks else chunkwise_dpo_step
    print(f"[init] DPO loss mode: {'aggregated (single rollout-level sigmoid)' if args.aggregate_chunks else 'per-chunk (one sigmoid per block)'}")

    # ----- Phase 0: load prompts -----
    train_prompts = [json.loads(l) for l in open(args.train_prompts)]
    eval_prompts  = [json.loads(l) for l in open(args.eval_prompts)]
    pairs_rows    = [json.loads(l) for l in open(args.pairs_index)]
    print(f"[init] {len(train_prompts)} train prompts, {len(eval_prompts)} held-out eval prompts, {len(pairs_rows)} pair rows")

    # Sample 30 extra eval prompts if eval set is short
    n_extra = args.n_eval_prompts - len(eval_prompts)
    eval_extra_path = out_dir / "eval_prompts_extra.jsonl"
    if n_extra > 0:
        if eval_extra_path.exists():
            extras = [json.loads(l) for l in open(eval_extra_path)]
            print(f"[init] reusing {len(extras)} extra eval prompts from {eval_extra_path}")
        else:
            extras_text = sample_extra_eval_prompts(
                n_extra, train_prompts, eval_prompts,
                ROOT / args.vidprom_path, seed=args.vidprom_seed,
            )
            extras = [{"id": 1000 + i, "prompt": t} for i, t in enumerate(extras_text)]
            with eval_extra_path.open("w") as f:
                for ep in extras: f.write(json.dumps(ep) + "\n")
            print(f"[init] sampled {len(extras)} extra eval prompts → {eval_extra_path}")
        eval_prompts_full = eval_prompts + extras
    else:
        eval_prompts_full = eval_prompts[: args.n_eval_prompts]
    assert len(eval_prompts_full) == args.n_eval_prompts, len(eval_prompts_full)

    # Build the MQ pair list from pairs_index.
    # seed_a(pid) = 1000 + 31*pid is the first of the two rollouts in data prep.
    mq_pairs = []
    for row in pairs_rows:
        if args.reward_dim in row.get("pairs", {}):
            pid = row["id"]
            base_seed = 1000 + 31 * pid
            p = row["pairs"][args.reward_dim]
            mq_pairs.append({
                "id": pid,
                "prompt": row["prompt"],
                "chosen": p["chosen_latent"],
                "rejected": p["rejected_latent"],
                "base_seed": base_seed,
                "base_MQ": row["scores"][str(base_seed)][args.reward_dim],
            })
    print(f"[init] {len(mq_pairs)} {args.reward_dim} pairs available for training")

    # Pre-build the training shuffle so we know which prompts each 100-step
    # window will touch (needed for the train-window-100 eval phase).
    rng = random.Random(args.train_seed)
    pair_order = mq_pairs.copy(); rng.shuffle(pair_order)
    train_schedule = [pair_order[i % len(pair_order)] for i in range(args.max_steps)]

    # ----- Phase 1: load pipeline (its TE is on GPU) -----
    print("[init] loading inference pipeline (incl. TE on GPU for one-time encode) ...")
    t0 = time.time()
    pipe = load_pipeline(device=device, dtype=dtype, seed=0)
    print(f"[init] pipeline ready in {time.time()-t0:.1f}s")

    # ----- Phase 2: encode all prompts via pipe.text_encoder, then free TE -----
    all_prompts = set()
    for r in train_prompts: all_prompts.add(r["prompt"])
    for r in eval_prompts_full: all_prompts.add(r["prompt"])
    print(f"[init] encoding {len(all_prompts)} unique prompts via pipe.text_encoder ...")
    t0 = time.time()
    prompt_emb = {}
    with torch.no_grad():
        for pr in sorted(all_prompts):
            d = pipe.text_encoder(text_prompts=[pr])
            prompt_emb[pr] = {k: v.detach().to("cpu") for k, v in d.items()}
    print(f"[init] encoded {len(prompt_emb)} prompts in {time.time()-t0:.1f}s; replacing TE with cache")
    te_original = pipe.text_encoder
    pipe.text_encoder = CachedTextEncoder(prompt_emb, device=device, dtype=dtype)
    del te_original; gc.collect(); torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(0)
    print(f"[mem] after freeing TE: {(total-free)/1e9:.1f}/{total/1e9:.1f} GB used")

    policy = pipe.generator
    for p in policy.parameters(): p.requires_grad_(True)
    if hasattr(policy, "enable_gradient_checkpointing"):
        policy.enable_gradient_checkpointing()
    policy.train()

    print("[init] deepcopy ref model ...")
    t0 = time.time()
    ref = copy.deepcopy(policy)
    for p in ref.parameters(): p.requires_grad_(False)
    ref.eval()
    print(f"[init] ref ready in {time.time()-t0:.1f}s")

    print("[init] loading reward model ...")
    t0 = time.time()
    rwd = load_reward(device=device, dtype=dtype)
    print(f"[init] reward ready in {time.time()-t0:.1f}s")

    scheduler = policy.get_scheduler()
    free, total = torch.cuda.mem_get_info(0)
    print(f"[mem] after init: {(total-free)/1e9:.1f}/{total/1e9:.1f} GB used")

    # ----- Phase 3: cache baseline MQ on eval-50 (per-prompt seed = 1000 + 31*pid) -----
    # Each eval prompt gets its own deterministic noise via per-prompt seed.
    # Same seed will be used at every checkpoint, so noise is fixed across the run.
    for ep in eval_prompts_full:
        ep["eval_seed"] = 1000 + 31 * ep["id"]
    base_eval_path = out_dir / "base_eval.jsonl"
    if base_eval_path.exists():
        base_eval = [json.loads(l) for l in open(base_eval_path)]
        print(f"[base] reusing {len(base_eval)} cached base eval scores")
    else:
        print(f"[base] generating {len(eval_prompts_full)} base rollouts (per-prompt seeds) ...")
        policy.eval()
        # NOTE: pipe.generator's weights ARE the LongLive base right now (we haven't trained yet)
        base_eval = []
        with base_eval_path.open("w") as fout:
            for ep in eval_prompts_full:
                t_gen = time.time()
                pix, _ = generate_one(pipe, ep["prompt"], seed=ep["eval_seed"])
                sc = score_one(rwd, pix, ep["prompt"], use_norm=True)
                row = {"id": ep["id"], "prompt": ep["prompt"], "seed": ep["eval_seed"], "scores": sc}
                base_eval.append(row); fout.write(json.dumps(row) + "\n"); fout.flush()
                print(f"  [base id={ep['id']:5d} seed={ep['eval_seed']}] MQ={sc['MQ']:+.3f}  ({time.time()-t_gen:.1f}s)")
                del pix; torch.cuda.empty_cache()
        policy.train()
    base_eval_mean_MQ = statistics.mean(r["scores"]["MQ"] for r in base_eval)
    print(f"[base] eval-50 mean MQ = {base_eval_mean_MQ:+.4f}")

    # ----- Phase 4: training loop with periodic eval -----
    params = [p for p in policy.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler_train = policy.get_scheduler()

    metrics_history = {
        "config": {"beta": args.beta, "lr": args.lr, "anchor_alpha": args.anchor_alpha,
                   "max_steps": args.max_steps, "reward_dim": args.reward_dim,
                   "n_eval_prompts": args.n_eval_prompts,
                   "rollout_seed_formula": "1000 + 31 * prompt_id",
                   "train_pool": len(mq_pairs),
                   "aggregate_chunks": args.aggregate_chunks},
        "baseline_eval_mean_MQ": base_eval_mean_MQ,
        "train_steps": [],
        "checkpoints": [],
    }
    best_eval_mean_MQ = float("-inf")

    print(f"\n========== training {args.max_steps} steps ==========")

    def do_eval(step_id):
        """Run the periodic eval block. Returns dict of metrics."""
        nonlocal best_eval_mean_MQ
        policy.eval()
        t_eval_start = time.time()

        # ----- eval-50 (per-prompt seed = 1000 + 31*pid) -----
        eval_rows = []
        for ep in eval_prompts_full:
            pix, _ = generate_one(pipe, ep["prompt"], seed=ep["eval_seed"])
            sc = score_one(rwd, pix, ep["prompt"], use_norm=True)
            eval_rows.append({"id": ep["id"], "MQ": sc["MQ"]})
            del pix; torch.cuda.empty_cache()
        eval_MQs = [r["MQ"] for r in eval_rows]
        base_lookup = {r["id"]: r["scores"]["MQ"] for r in base_eval}
        eval_wins = [1.0 if r["MQ"] > base_lookup[r["id"]] else 0.0 for r in eval_rows]
        eval_mean = statistics.mean(eval_MQs)
        eval_win  = statistics.mean(eval_wins)

        # ----- train-window-100 (per-prompt seed matching cached base) -----
        window_pairs = train_schedule[max(0, step_id - args.eval_every): step_id]
        train_rows = []
        for tp in window_pairs:
            pix, _ = generate_one(pipe, tp["prompt"], seed=tp["base_seed"])
            sc = score_one(rwd, pix, tp["prompt"], use_norm=True)
            train_rows.append({"id": tp["id"], "MQ": sc["MQ"], "base_MQ": tp["base_MQ"]})
            del pix; torch.cuda.empty_cache()
        train_MQs = [r["MQ"] for r in train_rows]
        train_wins = [1.0 if r["MQ"] > r["base_MQ"] else 0.0 for r in train_rows]
        train_mean = statistics.mean(train_MQs)
        train_win  = statistics.mean(train_wins)

        eval_time = time.time() - t_eval_start
        print(f"\n  [eval step={step_id}]")
        print(f"    eval-50:    mean MQ {eval_mean:+.4f} (base {base_eval_mean_MQ:+.4f}, Δ {eval_mean-base_eval_mean_MQ:+.4f})  win {eval_win:.2f}")
        print(f"    train-100:  mean MQ {train_mean:+.4f}  win {train_win:.2f}  ({len(window_pairs)} prompts)")
        print(f"    eval took {eval_time/60:.1f} min")

        # Save best-by-eval-MQ checkpoint (unless --no_save_policy)
        if eval_mean > best_eval_mean_MQ:
            best_eval_mean_MQ = eval_mean
            if not args.no_save_policy:
                best_path = out_dir / "policy_best.pt"
                torch.save(policy.state_dict(), best_path)
                print(f"    [best] saved {best_path}  (eval mean MQ {eval_mean:+.4f})")
            else:
                print(f"    [best] step {step_id} (eval mean MQ {eval_mean:+.4f}) — policy save skipped")

        policy.train()
        return {
            "step": step_id,
            "eval50_mean_MQ": eval_mean, "eval50_win_rate": eval_win,
            "train_window_mean_MQ": train_mean, "train_window_win_rate": train_win,
            "eval_time_sec": eval_time,
            "eval_per_prompt": eval_rows, "train_per_prompt": train_rows,
        }

    # Training
    opt.zero_grad(set_to_none=True)
    t_global = time.time()
    for step in range(1, args.max_steps + 1):
        pair = train_schedule[step - 1]
        cond = {k: v.to(device=device, dtype=dtype) for k, v in prompt_emb[pair["prompt"]].items()}
        cw = torch.load(pair["chosen"], map_location=device, weights_only=False).to(dtype)
        rj = torch.load(pair["rejected"], map_location=device, weights_only=False).to(dtype)
        if cw.dim() == 4: cw = cw.unsqueeze(0)
        if rj.dim() == 4: rj = rj.unsqueeze(0)

        loss, m = train_step_fn(
            policy, ref, scheduler_train,
            chosen_lat=cw, rejected_lat=rj, cond=cond,
            num_frame_per_block=3, beta=args.beta,
            timesteps_grid=(1000, 750, 500, 250), dtype=dtype,
            loss_scale=1.0, anchor_alpha=args.anchor_alpha,
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        m["grad_norm"] = float(grad_norm.detach().item())
        m["step"] = step
        m["pair_id"] = pair["id"]
        metrics_history["train_steps"].append(m)

        if step % 5 == 0 or step <= 3:
            elapsed_min = (time.time() - t_global) / 60
            eta_min = elapsed_min * (args.max_steps - step) / step
            print(f"  [step {step:4d}/{args.max_steps}] loss={m['dpo_loss']:.4f} margin={m['dpo_margin']:+.3f} acc={m['dpo_accuracy']:.2f} grad={m['grad_norm']:.1f}  elapsed={elapsed_min:.0f}min eta={eta_min:.0f}min")

        del cw, rj
        torch.cuda.empty_cache()

        if step % args.eval_every == 0:
            ckpt_metrics = do_eval(step)
            metrics_history["checkpoints"].append(ckpt_metrics)
            with metrics_path.open("w") as fout:
                json.dump(metrics_history, fout, indent=2)

    # Save final policy (unless --no_save_policy)
    if not args.no_save_policy:
        final_path = out_dir / "policy_final.pt"
        torch.save(policy.state_dict(), final_path)
        print(f"\n[final] saved {final_path}")
    else:
        print(f"\n[final] policy save skipped (--no_save_policy)")
    print(f"[final] best eval mean MQ = {best_eval_mean_MQ:+.4f} (baseline {base_eval_mean_MQ:+.4f})")
    print(f"[final] total wall time: {(time.time() - t_global)/3600:.2f} h")

    with metrics_path.open("w") as fout:
        json.dump(metrics_history, fout, indent=2)


if __name__ == "__main__":
    main()
