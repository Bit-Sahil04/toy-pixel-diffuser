"""Training loop: standard DDPM forward-noising + denoising-loss training.

Structure follows DeepFindr's "Diffusion models from scratch in PyTorch"
tutorial, extended with: mixed precision (default ON), gradient accumulation,
EMA weights, CSV logging, resumable checkpointing, and fixed-seed qualitative
sample grids.

Usage:
    python train.py                                    # default (unconditional)
    python train.py --batch_size 16 --grad_accum_steps 2
    python train.py --resume_from checkpoints/latest.pt
    python train.py --limit 256                        # smoke test on a subset
"""

import argparse
import csv
import dataclasses
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

from config import Config
from data import PixelArtDataset, ensure_data
from diffusion import Diffusion
from model import build_model


# ------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def estimate_vram_gb(cfg: Config) -> float:
    """Crude but explicit VRAM heuristic for the --max_vram_gb soft guard.

    Components (fp16/fp32 mixed):
      * weights + grads + Adam moments: ~ (2 + 4 + 8) bytes/param = 14
      * fp32 master copy of weights:    + 4 bytes/param
      * activations: batch * img^2 * (base_ch * sum(mults) * num_res_blocks)
        feature maps per level, duplicated across encoder+decoder (~x2) and
        saved-for-backward (~x4 with autocast): factor 8 * 2 bytes
    Deliberately rough — it exists to catch "batch_size 128" mistakes, not to
    be a precise predictor.
    """
    n_params_approx = 0
    chs = [cfg.base_channels * m for m in cfg.channel_mults]
    prev = cfg.base_channels
    for c in chs:
        for _ in range(cfg.num_res_blocks):
            n_params_approx += prev * c * 9 + c * c * 9
            prev = c
    param_gb = n_params_approx * 18 / 1024**3

    feat_ch = cfg.base_channels * sum(cfg.channel_mults) * cfg.num_res_blocks
    act_gb = (cfg.batch_size * cfg.image_size**2 * feat_ch * 8 * 2) / 1024**3
    return param_gb + act_gb


def vram_guard(cfg: Config, total_vram_gb: float) -> None:
    est = estimate_vram_gb(cfg)
    limit = total_vram_gb * cfg.vram_safety_fraction
    print(f"[vram-guard] heuristic estimate: {est:.2f} GB "
          f"(batch={cfg.batch_size}, size={cfg.image_size}, base_ch={cfg.base_channels}) "
          f"vs {total_vram_gb:.2f} GB total VRAM")
    if est > limit:
        print(f"[vram-guard] WARNING: estimate exceeds safe fraction "
              f"({cfg.vram_safety_fraction:.0%}) of {total_vram_gb:.2f}GB VRAM. "
              f"Lower batch_size (try 4-8), reduce image_size, or lower base_channels.")
        print("[vram-guard] Continuing anyway — this is a soft guard. Expect OOM risk.")


def format_eta(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


class EMA:
    """Exponential moving average of model weights.

    EMA weights are saved separately in every checkpoint and are what
    sampling/eval uses by default — EMA samples are typically much cleaner
    than raw weights mid-training (same trick as the reference tutorial).

    Includes standard bias warmup: the effective decay ramps as
    min(decay, (1+n)/(10+n)) over updates n, so early EMA isn't dominated by
    the random initial weights.
    """

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.num_updates = 0
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)
            else:
                self.shadow[k] = v.detach().clone().float()

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict({k: v.to(dtype=next(model.parameters()).dtype)
                               for k, v in self.shadow.items()})


def save_checkpoint(cfg: Config, checkpoint_dir: Path, model, optimizer, scaler, ema, step: int) -> None:
    """Save ckpt_XXXXXXX.pt + latest.pt with ONE serialization.

    torch.save streams bytes through whatever filesystem `path` sits on. On
    Colab that is the Drive FUSE mount, where small writes are extremely slow
    and the previous code paid that cost twice per checkpoint (once for
    ckpt_XXXX.pt, once for latest.pt). Instead: serialize once to a LOCAL
    temp file (fast disk), then plain-copy the finished file to the target
    directory — a single large sequential write is much cheaper than a
    streamed serialization.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),     # Adam moments — required for clean resume
        "scaler": scaler.state_dict(),           # AMP scaler state
        "ema": {"shadow": ema.shadow, "num_updates": ema.num_updates},
        "config": dataclasses.asdict(cfg),
    }
    fd, tmp = tempfile.mkstemp(suffix=".pt")     # always local disk
    os.close(fd)
    try:
        torch.save(payload, tmp)
        shutil.copyfile(tmp, checkpoint_dir / f"ckpt_{step:07d}.pt")
        shutil.copyfile(tmp, checkpoint_dir / "latest.pt")
    finally:
        os.unlink(tmp)


def _same(a, b) -> bool:
    """Config-value comparison tolerating tuple-vs-list (JSON round-trip)."""
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return list(a) == list(b)
    return a == b


def warn_config_mismatch(ckpt_cfg: dict, cfg: Config) -> None:
    """Warn if structural settings differ from the checkpoint being resumed.

    Shape-affecting mismatches fail loudly at load_state_dict anyway; this
    catches the SILENT ones (prediction_target / timesteps / beta_schedule
    change the math without changing any tensor shape).
    """
    structural = ("image_size", "base_channels", "channel_mults", "num_res_blocks",
                  "attention_resolutions", "timesteps", "beta_schedule", "prediction_target")
    for key in structural:
        old = ckpt_cfg.get(key) if ckpt_cfg else None
        if old is not None and not _same(old, getattr(cfg, key)):
            print(f"[resume] WARNING: config '{key}' differs from checkpoint "
                  f"({old!r} -> {getattr(cfg, key)!r}). Shape-affecting changes fail "
                  f"at load; math-affecting ones (timesteps/beta/prediction_target) "
                  f"silently change behavior. Proceed only if intended.")


def keep_last_k(checkpoint_dir: Path, k: int) -> None:
    ckpts = sorted(checkpoint_dir.glob("ckpt_*.pt"))
    for old in ckpts[:-k]:
        old.unlink(missing_ok=True)


def flush_log(log_path: Path, rows: list) -> None:
    """Append buffered CSV rows in ONE open/append/close.

    train.py used to open+write+close the CSV every step. When log_csv is on
    a Drive FUSE mount (Colab) that is a synchronous round-trip per step.
    Rows are now buffered in memory and flushed in bulk every LOG_FLUSH_EVERY
    steps and before every long pause (sampling / checkpointing), so a crash
    loses at most ~50 rows.
    """
    if not rows:
        return
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerows(rows)
    rows.clear()


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="use only N images (smoke test)")
    parser.add_argument("--unconditional", dest="unconditional", action="store_true", default=True)
    parser.add_argument("--conditional", dest="unconditional", action="store_false",
                        help="phase 2 only: enable caption path (not implemented yet)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="e.g. checkpoints/latest.pt")
    parser.add_argument("--max_vram_gb", type=float, default=None)
    parser.add_argument("--prediction_target", type=str, default=None,
                        choices=("epsilon", "v_prediction"))
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="override output dir for checkpoints (e.g. a Google Drive path on Colab)")
    parser.add_argument("--samples_dir", type=str, default=None)
    parser.add_argument("--log_csv", type=str, default=None)
    parser.add_argument("--sample_every", type=int, default=None,
                        help="steps between fixed-seed sample grids; raise on fast GPUs "
                             "to cut wall-clock overhead")
    args = parser.parse_args()

    cfg = Config()
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.grad_accum_steps is not None: cfg.grad_accum_steps = args.grad_accum_steps
    if args.max_train_steps is not None: cfg.max_train_steps = args.max_train_steps
    if args.image_size is not None: cfg.image_size = args.image_size
    cfg.unconditional = args.unconditional
    if args.max_vram_gb is not None: cfg.max_vram_gb = args.max_vram_gb
    if args.prediction_target is not None: cfg.prediction_target = args.prediction_target
    if args.checkpoint_dir is not None: cfg.checkpoint_dir = args.checkpoint_dir
    if args.samples_dir is not None: cfg.samples_dir = args.samples_dir
    if args.log_csv is not None: cfg.log_csv = args.log_csv
    if args.sample_every is not None: cfg.sample_every = args.sample_every

    if not cfg.unconditional:
        # Phase-2 seam: the dataset already returns captions when
        # unconditional=False, but no text encoder / cross-attention is built
        # in this pass. Fail loudly rather than silently training wrong.
        sys.exit("--conditional is reserved for phase 2 (caption conditioning); "
                 "not implemented in this pass. Run unconditional training.")

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cpu":
        sys.exit("No CUDA device found — run check_env.py first (see setup.md).")

    # Conv autotuning: all batches have identical shapes (drop_last=True), so
    # cudnn.benchmark's one-time kernel search pays off every step after.
    torch.backends.cudnn.benchmark = True

    vram_guard(cfg, cfg.max_vram_gb if cfg.max_vram_gb is not None
               else torch.cuda.get_device_properties(0).total_memory / 1024**3)

    # ------------------------------ data ------------------------------
    images_dir, captions_csv = ensure_data()
    dataset = PixelArtDataset(
        images_dir, captions_csv,
        image_size=cfg.image_size,
        unconditional=cfg.unconditional,
        background=cfg.background,
        limit=args.limit,
    )
    print(f"dataset: {len(dataset)} images "
          f"({'unconditional — captions ignored' if cfg.unconditional else 'captioned'})")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )

    # ------------------------------ model ------------------------------
    model = build_model(cfg).to(device)
    # NHWC layout: feeds tensor cores in their preferred format (~1.2-1.5x on
    # conv-heavy nets). Weights + inputs must both be channels_last.
    model = model.to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.2f}M params, prediction_target={cfg.prediction_target}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    ema = EMA(model, cfg.ema_decay)
    diffusion = Diffusion(cfg.timesteps, cfg.beta_schedule, cfg.prediction_target, device)

    # ------------------------------ resume ------------------------------
    start_step = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])   # restores Adam m/v — no loss spike on resume
        scaler.load_state_dict(ckpt["scaler"])
        ema = EMA(model, cfg.ema_decay)
        raw_ema = ckpt["ema"]
        if isinstance(raw_ema, dict) and "shadow" in raw_ema:
            # current format: {"shadow": ..., "num_updates": ...}
            ema.shadow = {k: v.to(device) for k, v in raw_ema["shadow"].items()}
            ema.num_updates = int(raw_ema.get("num_updates", 0))
        else:
            # legacy format (pre-fix checkpoints): bare tensor dict.
            # Seed the warmup counter with the resume step — otherwise it
            # restarts at 0 and min(decay,(1+n)/(10+n)) runs at tiny effective
            # decay for ~100 updates, washing the restored shadow out with raw
            # weights (first post-resume grids would be raw-weight quality).
            ema.shadow = {k: v.to(device) for k, v in raw_ema.items()}
            ema.num_updates = ckpt["step"]
        start_step = ckpt["step"]
        warn_config_mismatch(ckpt.get("config", {}), cfg)
        print(f"resumed from {args.resume_from} at step {start_step} "
              f"(optimizer + AMP scaler + EMA state restored)")

    checkpoint_dir = Path(cfg.checkpoint_dir)
    samples_dir = Path(cfg.samples_dir)
    log_path = Path(cfg.log_csv)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists() and start_step == 0:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "loss", "lr", "elapsed_s"])
    log_buffer: list = []
    LOG_FLUSH_EVERY = 50

    # ------------------------------ train ------------------------------
    model.train()
    step = start_step
    t0 = time.time()
    data_iter = iter(loader)
    print(f"training {cfg.max_train_steps - start_step} steps "
          f"(effective batch = {cfg.batch_size * cfg.grad_accum_steps})")

    while step < cfg.max_train_steps:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = torch.zeros((), device=device)   # stays on GPU — no sync per micro-batch
        for _ in range(cfg.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            x0 = batch["image"].to(device, non_blocking=True)
            x0 = x0.to(memory_format=torch.channels_last)

            # Mixed precision is ON by default — required to fit 4GB comfortably.
            with torch.amp.autocast("cuda"):
                loss = diffusion.training_losses(model, x0) / cfg.grad_accum_steps
            scaler.scale(loss).backward()
            accum_loss += loss.detach()

        # single GPU sync per optimizer step (was one .item() per micro-batch)
        loss_value = accum_loss.item()

        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        step += 1

        if step % 10 == 0 or step == start_step + 1:
            lr_now = optimizer.param_groups[0]["lr"]
            # average includes sampling/checkpoint pauses so the ETA is honest wall-clock
            sps = (time.time() - t0) / max(step - start_step, 1)
            eta = format_eta((cfg.max_train_steps - step) * sps)
            print(f"step {step:6d} | loss {loss_value:.4f} | lr {lr_now:.2e} | "
                  f"{sps:.2f} s/step | eta {eta}")

        # buffered CSV: one FUSE round-trip every LOG_FLUSH_EVERY steps,
        # not one per step
        log_buffer.append([step, f"{loss_value:.6f}",
                           optimizer.param_groups[0]["lr"],
                           f"{time.time() - t0:.1f}"])
        if step % LOG_FLUSH_EVERY == 0:
            flush_log(log_path, log_buffer)

        # OPTIONAL WANDB HOOK: if cfg.wandb_project is set and wandb is
        # installed, a wandb.log({"loss": loss_value, "step": step}) call goes
        # here. Deliberately not wired — logging stays local by default.

        # ------------- fixed-seed qualitative samples (primary eval) -------------
        if step % cfg.sample_every == 0 or step == start_step + 1:
            flush_log(log_path, log_buffer)   # don't lose rows across the multi-minute pause
            # Release cached training activation blocks FIRST: on small-VRAM
            # GPUs the batch-16 sampling activations + the EMA model copy can
            # otherwise spill into shared system memory, slowing sampling ~4x.
            torch.cuda.empty_cache()
            ema_model = build_model(cfg).to(device).to(memory_format=torch.channels_last)
            ema.copy_to(ema_model)
            grid_n = cfg.sample_grid * cfg.sample_grid
            imgs = diffusion.sample(ema_model, grid_n, cfg.image_size, seed=cfg.sample_seed)
            # fixed seed => same initial noise every time; grids are directly
            # comparable across steps so you can watch sprites emerge.
            out = samples_dir / f"step_{step:05d}.png"
            save_image(imgs, out, nrow=cfg.sample_grid)
            print(f"  saved sample grid -> {out}")
            del ema_model, imgs
            torch.cuda.empty_cache()

        # ------------- checkpointing (resumable, keep last k + latest) -------------
        if step % cfg.ckpt_every == 0 or step == cfg.max_train_steps:
            flush_log(log_path, log_buffer)   # rows survive even if a Drive copy fails
            save_checkpoint(cfg, checkpoint_dir, model, optimizer, scaler, ema, step)
            keep_last_k(checkpoint_dir, cfg.keep_last_k)
            print(f"  checkpoint saved (step {step})")

    flush_log(log_path, log_buffer)
    print("done.")


if __name__ == "__main__":
    main()
