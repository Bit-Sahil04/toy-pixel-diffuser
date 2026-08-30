"""Standalone inference: sample sprites from a checkpoint using its EMA weights.

Usage:
    python sample.py                                     # 4x4 grid from checkpoints/latest.pt
    python sample.py --checkpoint checkpoints/latest.pt --n 16 --seed 1234 --out samples/infer.png
    python sample.py --save_individual                   # also write each sprite separately
"""

import argparse
import dataclasses
from pathlib import Path

import torch
from torchvision.utils import save_image

from config import Config
from diffusion import Diffusion
from model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt")
    parser.add_argument("--n", type=int, default=16, help="number of images to sample")
    parser.add_argument("--nrow", type=int, default=4, help="images per row in the grid")
    parser.add_argument("--seed", type=int, default=None,
                        help="defaults to the checkpoint's config sample_seed")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--save_individual", action="store_true",
                        help="additionally save each sprite as its own PNG")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config(**dataclasses.asdict(Config(**ckpt["config"])))
    step = ckpt["step"]

    # Use the saved EMA weights — cleaner samples than raw training weights.
    model = build_model(cfg).to(device)
    raw_ema = ckpt["ema"]
    if isinstance(raw_ema, dict) and "shadow" in raw_ema:
        raw_ema = raw_ema["shadow"]   # current format {"shadow", "num_updates"}
    model.load_state_dict({k: v.to(device) for k, v in raw_ema.items()})
    model.eval()

    diffusion = Diffusion(cfg.timesteps, cfg.beta_schedule, cfg.prediction_target, device)
    seed = args.seed if args.seed is not None else cfg.sample_seed
    imgs = diffusion.sample(model, args.n, cfg.image_size, seed=seed, verbose=True)

    out = Path(args.out) if args.out else Path(cfg.samples_dir) / f"infer_step{step:07d}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(imgs, out, nrow=args.nrow)
    print(f"saved {args.n}x samples (checkpoint step {step}, seed {seed}, EMA weights) -> {out}")

    if args.save_individual:
        ind_dir = out.parent / out.stem
        ind_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(imgs):
            save_image(img, ind_dir / f"sample_{i:03d}.png")
        print(f"saved individual sprites -> {ind_dir}/")


if __name__ == "__main__":
    main()
