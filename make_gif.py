"""Stitch samples/step_*.png grids into a GIF of training progression.

Usage:
    python make_gif.py                          # -> samples/training_progress.gif
    python make_gif.py --fps 5 --out out.gif
"""

import argparse
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples_dir", type=str, default="samples")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--fps", type=float, default=4.0, help="frames per second")
    parser.add_argument("--size", type=int, default=512, help="output GIF width in px")
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    frames = sorted(samples_dir.glob("step_*.png"))
    if not frames:
        raise SystemExit(f"No step_*.png files found in {samples_dir} — run training first.")

    out = Path(args.out) if args.out else samples_dir / "training_progress.gif"
    duration_ms = int(1000 / args.fps)

    pil_frames = []
    for f in frames:
        img = Image.open(f).convert("RGB")
        h = round(img.height * args.size / img.width)
        pil_frames.append(img.resize((args.size, h), Image.NEAREST))

    pil_frames[0].save(
        out, save_all=True, append_images=pil_frames[1:],
        duration=duration_ms, loop=0,
    )
    print(f"wrote {out} with {len(pil_frames)} frames "
          f"(steps {frames[0].stem} .. {frames[-1].stem})")


if __name__ == "__main__":
    main()
