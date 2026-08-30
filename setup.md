# Setup

Environment for the pixel-art diffusion training repo. Target machine:
RTX 3050 Laptop, 4GB VRAM, Windows.

## 1. Python

Requires **Python >= 3.10** (this machine has 3.13.12 via the system
`py` launcher). The CUDA wheels we pin do not build for 3.9.

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate          # Windows (bash: source .venv/Scripts/activate)
python -m pip install --upgrade pip
```

## 2. Install PyTorch (CUDA)

Verified 2026-08-30 against pytorch.org + download.pytorch.org: the current
stable build is **torch 2.13.0 + cu130** (CUDA 13.0), with matching
**torchvision 0.28.0**. cu130 supports Ampere (sm_86), which is the RTX 3050.
Driver version 610.88 on this machine supports CUDA 13 — no driver update needed.

```bash
pip install -r requirements.txt
```

or explicitly:

```bash
pip install torch==2.13.0+cu130 torchvision==0.28.0+cu130 \
    --index-url https://download.pytorch.org/whl/cu130
pip install datasets pillow numpy tqdm matplotlib
```

**Fallback for Python 3.9** (if you cannot change interpreters): the last CUDA
build supporting 3.9 is torch 2.8.0+cu128 / torchvision 0.23.0+cu128:

```bash
pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
```

No wandb or other trackers — logging is local (CSV + PNG sample grids).

## 3. Sanity check (run this FIRST)

```bash
python check_env.py
```

Expected output: CUDA `True`, GPU name `NVIDIA GeForce RTX 3050 Laptop GPU`,
~4.00 GB VRAM, and `AMP: usable`.

## 4. Dataset

`data.py` pulls `carlosuperb/lpc-4view-pixel-art-diffusion` from Hugging Face
(~300 MB zip: 4-view character sprite composites + captions). First run
downloads and caches to `data_raw/`, then prints the **actual** image
dimensions/modes (the dataset card does not state resolution — never assume,
inspect):

```bash
python data.py
```

Verified on first run (2026-08-30): **50,000 images, all 128 x 128 RGBA**, so
the default `image_size: 128` trains at native resolution with no resampling.

Notes baked into the code (see data.py docstring for the full rationale):

* RGBA sprites are composited onto a fixed **white** background, not silently
  converted.
* **No flip/rotation augmentation, on purpose** — flipping a left-facing
  sprite would corrupt the left/right caption pairing (matters for phase 2).
* All resizing uses NEAREST interpolation to preserve pixel-art edges.
* Images are normalized to [-1, 1] via `data.normalize()`; sampling uses the
  matching `data.denormalize()` — the two can't drift apart.
* Unconditional training is the default (`captions.csv` ignored). Caption
  conditioning is phase 2 (`--conditional` fails loudly for now).

## 5. Training

```bash
python train.py                              # full default run (batch 8 x 4 accum)
python train.py --limit 512 --max_train_steps 60    # smoke test
python train.py --resume_from checkpoints/latest.pt # resume after interrupt
```

Defaults are sized for 4GB: batch_size 8, image_size 128, base_channels 64,
channel_mults (1,2,4), AMP on. A `--max_vram_gb` soft guard prints a warning
at startup if the configured run looks likely to OOM (heuristic estimate for
the defaults: ~1.8 GB).

Timing note (measured on this 3050): training is ~1-2 s/step at the defaults,
but each 4x4 sample grid runs a full 1000-step DDPM ancestral pass (~15-20
min) — `sample_every: 200` therefore adds roughly 8-10% sampling overhead.
Run `python -u train.py ...` when redirecting to a log file so progress
prints appear immediately (stdout is block-buffered otherwise).

Outputs:

| Path                        | What                                              |
|-----------------------------|---------------------------------------------------|
| `logs/train_log.csv`        | loss per step (local, no dashboard)               |
| `samples/step_XXXXX.png`    | fixed-seed 4x4 sample grid every 200 steps        |
| `checkpoints/latest.pt`     | full resumable state (weights, Adam, AMP, EMA)    |
| `checkpoints/ckpt_*.pt`     | numbered checkpoints, last 3 kept                 |

## 6. Watching progress

```bash
python make_gif.py    # samples/step_*.png -> samples/training_progress.gif
```

Sample grids use a fixed seed, so frame N and frame M show the *same noise*
being denoised at different training stages — you can literally watch a sprite
emerge.
