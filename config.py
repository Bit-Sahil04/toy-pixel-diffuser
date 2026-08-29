"""Central configuration for the pixel-art diffusion project.

Every hyperparameter lives here (or can be overridden from train.py CLI
flags) so nothing is scattered across modules. Defaults are chosen for an
RTX 3050 Laptop with 4GB VRAM — deliberately conservative.
"""

from dataclasses import dataclass, asdict, field
import json
from pathlib import Path


@dataclass
class Config:
    # ------------------------------------------------------------------ data
    dataset_id: str = "carlosuperb/lpc-4view-pixel-art-diffusion"
    data_dir: str = "data_raw"          # raw download/extraction cache
    image_size: int = 128               # training resolution (square, multiple of 8).
                                        # Native composite resolution is NOT stated in the
                                        # dataset card — data.py prints actual dims on first
                                        # run; 128 is the safe default for 4GB (a 2x2 LPC
                                        # view composite at native size is likely 128x128).
    background: tuple = (255, 255, 255) # RGBA -> RGB composite background (white, see data.py)
    unconditional: bool = True          # True: ignore captions entirely (phase 1 default).
                                        # Caption conditioning is phase 2 (see train.py).
    num_workers: int = 2

    # ----------------------------------------------------------------- model
    base_channels: int = 64             # width of first UNet level (SD uses 320; keep small)
    channel_mults: tuple = (1, 2, 4)    # depth multiplier per resolution level
    num_res_blocks: int = 2             # residual blocks per level
    attention_resolutions: tuple = (16, 8)  # apply self-attention when feature map is <= this px
    dropout: float = 0.0
    prediction_target: str = "epsilon"  # "epsilon" (implemented) or "v_prediction" (seam, see model.py)

    # ------------------------------------------------------------- diffusion
    timesteps: int = 1000               # DDPM schedule length
    beta_schedule: str = "cosine"       # "cosine" or "linear"

    # --------------------------------------------------------------- training
    batch_size: int = 8                 # conservative for 4GB; raise only after VRAM check
    grad_accum_steps: int = 4           # effective batch = 8 * 4 = 32 without extra memory
    lr: float = 2e-4
    weight_decay: float = 0.0
    max_train_steps: int = 20000
    ema_decay: float = 0.9999
    max_vram_gb: float = 4.0            # soft guard: warn if estimate exceeds this
    vram_safety_fraction: float = 0.85  # warn if heuristic estimate > this fraction of VRAM
    seed: int = 42

    # ------------------------------------------------------------ checkpointing
    checkpoint_dir: str = "checkpoints"
    ckpt_every: int = 500               # steps between full checkpoints
    keep_last_k: int = 3                # numbered checkpoints to keep (latest.pt is permanent)

    # ------------------------------------------------------------- monitoring
    sample_every: int = 200             # steps between fixed-seed sample grids (< ckpt_every)
    sample_grid: int = 4                # 4x4 grid per sample step
    sample_seed: int = 1234             # FIXED across runs so grids are comparable over time
    samples_dir: str = "samples"
    log_csv: str = "logs/train_log.csv"

    # Optional hook (not wired): set a run name here if you later add wandb —
    # the single call site in train.py is marked "# OPTIONAL WANDB HOOK".
    wandb_project: str = ""

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Config":
        with open(path) as f:
            d = json.load(f)
        # tuples arrive from JSON as lists; coerce back where it matters
        for key in ("channel_mults", "attention_resolutions", "background"):
            if key in d and isinstance(d[key], list):
                d[key] = tuple(d[key])
        return cls(**d)
