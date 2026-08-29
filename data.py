"""Dataset handling for the LPC 4-view pixel-art sprite dataset.

Design notes (read before "fixing" anything here):

* The dataset repo is `carlosuperb/lpc-4view-pixel-art-diffusion` on Hugging
  Face (source author is `carlosuperb` — verified 2026-08). `images/train.zip`
  holds 4-view character sprite composites; `captions/captions.csv` holds
  paired captions keyed by `image_path` (e.g. `char_00000.png`).
* The dataset card does NOT state native resolution. We therefore print the
  real dimensions of several samples on first run instead of assuming square
  images — a 4-view composite is likely wider than tall.
* RGBA -> RGB conversion composites onto a FIXED WHITE background. LPC sprite
  sheets use alpha for the area around/between sprites; dropping the channel
  silently would turn those pixels black. White is chosen deliberately so the
  empty parts of the 2x2 composite read as paper-white and the sprite's own
  outline colors stay intact.
* There is NO RandomHorizontalFlip / rotation / any geometric augmentation in
  this pipeline, and that is intentional. These are *labeled directional*
  sprites: flipping a "left-facing" view makes it visually right-facing while
  its caption still says "left", corrupting the text-image pairing. Adding
  flips here would silently poison a future text-conditioned (phase 2) run.
  Do not add flip/rotation augmentation to this dataset.
* Any resize uses NEAREST interpolation only. Bilinear/bicubic blur the hard
  pixel edges that define pixel art and would defeat the whole project.
* Normalization to [-1, 1] lives in normalize()/denormalize() below; both
  training and sampling import these so the two can never drift apart.
* Unconditional mode (default) ignores captions entirely. The caption lookup
  is kept behind `unconditional=False` as the phase-2 seam — no text encoder
  is built or debugged in this pass.
"""

from __future__ import annotations

import csv
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DATA_ROOT = Path("data_raw")
ZIP_NAME = "images/train.zip"
CAPTIONS_NAME = "captions/captions.csv"

# ------------------------------------------------------------------------------
# Normalization: single source of truth for [-1, 1] <-> [0, 1] conversion.
# ------------------------------------------------------------------------------

def normalize(img: torch.Tensor) -> torch.Tensor:
    """[0, 1] float tensor -> [-1, 1] (DDPM convention)."""
    return img * 2.0 - 1.0


def denormalize(img: torch.Tensor) -> torch.Tensor:
    """[-1, 1] float tensor -> [0, 1] (clamped), for saving/inspection."""
    return (img.clamp(-1.0, 1.0) + 1.0) / 2.0


# ------------------------------------------------------------------------------
# Transforms. NOTE: deliberately no flip / rotation — see module docstring.
# ------------------------------------------------------------------------------

def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        # NEAREST only: bilinear/bicubic blur pixel-art edges (see docstring).
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Lambda(normalize),
    ])


def load_image_rgb(path: Path, background: tuple = (255, 255, 255)) -> Image.Image:
    """Open image, compositing RGBA onto a fixed background color."""
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, tuple(background) + (255,))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


# ------------------------------------------------------------------------------
# Dataset download: try load_dataset first, fall back to manual zip handling.
# ------------------------------------------------------------------------------

def download_manual() -> tuple[Path, Path]:
    """Manually fetch train.zip + captions.csv via huggingface_hub and extract."""
    from huggingface_hub import hf_hub_download

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    zip_local = hf_hub_download(
        repo_id="carlosuperb/lpc-4view-pixel-art-diffusion",
        filename=ZIP_NAME,
        repo_type="dataset",
    )
    cap_local = hf_hub_download(
        repo_id="carlosuperb/lpc-4view-pixel-art-diffusion",
        filename=CAPTIONS_NAME,
        repo_type="dataset",
    )
    images_dir = DATA_ROOT / "images" / "train"
    if not images_dir.exists():
        images_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {zip_local} -> {images_dir.parent} ...")
        with zipfile.ZipFile(zip_local) as zf:
            zf.extractall(images_dir.parent)
    captions_csv = DATA_ROOT / "captions.csv"
    if not captions_csv.exists():
        shutil.copy(cap_local, captions_csv)
    return images_dir, captions_csv


def ensure_data() -> tuple[Path, Path | None]:
    """Return (images_dir, captions_csv_or_None). Uses load_dataset if possible."""
    try:
        import datasets as hf_datasets
        hf_datasets.load_dataset("carlosuperb/lpc-4view-pixel-art-diffusion")
        # If load_dataset succeeded it caches the underlying files; the manual
        # path below reuses those cached downloads via hf_hub_download, so this
        # costs nothing extra while giving us a stable on-disk layout we control.
        print("[data] load_dataset resolved the repo; using manual file layout on top of its cache.")
    except Exception as e:  # noqa: BLE001
        print(f"[data] load_dataset failed ({type(e).__name__}: {e}); falling back to manual download.")
    return download_manual()


# ------------------------------------------------------------------------------
# torch Dataset
# ------------------------------------------------------------------------------

class PixelArtDataset(Dataset):
    """4-view LPC pixel-art sprites. Unconditional by default.

    When `unconditional=False` (phase 2), __getitem__ also returns the raw
    caption string. No tokenization/text encoder is wired in this pass.
    """

    def __init__(
        self,
        images_dir: Path | str,
        captions_csv: Path | str | None = None,
        image_size: int = 128,
        unconditional: bool = True,
        background: tuple = (255, 255, 255),
        limit: int | None = None,
    ):
        self.images_dir = Path(images_dir)
        self.unconditional = unconditional
        self.background = background
        self.transform = build_transform(image_size)

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
        self.samples = sorted(
            p for p in self.images_dir.rglob("*") if p.suffix.lower() in exts
        )
        if limit is not None:
            self.samples = self.samples[:limit]
        if not self.samples:
            raise RuntimeError(f"No images found under {self.images_dir}")

        self.captions: dict[str, str] = {}
        if not unconditional:
            if captions_csv is None:
                raise ValueError("captions_csv is required when unconditional=False")
            with open(captions_csv, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.captions[Path(row["image_path"]).name] = row["text"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path = self.samples[idx]
        img = load_image_rgb(path, self.background)
        img = self.transform(img)  # -> float32 [-1, 1], shape (3, S, S)
        item = {"image": img, "path": str(path)}
        if not self.unconditional:
            item["caption"] = self.captions.get(path.name, "")
        return item


# ------------------------------------------------------------------------------
# Standalone inspection: `python data.py`
# ------------------------------------------------------------------------------

def inspect(n: int = 5, limit: int | None = None) -> None:
    images_dir, captions_csv = ensure_data()
    print(f"\nimages_dir: {images_dir}")
    print(f"captions:   {captions_csv}")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in exts)
    print(f"total images found: {len(files)}")

    print(f"\n--- actual dimensions / modes of first {n} images (pre-resize) ---")
    for p in files[:n]:
        img = Image.open(p)
        print(f"  {p.name}: size={img.size} (W x H), mode={img.mode}")

    if captions_csv and captions_csv.exists():
        with open(captions_csv, newline="", encoding="utf-8") as f:
            first = next(csv.DictReader(f), None)
        if first:
            print("\n--- sample caption ---")
            print(f"  {first['image_path']}: {first['text'][:140]}...")

    # Confirm the transform policy: no geometric augmentation allowed.
    tf = build_transform(128)
    kinds = [type(t).__name__ for t in tf.transforms]
    forbidden = {"RandomHorizontalFlip", "RandomVerticalFlip", "RandomRotation",
                 "RandomAffine", "RandomResizedCrop"}
    assert not (forbidden & set(kinds)), f"FORBIDDEN geometric augmentation present: {kinds}"
    print("\ntransform stages:", kinds)
    print("OK: no flip/rotation transforms present (deliberate — see data.py docstring).")

    if limit is None and files:
        limit = len(files)

    ds = PixelArtDataset(images_dir, captions_csv, image_size=128, unconditional=True, limit=limit)
    x = ds[0]["image"]
    print(f"\ndataset __getitem__: shape={tuple(x.shape)}, dtype={x.dtype}, "
          f"min={x.min():.2f}, max={x.max():.2f} (expected range [-1, 1])")


if __name__ == "__main__":
    inspect()
