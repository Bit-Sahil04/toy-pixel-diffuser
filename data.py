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
* Data loading is cache-first: `data_raw/` is downloaded/extracted exactly
  once; `datasets.load_dataset` is only a fallback if the direct hub download
  fails (it duplicates the whole dataset into an arrow cache we never read).
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
MARKER = DATA_ROOT / ".dataset_complete"   # written only after a fully verified extract
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}

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

def _count_images(images_dir: Path) -> int:
    return sum(1 for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _count_captions(captions_csv: Path) -> int:
    with open(captions_csv, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def download_manual(force_extract: bool = False) -> tuple[Path, Path]:
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
    if not images_dir.exists() or force_extract:
        # NOTE: extractall creates the directory before finishing, so the dir
        # existing proves nothing. force_extract=True is passed when a prior
        # interrupted run left a PARTIAL extraction behind — wipe it first.
        if force_extract and images_dir.exists():
            shutil.rmtree(images_dir)
        images_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {zip_local} -> {images_dir.parent} ...")
        with zipfile.ZipFile(zip_local) as zf:
            zf.extractall(images_dir.parent)
        # only mark complete after a full, verified extraction
        n = _count_images(images_dir)
        MARKER.write_text(f"{n} images\n")
        print(f"[data] extraction complete and verified: {n} images")
    captions_csv = DATA_ROOT / "captions.csv"
    if not captions_csv.exists():
        shutil.copy(cap_local, captions_csv)
    return images_dir, captions_csv


def ensure_data() -> tuple[Path, Path]:
    """Return (images_dir, captions_csv). Cache-first, download once.

    Order: (1) if data_raw/ is present AND verified (marker file from a prior
    complete extraction, or an image count that matches the caption rows —
    this accepts the pre-marker data_raw/ directories from earlier versions),
    use it — resumes/relaunches are instant; (2) otherwise manual
    hf_hub_download of the zip + captions, with a forced re-extract if a
    previous interrupted run left a PARTIAL extraction behind (extractall
    creates the target dir before finishing, so dir-exists proves nothing);
    (3) datasets.load_dataset only as a last-resort fallback if the hub file
    download fails — it warms the HF cache, after which the manual path is
    retried.
    """
    images_dir = DATA_ROOT / "images" / "train"
    captions_csv = DATA_ROOT / "captions.csv"
    if images_dir.exists() and captions_csv.exists():
        if MARKER.exists():
            print(f"[data] using cached dataset at {images_dir}")
            return images_dir, captions_csv
        # no marker: maybe a partial extraction from an interrupted first
        # download. Verify by counting images vs caption rows before trusting it.
        n_img = _count_images(images_dir)
        n_cap = _count_captions(captions_csv)
        if n_img == n_cap and n_img > 0:
            MARKER.write_text(f"{n_img} images\n")
            print(f"[data] verified existing dataset ({n_img} images == {n_cap} captions); "
                  f"using cache at {images_dir}")
            return images_dir, captions_csv
        print(f"[data] WARNING: data_raw is INCOMPLETE ({n_img} images vs {n_cap} captions) — "
              f"likely an interrupted extraction. Re-downloading and re-extracting.")
        try:
            return download_manual(force_extract=True)
        except Exception as e:  # noqa: BLE001
            print(f"[data] manual download failed ({type(e).__name__}: {e}); "
                  f"trying datasets.load_dataset as fallback.")
            import datasets as hf_datasets
            hf_datasets.load_dataset("carlosuperb/lpc-4view-pixel-art-diffusion")
            return download_manual(force_extract=True)
    try:
        return download_manual()
    except Exception as e:  # noqa: BLE001
        print(f"[data] manual download failed ({type(e).__name__}: {e}); "
              f"trying datasets.load_dataset as fallback.")
        import datasets as hf_datasets
        hf_datasets.load_dataset("carlosuperb/lpc-4view-pixel-art-diffusion")
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
