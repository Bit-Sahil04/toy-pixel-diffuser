"""First sanity check before anything else: run `python check_env.py`.

Prints CUDA availability, GPU name, total VRAM, and verifies that mixed
precision (torch.cuda.amp / torch.amp) actually runs a small matmul on this
machine — not just that the API exists.
"""

import sys


def main() -> int:
    import torch

    print(f"python      : {sys.version.split()[0]}")
    print(f"torch       : {torch.__version__}")
    try:
        import torchvision
        print(f"torchvision : {torchvision.__version__}")
    except ImportError:
        print("torchvision : MISSING")

    cuda_ok = torch.cuda.is_available()
    print(f"CUDA        : {cuda_ok}")
    if not cuda_ok:
        print("\n!! CUDA is NOT available — training will run on CPU (or fail).")
        print("   If you have an NVIDIA GPU, reinstall torch with the CUDA build:")
        print("   see setup.md, section 'Install PyTorch (CUDA)'.")
        return 1

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1024**3
    print(f"GPU         : {props.name}")
    print(f"total VRAM  : {total_gb:.2f} GB")
    if total_gb < 3.5:
        print("WARNING: less VRAM than expected (RTX 3050 has 4GB); "
              "keep batch_size small.")

    # ---- mixed precision check: actually run an autocast matmul on device ----
    try:
        x = torch.randn(64, 64, device="cuda")
        w = torch.randn(64, 64, device="cuda")
        with torch.amp.autocast("cuda"):
            y = x @ w
        assert y.dtype in (torch.float16, torch.bfloat16), f"unexpected dtype {y.dtype}"
        scaler = torch.amp.GradScaler("cuda")
        print(f"AMP         : usable (autocast output dtype = {y.dtype}, GradScaler OK)")
    except Exception as e:  # noqa: BLE001
        print(f"AMP         : FAILED ({type(e).__name__}: {e})")
        return 1

    print("\nAll checks passed — environment is ready for training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
