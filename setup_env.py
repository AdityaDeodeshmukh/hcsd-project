"""One-cell environment setup for local / Colab / Kaggle.

In a Colab or Kaggle notebook, after cloning the repo:
    %cd hcsd-project
    %run setup_env.py
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def detect_platform() -> str:
    if "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    return "local"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    platform = detect_platform()
    print(f"Platform: {platform}")

    if platform in ("colab", "kaggle"):
        # Use the pre-installed torch/CUDA; install everything else with uv.
        run([sys.executable, "-m", "pip", "install", "-q", "uv"])
        run(["uv", "pip", "install", "--system", "-r", str(ROOT / "requirements-colab.txt")])
    else:
        print("Local machine: run `uv sync --extra local` and then `uv run python <script>`.")

    sys.path.insert(0, str(ROOT))
    from src.generators.sg2_path import add_stylegan2_ada_to_path

    add_stylegan2_ada_to_path()

    import torch

    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
