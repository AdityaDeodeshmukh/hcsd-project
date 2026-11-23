"""Generate images with a pretrained StyleGAN2-ADA network.

Usage (from the repo root):
    python examples/generate.py --seeds 0 1 2 --outdir outputs
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import PIL.Image
import torch

from src.generators.sg2_path import add_stylegan2_ada_to_path

add_stylegan2_ada_to_path()
import dnnlib  # noqa: E402
import legacy  # noqa: E402

DEFAULT_NETWORK = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"


def load_generator(network: str, device: torch.device):
    with dnnlib.util.open_url(network) as f:
        return legacy.load_network_pkl(f)["G_ema"].to(device).eval()


@torch.no_grad()
def generate(G, seed: int, device: torch.device, truncation_psi: float = 0.7) -> PIL.Image.Image:
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    label = torch.zeros([1, G.c_dim], device=device)
    # force_fp32 so the network also runs on CPU (fp16 is CUDA-only).
    img = G(z, label, truncation_psi=truncation_psi, noise_mode="const", force_fp32=device.type != "cuda")
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return PIL.Image.fromarray(img[0].cpu().numpy(), "RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", default=DEFAULT_NETWORK, help="URL or path to a network .pkl")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--trunc", type=float, default=0.7)
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    G = load_generator(args.network, device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        path = outdir / f"seed{seed:04d}.png"
        generate(G, seed, device, args.trunc).save(path)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
