"""Latent sampling and synthesis for StyleGAN2-ADA (W space).

All functions work on single w vectors of shape [N, 252].
They are broadcast to the generator's W+ shape [N, num_ws, 512] only at synthesis time, so an edit can later touch a subset of layers.
"""

import numpy as np
import PIL.Image
import torch

from src.generators.sg2_path import add_stylegan2_ada_to_path

add_stylegan2_ada_to_path()
import dnnlib
import legacy

FFHQ_URL = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"


def load_generator(network: str = FFHQ_URL, device: torch.device | str = "cuda"):
    """Load G_ema from a network pickle (URL or path), in eval mode."""
    with dnnlib.util.open_url(network) as f:
        return legacy.load_network_pkl(f)["G_ema"].to(device).eval()


def sample_z(n: int, seed: int, z_dim: int = 512) -> np.ndarray:
    """n latent codes z from one fixed seed, shape [n, z_dim], float32.
    
    Uses one RandomState stream, so the first k rows for n=100 are the same as the first k rows for n=5000.
    """
    return np.random.RandomState(seed).randn(n, z_dim).astype(np.float32)


@torch.no_grad()
def z_to_w(G, z: np.ndarray, truncation_psi: float = 0.7, batch_size: int = 1000) -> np.ndarray:
    """Map z -> w with truncation. Returns shape [n, w_dim], float32.

    The mapping network broadcasts the same w to all num_ws layers, so only layer 0 is kept.
    """
    device = G.mapping.w_avg.device
    c = torch.zeros([1, G.c_dim], device=device)
    out = []

    for start in range(0, len(z), batch_size):
        zb = torch.from_numpy(z[start : start + batch_size]).to(device)
        ws = G.mapping(zb, c.expand(len(zb), -1), truncation_psi=truncation_psi, skip_w_avg_update=True)
        out.append(ws[:, 0].float().cpu().numpy())

    return np.concatenate(out)


def w_to_ws(G, w: torch.Tensor) -> torch.Tensor:
    """Broadcast w [N, 512] to W+ [N, num_ws, 512]."""
    return w.unsqueeze(1).repeat(1, G.num_ws, 1)


@torch.no_grad()
def synthesize(G, ws: torch.Tensor) -> torch.Tensor:
    """W+ [N, num_ws, 512] -> uint8 images [N, H, W, 3] on the same device.

    noise_mode="const" makes the output deterministic for a given w.
    force_fp32 on CPU, because the fp16 layers are CUDA-only.
    """
    img = G.synthesis(ws, noise_mode="const", force_fp32=ws.device.type != "cuda")
    return (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)


def to_pil(images: torch.Tensor, size: int | None = None) -> list[PIL.Image.Image]:
    """uint8 [N, H, W, 3] tensor -> list of PIL images, optionally resized to size x size."""
    pils = [PIL.Image.fromarray(x) for x in images.cpu().numpy()]
    if size is not None:
        pils = [p.resize((size, size), PIL.Image.LANCZOS) for p in pils]

    return pils


def image_grid(images: list[PIL.Image.Image], cols: int) -> PIL.Image.Image:
    """Paste equally sized images into a grid, row by row."""
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    grid = PIL.Image.new("RGB", (cols * w, rows * h), "white")
    for i, img in enumerate(images):
        grid.paste(img, ((i % cols) * w, (i // cols) * h))

    return grid