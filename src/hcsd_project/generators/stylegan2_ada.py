import sys
import warnings
from pathlib import Path

import numpy as np
import torch

from hcsd_project.config import resolve


# Hide the "Failed to build CUDA kernels" warning (the reference ops are used instead).
warnings.filterwarnings("ignore", message="Failed to build CUDA kernels")
def _ensure_path(repo_dir):
    p = str(repo_dir)
    if p not in sys.path:
        sys.path.insert(0, p)


class StyleGAN2ADA:
    def __init__(self, pkl, repo_dir, device=None):
        _ensure_path(repo_dir)
        import dnnlib, legacy  # noqa: E401
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        with dnnlib.util.open_url(str(pkl)) as f:
            G = legacy.load_network_pkl(f)["G_ema"]
        self.G = G.eval().requires_grad_(False).to(self.device)
        self.z_dim, self.w_dim, self.num_ws = G.z_dim, G.w_dim, G.num_ws

    @classmethod
    def from_config(cls, cfg, name):
        from hcsd_project.download_models import ensure_weights
        m = cfg["models"][name]
        pkl = ensure_weights(cfg, m["weights"])
        return cls(pkl, resolve(m["repo"]), cfg["run"].get("device"))

    @torch.no_grad()
    def sample_w(self, n, seed, psi=1.0):
        """Returns W of shape [n, w_dim] (all layers are identical before editing)."""
        z = np.random.RandomState(seed).randn(n, self.z_dim)
        z = torch.from_numpy(z).float().to(self.device)
        return self.G.mapping(z, None, truncation_psi=psi)[:, 0, :]

    def to_wplus(self, w):
        """[n, w_dim] -> [n, num_ws, w_dim]"""
        return w[:, None, :].repeat(1, self.num_ws, 1)

    @torch.no_grad()
    def synthesize(self, w, batch=8):
        """w: [n, w_dim] or [n, num_ws, w_dim] -> uint8 images [n, H, W, 3]"""
        if w.ndim == 2:
            w = self.to_wplus(w)
        w = w.to(self.device)
        fp32 = self.device.type == "cpu"   # fp16 layers are slow or unsupported on CPU
        out = []
        for i in range(0, len(w), batch):
            img = self.G.synthesis(w[i:i + batch], noise_mode="const", force_fp32=fp32)
            img = (img.clamp(-1, 1) + 1) * 127.5
            out.append(img.permute(0, 2, 3, 1).to(torch.uint8).cpu())
        return torch.cat(out)
