# hcsd-project

Inference experiments with **StyleGAN2-ADA** (PyTorch) and **CLIP**.
The environment works on a local machine (GPU or CPU), Google Colab and Kaggle.
No training dependencies or CUDA toolkit (`nvcc`) are needed.

## Layout

```
pyproject.toml                  # single source of truth for dependencies (managed with uv)
uv.lock                         # locked versions
requirements-colab.txt          # exported from uv.lock, without torch (Colab/Kaggle ship their own)
setup_env.py                    # one-cell setup that detects local / Colab / Kaggle
scripts/export_colab_requirements.sh
src/generators/stylegan2_ada/   # vendored StyleGAN2-ADA (inference parts only)
src/generators/sg2_path.py      # puts the vendored code on sys.path
examples/generate.py            # generate images from a pretrained .pkl
examples/clip_rank.py           # score/rank images against text prompts with CLIP
```

## Setup

### Local (GPU or CPU)

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra local
uv run python examples/generate.py --seeds 0 1 2
uv run python examples/clip_rank.py --images outputs/*.png --prompts "a smiling person" "a person with glasses"
```

The `local` extra installs `torch` and `torchvision`.

### Google Colab / Kaggle

Select a GPU runtime (e.g. T4), then in a notebook cell:

```python
!git clone https://github.com/AdityaDeodeshmukh/hcsd-project.git
%cd hcsd-project
%run setup_env.py
```

`setup_env.py` installs `requirements-colab.txt` with `uv pip install --system` and keeps the pre-installed PyTorch/CUDA.
Then run the examples:

```python
!python examples/generate.py --seeds 0 1 2
!python examples/clip_rank.py --images outputs/*.png --prompts "a smiling person" "a person with glasses"
```

Docker cannot run inside Colab, so Colab and Kaggle use this path.

## Using StyleGAN2-ADA from your own code

```python
from src.generators.sg2_path import add_stylegan2_ada_to_path
add_stylegan2_ada_to_path()

import dnnlib, legacy
with dnnlib.util.open_url("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl") as f:
    G = legacy.load_network_pkl(f)["G_ema"]
```

The upstream code uses top-level imports (`import dnnlib`, `from torch_utils import ...`), and the network pickles reference those module names.
This is why the folder is added to `sys.path` and is not imported as a sub-package.

## Notes

- **Custom CUDA ops:** `bias_act` and `upfirdn2d` try to compile CUDA kernels.
  Without `ninja`/`nvcc` they print a warning and fall back to the reference PyTorch implementation.
  This is slower but fine for inference.
- **CPU:** `examples/generate.py` passes `force_fp32=True` on CPU, because FP16 layers are CUDA-only.
- **Pinned packages on Colab/Kaggle:** these platforms pre-pin some packages (e.g. NumPy).
  If an entry in `requirements-colab.txt` conflicts, remove that entry.
- **Updating dependencies:** edit `pyproject.toml`, then run `uv lock` and `scripts/export_colab_requirements.sh`.

## Vendored code and license

`src/generators/stylegan2_ada/` contains `dnnlib/`, `torch_utils/`, `legacy.py` and `training/networks.py` from
[NVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) at commit `d72cc7d`.
The upstream `LICENSE.txt` is kept.
The **NVIDIA Source Code License is non-commercial**.

Local change: the `click` CLI wrapper in `legacy.py` was removed, so `click` is not a dependency.
