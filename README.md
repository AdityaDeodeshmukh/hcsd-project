# hcsd-project

Sample latents from a StyleGAN2-ADA generator, embed the images with CLIP, and score them on text-defined attributes (for example *smiling* vs *neutral*). The scores can then be used to study or edit directions in W space.

For each model in the config, a run writes:

```
outputs/<model_name>/
├── images/              PNGs of the first run.n_images latents
├── grid.png             thumbnail grid of those images
├── w.npy                [n_latents, w_dim] sampled W vectors
├── embeddings.npy       [n_latents, D] L2-normalized CLIP image embeddings
├── attribute_scores.json  prompts + P(target) per latent, per attribute
└── config_used.yaml     the exact config that produced the run
```

Row `k` of `embeddings.npy` and every score list corresponds to `w[k]`.

---

## Pipeline

1. `sample_w` draws z with the given seed, maps it to W with truncation `psi`.
2. `synthesize` renders uint8 HxWx3 images.
3. `ClipFeaturizer` encodes the images (batched) and caches the text features.
4. For each `[target, neutral]` prompt pair, `P(target)` is a softmax over the two cosine similarities, scaled by CLIP's `logit_scale`.

---

## Requirements

- Python 3.9+ (3.10+ recommended)
- A GPU is strongly recommended. CPU works but is slow.
- Python packages: `torch`, `torchvision`, `open_clip_torch`, `numpy`, `pillow`, `pyyaml`, `tqdm`
- Optional: `ninja` and a CUDA toolkit matching `torch.version.cuda`, to compile StyleGAN2-ADA's fast CUDA kernels (see [Troubleshooting](#troubleshooting))
- The StyleGAN2-ADA code is included in this repository (vendored); no separate clone is needed

---

## Setup

Replace `<YOUR_REPO_URL>` with this project's repository URL.

### Local

```bash
!git clone -q -b <branch> https://github.com/AdityaDeodeshmukh/hcsd-project.git
cd hcsd-project

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# Install torch for your CUDA version first: https://pytorch.org/get-started/locally/
pip install torch torchvision

pip install open_clip_torch numpy pillow pyyaml tqdm
pip install -e .                     # if the project has a pyproject.toml / setup.py
```

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available())"
hcsd-project --list
```

### Google Colab

1. **Runtime → Change runtime type → GPU** (T4 is enough).
2. In a cell:

```python
# (Optional) keep outputs after the session ends
from google.colab import drive
drive.mount("/content/drive")

!git clone -q -b <branch> https://github.com/AdityaDeodeshmukh/hcsd-project.git
%cd /content/hcsd-project

# torch and torchvision are preinstalled on Colab; do not reinstall them
!pip install -q open_clip_torch pyyaml tqdm ninja
!pip install -q -e .

```

3. To save outputs on Drive, set `paths.outputs_dir` to a folder under `/content/drive/MyDrive/...`.
4. Run:

```python
import sys
sys.path.insert(0, "/content/drive/hcsd-project/src")
```
```python
from hcsd_project.config import load_config
from hcsd_project.cli import run_model

# Change the overrides based on your experiment
cfg = load_config(overrides=[
    "paths.models_dir=/content/drive/MyDrive/hcsd/models",
    "paths.outputs_dir=/content/drive/MyDrive/hcsd/outputs",
    "run.n_latents=1000",
    "run.n_images=4",
    "run.seed=56"
])

run_model(cfg, "ffhq")
```

Colab sessions are temporary. Anything not written to Drive is lost when the runtime disconnects.

### Kaggle

1. In the notebook sidebar, set **Accelerator** to a GPU (T4 or P100) and turn **Internet** on (needed to clone this repo, install packages and download weights).
2. In a cell:

```python
!git clone -q -b <branch> https://github.com/AdityaDeodeshmukh/hcsd-project.git
%cd /kaggle/working/hcsd-project

# torch and torchvision are preinstalled on Kaggle
!pip install -q open_clip_torch pyyaml tqdm ninja
!pip install -q -e .

```

3. Set `paths.outputs_dir` to a path under `/kaggle/working/` (for example `/kaggle/working/outputs`). Only `/kaggle/working` is kept as notebook output.
4. Run:

```python
import sys
sys.path.insert(0, "/kaggle/working/hcsd-project/src")
```
```python
from hcsd_project.config import load_config
from hcsd_project.cli import run_model

# Change the overrides based on your experiment
cfg = load_config(overrides=[
    "paths.models_dir=/content/drive/MyDrive/hcsd/models",
    "paths.outputs_dir=/content/drive/MyDrive/hcsd/outputs",
    "run.n_latents=1000",
    "run.n_images=4",
    "run.seed=56"
])

run_model(cfg, "ffhq")
```

If you have the weights as a Kaggle Dataset, add it to the notebook and point `models.<name>.weights` at the path under `/kaggle/input/...`, so nothing needs to be downloaded.

---

## Configuration

- `configs/config.yaml`: run settings, output paths, models and featurizers
- `configs/prompts.yaml`: the CLIP attribute prompt pairs (`[target, neutral]`)

Any value can be overridden from the command line with `--set`, for example `--set run.n_images=4`. Keep target prompts unique, because two pairs with the same target overwrite each other in `attribute_scores.json`.

---

## Usage

```bash
# list configured models
hcsd-project --list

# download weights only and print checksums
hcsd-project --download-only --models ffhq

# run with the default config
hcsd-project

# override values from the command line
hcsd-project --set run.n_latents=5000 --set run.psi=0.6
hcsd-project --models ffhq --set run.n_images=16
```

---

## Reading the scores

- CLIP image-text cosine similarities are small (about 0.2 to 0.35), and two competing prompts typically differ by 0.01 to 0.05. The scores are therefore computed with CLIP's `logit_scale` (about 100). Without it every score is about 0.5.
- If all scores are still near 0.5, check that you are running the current `ClipFeaturizer` and print `featurizer.model.logit_scale.exp()` (expect about 100).
- Prompt wording matters a lot. "a photo of a person smiling" vs "a photo of a person with a neutral expression" is usually more reliable than short "face" prompts. Open `grid.png` and the saved PNGs, and compare them against the scores, to confirm the prompts separate what you intend.
- For fitting directions in W space, the raw gap `s_target - s_neutral` (z-scored) is often better than the probability, which saturates near 0 and 1.

---

## Third-party code and licenses

This project depends on third-party code and model weights that have their own licenses. They are not covered by this project's license, and you are responsible for complying with them.

| Component | Used for | License |
|---|---|---|
| [StyleGAN2-ADA (PyTorch)](https://github.com/NVlabs/stylegan2-ada-pytorch) | generator code (`dnnlib`, `legacy`, `torch_utils`, `training`) | NVIDIA Source Code License (research and non-commercial use only) |
| StyleGAN2-ADA pretrained weights | generator networks | separate from the code; check the license of the specific pickle you download |
| [OpenCLIP](https://github.com/mlfoundations/open_clip) | CLIP model loading and encoding | MIT |
| CLIP pretrained weights (`openai`, `laion*`, ...) | image and text encoders | depends on the `pretrained` tag; OpenAI CLIP is MIT, LAION weights list theirs on the model card |
| PyTorch, torchvision, NumPy, Pillow, PyYAML, tqdm | runtime | BSD-style, HPND or MPL-2.0 (permissive) |

### Vendored StyleGAN2-ADA code

A copy of the StyleGAN2-ADA PyTorch code is included in this repository. When keeping or updating it:

- Keep its `LICENSE.txt` and all file headers unchanged, in the same directory as the code.
- Record the upstream repo URL and the commit hash it was copied from (for example in a README next to the vendored code).
- Mark any files you modified, since the license requires that changes are identified.
- Do not commit the pretrained `.pkl` weights unless their license allows redistribution.

Because the StyleGAN2-ADA license restricts use to non-commercial research, code paths that depend on it inherit that restriction. Check the license text shipped with the vendored code before any commercial use.

This section is a summary, not legal advice.

---

## Troubleshooting

**`Setting up PyTorch plugin "upfirdn2d_plugin"... Failed!`**
StyleGAN2-ADA could not compile its custom CUDA kernels and fell back to slower reference ops. Outputs are still correct. To silence it, set `custom_ops.verbosity = "none"` after importing `torch_utils.custom_ops` in the generator wrapper. To fix the build, install `ninja`, make sure `nvcc` matches `torch.version.cuda`, and have a host C++ compiler. On recent PyTorch or newer GPUs the original repo often fails regardless, and a torch 2.x fork helps.

**Out of GPU memory**
Lower `CHUNK` in the CLI (default 8) or `batch` in `synthesize`.

**`.mapping(z, None, ...)` fails**
The pickle is a conditional model (`G.c_dim > 0`). The wrapper currently passes no class label, so it supports unconditional models such as FFHQ.

**Scores are all `{}` or the JSON has no scores**
The featurizer found no attribute prompts. Check `run.prompts_file` and that the file has `clip: attribute_prompt_list:` with at least one two-string pair.

**Colab or Kaggle can't clone or download**
Make sure Internet is enabled (Kaggle) and the runtime is connected (Colab).
