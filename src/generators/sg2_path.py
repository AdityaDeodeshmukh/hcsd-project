"""Make the vendored StyleGAN2-ADA code importable.

The upstream code uses top-level imports (``import dnnlib``, ``import legacy``,
``from torch_utils import ...``) and network pickles reference those module
names, so the vendored folder has to be on ``sys.path`` rather than imported as
a sub-package.
"""

import sys
from pathlib import Path

STYLEGAN2_ADA_DIR = Path(__file__).resolve().parent / "stylegan2_ada"


def add_stylegan2_ada_to_path() -> Path:
    path = str(STYLEGAN2_ADA_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
    return STYLEGAN2_ADA_DIR
