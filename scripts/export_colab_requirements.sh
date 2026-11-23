#!/usr/bin/env bash
# Regenerate requirements-colab.txt from uv.lock.
# torch/torchvision and their CUDA runtime packages are left out because Colab and
# Kaggle already ship a matching CUDA build of PyTorch.
set -euo pipefail
cd "$(dirname "$0")/.."
uv export --format requirements-txt --no-hashes --no-annotate --no-header --no-emit-project \
  | grep -Ev '^(torch|torchvision|triton|nvidia-|cuda-)' \
  > requirements-colab.txt
echo "Wrote requirements-colab.txt ($(grep -c . requirements-colab.txt) lines)"
