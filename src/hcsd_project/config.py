import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default_config.yaml"


def resolve(p):
    """Relative paths are resolved from the repo root."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def _set(cfg, dotted_key, value):
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        cfg = cfg.setdefault(k, {})
    cfg[keys[-1]] = value


def load_config(path=None, overrides=None):
    """Load YAML. `overrides` is a list like ["sampling.n=16", "paths.models_dir=/content/drive/models"]."""
    path = Path(path or os.environ.get("HCSD_CONFIG", DEFAULT_CONFIG))
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value, got '{item}'")
        _set(cfg, key, yaml.safe_load(value))   # parses 16 -> int, null -> None, 0.7 -> float
    cfg["_config_path"] = str(path)
    return cfg
