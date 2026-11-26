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
    
    # Load CLIP prompts from the prompts file defined in default_config.yaml
    if "prompts_file" in cfg.get("run", {}) and not overrides:
        try:
            prompts_cfg = load_text_prompts(cfg["run"]["prompts_file"])
            # Merge prompts into config under featurizers.clip
            if "featurizers" not in cfg or "clip" not in cfg["featurizers"]:
                cfg["featurizers"]["clip"] = {}
            for key, value in prompts_cfg.get("clip", {}).items():
                _set(cfg["featurizers"], f"clip.{key}", value)
        except FileNotFoundError:
            pass  # Prompts file not found, use defaults
    
    for item in overrides or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value, got '{item}'")
        _set(cfg, key, yaml.safe_load(value))   # parses 16 -> int, null -> None, 0.7 -> float
    cfg["_config_path"] = str(path)
    return cfg


def load_text_prompts(prompts_file: str | None = None):
    """Load CLIP prompts from the prompts file.
    
    If no file is specified, defaults to configs/prompts.yaml.
    Returns a dict with 'clip' key containing attribute prompts.
    """
    if prompts_file is None:
        prompts_file = "configs/prompts.yaml"
    with open(prompts_file) as f:
        return yaml.safe_load(f)
