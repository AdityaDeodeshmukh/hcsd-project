import hashlib
from pathlib import Path

import requests
from tqdm import tqdm

from hcsd_project.config import load_config, resolve


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def ensure_weights(cfg, name):
    """Return the local path of weights `name`, downloading first if it's missing."""
    if name not in cfg["weights"]:
        raise KeyError(f"Unknown weights '{name}'. Known: {list(cfg['weights'])}")
    spec = cfg["weights"][name]
    dest = resolve(cfg["paths"]["models_dir"]) / spec["filename"]
    expected = spec.get("sha256")

    if dest.exists():
        if expected and _sha256(dest) != expected:
            raise RuntimeError(f"{dest} exists but its checksum does not match. Delete it and retry.")
        return dest

    if not spec.get("url"):
        raise FileNotFoundError(f"{dest} is missing and no url is configured. Place the file there manually.")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {name} -> {dest}")
    with requests.get(spec["url"], stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

    if expected and _sha256(tmp) != expected:
        tmp.unlink()
        raise RuntimeError(f"Checksum mismatch for {name}; the download was removed.")
    tmp.rename(dest)
    return dest


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--set", action="append", default=[], dest="overrides")
    p.add_argument("names", nargs="*", help="weights to fetch (default: all in the config)")
    a = p.parse_args()
    cfg = load_config(a.config, a.overrides)
    for n in a.names or cfg["weights"]:
        path = ensure_weights(cfg, n)
        print(f"{n}: {path}  sha256={_sha256(path)}")
