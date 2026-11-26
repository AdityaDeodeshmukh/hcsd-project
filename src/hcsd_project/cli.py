import argparse
import gc
import json
import numpy as np
import yaml
import os
from PIL import Image

from hcsd_project.config import load_config, resolve
from hcsd_project.featurizers import load_featurizer
from hcsd_project.generators import load_generator

CHUNK = 8   # images rendered at once; keeps memory flat for large n_images


def run_model(cfg, name):
    r = cfg["run"]
    out = resolve(cfg["paths"]["outputs_dir"]) / name
    (out / "images").mkdir(parents=True, exist_ok=True)

    gen = load_generator(cfg, name)
    print(f"[{name}] device={gen.device} w_dim={gen.w_dim} num_ws={gen.num_ws}")

    # Load featurizer if configured (first one in cfg["featurizers"])
    featurizer = None
    featurizers_cfg = cfg.get("featurizers")
    if isinstance(featurizers_cfg, dict) and featurizers_cfg:
        featurizer = load_featurizer(cfg, next(iter(featurizers_cfg)))

    # Pull attribute prompts from prompts.yaml; keep the featurizer defaults if none are found
    if featurizer is not None and "prompts_file" in cfg.get("run", {}):
        try:
            from hcsd_project.config import load_text_prompts
            prompts_cfg = load_text_prompts(cfg["run"]["prompts_file"])
            prompts = prompts_cfg.get("clip", {}).get("attribute_prompt_list")
            if prompts:
                featurizer.attribute_prompt_list = prompts
        except FileNotFoundError:
            pass  # prompts file not found, use defaults

    w = gen.sample_w(r["n_latents"], r["seed"], r["psi"])
    np.save(out / "w.npy", w.cpu().numpy())

    n_lat = len(w)
    n_save = min(r["n_images"], n_lat)   # PNGs/thumbnails only for the first n_save
    thumbs = []
    embedding_chunks = []

    for start in range(0, n_lat, CHUNK):
        print(f"Starting {start} chunk")
        imgs = gen.synthesize(w[start:start + CHUNK]).numpy()

        for i, im in enumerate(imgs, start):
            if i < n_save:
                pil = Image.fromarray(im)
                pil.save(out / "images" / f"{i:06d}.png")
                thumbs.append(np.asarray(pil.resize((256, 256))))

        # One batched CLIP forward pass per chunk, for every latent
        if featurizer is not None:
            embedding_chunks.append(featurizer.encode_images(list(imgs)).cpu().numpy())

    if r.get("save_grid", True) and thumbs:
        cols = min(8, len(thumbs))
        rows = [thumbs[i:i + cols] for i in range(0, len(thumbs), cols)]
        rows = [np.concatenate(row + [np.zeros_like(thumbs[0])] * (cols - len(row)), axis=1) for row in rows]
        Image.fromarray(np.concatenate(rows, axis=0)).save(out / "grid.png")

    # Save exactly what produced these outputs, for reproducibility.
    with open(out / "config_used.yaml", "w") as f:
        yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, f, sort_keys=False)

    if embedding_chunks:
        all_embeddings = np.vstack(embedding_chunks)            # [n_lat, D], row k <-> w[k]
        np.save(out / "embeddings.npy", all_embeddings)

        # Attribute scores from the cached embeddings (text features are cached in the featurizer)
        scores = featurizer.compute_image_attribute_similarity(all_embeddings)   # {target: [n_lat floats]}
        print(scores)
        with open(out / "attribute_scores.json", "w") as f:
            json.dump({
                "prompts": featurizer.attribute_prompt_list,
                "scores": scores,
            }, f, indent=2)

        print(f"[{name}] saved embeddings {all_embeddings.shape} and {len(scores)} attribute score(s)")

    print(f"[{name}] saved {n_lat} latents and {n_save} images to {out}")

    del gen, featurizer   # free GPU memory before the next model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

def main(argv=None):
    p = argparse.ArgumentParser(prog="hcsd-project", description=__doc__)
    p.add_argument("--config", help="path to a YAML config (default: configs/config.yaml)")
    p.add_argument("--set", action="append", default=[], dest="overrides",
                   help="override a value, e.g. --set run.n_images=4")
    p.add_argument("--models", nargs="+", help="shortcut for --set run.models=[...]")
    p.add_argument("--list", action="store_true", help="list configured models and exit")
    p.add_argument("--download-only", action="store_true",
                   help="fetch the weights of the selected models, print checksums, and exit")
    args = p.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    if args.models:
        cfg["run"]["models"] = args.models

    if args.list:
        for name, m in cfg["models"].items():
            print(f"{name}\t{m['type']}\t{m['weights']}")
        return

    names = cfg["run"]["models"]
    print(f"config: {cfg['_config_path']}  models: {names}")

    if args.download_only:
        from hcsd_project.download_models import _sha256, ensure_weights
        for name in names:
            w = cfg["models"][name]["weights"]
            path = ensure_weights(cfg, w)
            print(f"{w}: {path}  sha256={_sha256(path)}")
        return

    for name in names:
        run_model(cfg, name)


if __name__ == "__main__":
    main()
