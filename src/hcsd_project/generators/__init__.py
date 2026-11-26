import importlib

# type name in the config -> "module:Class". Imported lazily, so torch loads only when needed.
REGISTRY = {
    "stylegan2_ada": "hcsd_project.generators.stylegan2_ada:StyleGAN2ADA",
}


def load_generator(cfg, name):
    if name not in cfg["models"]:
        raise KeyError(f"Model '{name}' is not in config. Known: {list(cfg['models'])}")
    t = cfg["models"][name]["type"]
    if t not in REGISTRY:
        raise KeyError(f"Unknown model type '{t}'. Known: {list(REGISTRY)}")
    module, cls = REGISTRY[t].split(":")
    return getattr(importlib.import_module(module), cls).from_config(cfg, name)
