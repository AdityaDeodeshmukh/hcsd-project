import importlib

# type name in the config -> "module:Class". Imported lazily, so torch loads only when needed.
REGISTRY = {
    "clip": "hcsd_project.featurizers.clip:ClipFeaturizer",
}


def load_featurizer(cfg, name):
    """Load a featurizer by name.
    
    Args:
        cfg: Full configuration dict
        name: Featurizer name (e.g., 'clip')
    """
    # Validate against registry
    if name not in REGISTRY:
        raise KeyError(f"Featurizer '{name}' is not registered. Known: {list(REGISTRY.keys())}")
    
    module, cls = REGISTRY[name].split(":")
    
    # Get featurizer config from cfg
    if name not in cfg["featurizers"]:
        raise KeyError(f"Featurizer '{name}' not found in cfg['featurizers']")
    
    return getattr(importlib.import_module(module), cls).from_config(cfg, name)
