"""Image-text featurizers.

CLIP is loaded lazily so importers of the base class do not require open_clip.
"""

from src.featurizer.base import Featurizer

__all__ = ["CLIP", "Featurizer"]


def __getattr__(name: str):
    if name == "CLIP":
        from src.featurizer.clip import CLIP

        return CLIP
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
