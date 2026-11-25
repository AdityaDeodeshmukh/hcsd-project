"""Shared interface for image-text featurizers.

A featurizer embeds images and text into one vector space. Cosine similarity
is the comparison. CLIP is the first implementation. A later model such as
SigLIP can subclass the same base without changing the callers.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from PIL import Image


def l2_normalize(features: torch.Tensor) -> torch.Tensor:
    """Return row-wise L2-normalized features."""
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class Featurizer(ABC):
    """Image and text embeddings compared by cosine similarity."""

    @abstractmethod
    def encode_image(self, images: Sequence[Image.Image]) -> torch.Tensor:
        """Embed images. Returns shape ``[n_images, dim]``, L2-normalized."""

    @abstractmethod
    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        """Embed texts. Returns shape ``[n_texts, dim]``, L2-normalized."""

    def cosine_similarity(
        self,
        images: Sequence[Image.Image],
        texts: Sequence[str],
    ) -> torch.Tensor:
        """Cosine similarity, shape ``[n_images, n_texts]``.

        Rows follow ``images`` and columns follow ``texts``. Embeddings are
        normalized here as well, so a subclass that returns raw vectors still
        produces a cosine.
        """
        if len(images) == 0:
            raise ValueError("cosine_similarity requires at least one image")
        if len(texts) == 0:
            raise ValueError("cosine_similarity requires at least one text")
        image_features = l2_normalize(self.encode_image(images))
        text_features = l2_normalize(self.encode_text(texts))
        return image_features @ text_features.T
