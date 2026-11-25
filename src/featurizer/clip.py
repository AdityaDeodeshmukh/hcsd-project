"""CLIP featurizer. OpenAI weights, loaded through open_clip."""

from collections.abc import Sequence

import open_clip
import torch
from PIL import Image

from src.featurizer.base import Featurizer, l2_normalize


class CLIP(Featurizer):
    """OpenCLIP wrapper for the OpenAI CLIP weights.

    The default architecture is ``ViT-B-32-quickgelu`` because those weights
    were trained with QuickGELU.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32-quickgelu",
        pretrained: str = "openai",
        device: torch.device | str | None = None,
        batch_size: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.model_name = model_name
        self.pretrained = pretrained
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def encode_image(self, images: Sequence[Image.Image]) -> torch.Tensor:
        if len(images) == 0:
            raise ValueError("encode_image requires at least one image")
        features = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            pixels = torch.stack([self.preprocess(image.convert("RGB")) for image in batch])
            pixels = pixels.to(self.device)
            with torch.no_grad():
                features.append(self.model.encode_image(pixels))
        return l2_normalize(torch.cat(features, dim=0))

    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        if len(texts) == 0:
            raise ValueError("encode_text requires at least one text")
        features = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            tokens = self.tokenizer(batch).to(self.device)
            with torch.no_grad():
                features.append(self.model.encode_text(tokens))
        return l2_normalize(torch.cat(features, dim=0))
