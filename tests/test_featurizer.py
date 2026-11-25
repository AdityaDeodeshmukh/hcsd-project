"""Featurizer contract. Does not download CLIP weights."""

import inspect
import sys
import unittest
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.featurizer.base import Featurizer, l2_normalize

try:
    from src.featurizer.clip import CLIP
except ModuleNotFoundError:
    CLIP = None


class _Fixed(Featurizer):
    def __init__(self, image_features: torch.Tensor, text_features: torch.Tensor) -> None:
        self._image_features = image_features
        self._text_features = text_features

    def encode_image(self, images):
        if len(images) != self._image_features.shape[0]:
            raise AssertionError("unexpected image count")
        return self._image_features

    def encode_text(self, texts):
        if len(texts) != self._text_features.shape[0]:
            raise AssertionError("unexpected text count")
        return self._text_features


class FeaturizerTests(unittest.TestCase):
    def test_cosine_normalizes_raw_vectors(self) -> None:
        images = [Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))]
        featurizer = _Fixed(
            torch.tensor([[3.0, 0.0], [0.0, 4.0]]),
            torch.tensor([[0.0, 5.0], [2.0, 0.0]]),
        )
        similarity = featurizer.cosine_similarity(images, ["up", "right"])
        expected = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        self.assertTrue(torch.allclose(similarity, expected, atol=1e-6))

    def test_cosine_rejects_empty_input(self) -> None:
        featurizer = _Fixed(torch.zeros(1, 2), torch.zeros(1, 2))
        with self.assertRaises(ValueError):
            featurizer.cosine_similarity([], ["a person"])
        with self.assertRaises(ValueError):
            featurizer.cosine_similarity([Image.new("RGB", (2, 2))], [])

    def test_l2_normalize_unit_rows(self) -> None:
        normalized = l2_normalize(torch.tensor([[3.0, 4.0], [0.0, -2.0]]))
        norms = normalized.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones(2)))

    @unittest.skipUnless(CLIP is not None, "open_clip is not installed")
    def test_clip_subclass_keeps_openai_quickgelu_default(self) -> None:
        self.assertTrue(issubclass(CLIP, Featurizer))
        signature = inspect.signature(CLIP.__init__)
        self.assertEqual(signature.parameters["model_name"].default, "ViT-B-32-quickgelu")
        self.assertEqual(signature.parameters["pretrained"].default, "openai")


if __name__ == "__main__":
    unittest.main()
