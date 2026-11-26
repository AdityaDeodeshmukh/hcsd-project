from PIL import Image
import numpy as np
import open_clip
import torch

from PIL import Image
import numpy as np
import open_clip
import torch


class ClipFeaturizer:
    @classmethod
    def from_config(cls, cfg, name):
        m = cfg["featurizers"][name]
        return cls(
            model_name=m.get("model", "ViT-B-32-quickgelu"),
            pretrained=m.get("pretrained", "openai"),
            device=cfg["run"].get("device"),
            attribute_prompt_list=None,  # populated from prompts.yaml in CLI
        )

    def __init__(self,
                 model_name: str = "ViT-B-32-quickgelu",
                 pretrained: str = "openai",
                 device: str | None = None,
                 attribute_prompt_list: list[list[str]] | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.attribute_prompt_list = attribute_prompt_list or [
            ["photo of a smiling face", "photo of a neutral face"]
        ]
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self._text_cache: dict[str, torch.Tensor] = {}   # prompt string -> normalized [D] feature

    # ------------------------------------------------------------------ encoding

    @torch.no_grad()
    def encode_images(self, images: list[np.ndarray]) -> torch.Tensor:
        """Encode a batch of uint8 HxWx3 images -> [N, D], L2-normalized."""
        batch = torch.stack([
            self.preprocess(Image.fromarray(img).convert("RGB")) for img in images
        ]).to(self.device)
        feat = self.model.encode_image(batch)
        return feat / feat.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode texts -> [T, D], L2-normalized. Each distinct string is encoded once and cached."""
        missing = [t for t in dict.fromkeys(texts) if t not in self._text_cache]
        if missing:
            tokens = self.tokenizer(missing).to(self.device)
            feat = self.model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            for t, f in zip(missing, feat):
                self._text_cache[t] = f
        return torch.stack([self._text_cache[t] for t in texts])

    def clear_text_cache(self) -> None:
        self._text_cache.clear()

    def encode_image(self, image: np.ndarray) -> torch.Tensor:
        return self.encode_images([image])[0]

    def encode_text(self, text: str) -> torch.Tensor:
        return self.encode_texts([text])[0]

    # ---------------------------------------------------------------- similarity

    def compute_similarity(self, image: np.ndarray, text: str) -> float:
        return float(self.encode_image(image) @ self.encode_text(text))

    def compute_similarities(self, images: list[np.ndarray], texts: list[str]) -> np.ndarray:
        """Cosine similarity matrix, shape [n_images, n_texts]."""
        img_feats = self.encode_images(images)
        txt_feats = self.encode_texts(texts)
        return (img_feats @ txt_feats.T).cpu().numpy()

    def get_top_matches(self, image: np.ndarray, texts: list[str], top_k: int = 5):
        sims = self.compute_similarities([image], texts)[0]
        order = np.argsort(-sims)[:top_k]
        return [(texts[i], float(sims[i])) for i in order]

    def _pair_features(self, pairs: list[list[str]]) -> torch.Tensor:
        """[P, 2, D] text features for (target, neutral) pairs, served from the cache."""
        flat = [t for p in pairs for t in p[:2]]
        return self.encode_texts(flat).float().view(len(pairs), 2, -1)

    @torch.no_grad()
    def compute_image_attribute_similarity(
        self,
        image_embedding: torch.Tensor | np.ndarray,
        attribute_prompt_list: list[list[str]] | None = None,
    ) -> dict:
        """
        For each [target, neutral] pair, return P(target) via a CLIP-scaled softmax.

        image_embedding: L2-normalized, shape [D] or [N, D].
        Returns {target_text: float} for a [D] input,
                {target_text: list[float]} for an [N, D] input.
        """
        pairs = attribute_prompt_list or self.attribute_prompt_list
        pairs = [p for p in pairs if len(p) >= 2]
        if not pairs:
            return {}

        emb = torch.as_tensor(image_embedding, dtype=torch.float32, device=self.device)
        single = emb.ndim == 1
        if single:
            emb = emb.unsqueeze(0)                                   # [1, D]

        txt_feats = self._pair_features(pairs)                       # [P, 2, D], cached
        scale = self.model.logit_scale.exp()
        logits = scale * torch.einsum("nd,pkd->npk", emb, txt_feats)  # [N, P, 2]
        probs = logits.softmax(dim=-1)[..., 0]                       # [N, P]

        if single:
            return {p[0]: float(probs[0, j]) for j, p in enumerate(pairs)}
        return {p[0]: probs[:, j].cpu().tolist() for j, p in enumerate(pairs)}
