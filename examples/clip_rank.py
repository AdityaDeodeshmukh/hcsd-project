"""Score and rank images against text prompts with CLIP.

Usage (from the repo root):
    python examples/clip_rank.py --images outputs/*.png --prompts "a smiling person" "a person with glasses"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import PIL.Image

from src.featurizer.clip import CLIP


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--prompts", nargs="+", required=True)
    # The OpenAI weights were trained with QuickGELU, hence the "-quickgelu" config.
    parser.add_argument("--model", default="ViT-B-32-quickgelu")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    featurizer = CLIP(model_name=args.model, pretrained=args.pretrained, batch_size=args.batch_size)
    images = []
    for path in args.images:
        with PIL.Image.open(path) as handle:
            images.append(handle.convert("RGB"))
    similarity = featurizer.cosine_similarity(images, args.prompts)

    for j, prompt in enumerate(args.prompts):
        print(f'\nPrompt: "{prompt}"')
        order = similarity[:, j].argsort(descending=True).tolist()
        for rank, i in enumerate(order, 1):
            print(f"  {rank}. {args.images[i]}  (cos={similarity[i, j].item():.4f})")


if __name__ == "__main__":
    main()
