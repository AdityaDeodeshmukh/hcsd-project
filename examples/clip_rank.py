"""Score and rank images against text prompts with CLIP (OpenAI weights via open_clip).

Usage (from the repo root):
    python examples/clip_rank.py --images outputs/*.png --prompts "a smiling person" "a person with glasses"
"""

import argparse

import open_clip
import PIL.Image
import torch


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--prompts", nargs="+", required=True)
    # The OpenAI weights were trained with QuickGELU, hence the "-quickgelu" config.
    parser.add_argument("--model", default="ViT-B-32-quickgelu")
    parser.add_argument("--pretrained", default="openai")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()

    images = torch.stack([preprocess(PIL.Image.open(p).convert("RGB")) for p in args.images]).to(device)
    text = tokenizer(args.prompts).to(device)

    img_feat = model.encode_image(images)
    txt_feat = model.encode_text(text)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
    sim = img_feat @ txt_feat.T  # cosine similarity, shape [n_images, n_prompts]

    for j, prompt in enumerate(args.prompts):
        print(f'\nPrompt: "{prompt}"')
        for rank, i in enumerate(sim[:, j].argsort(descending=True).tolist(), 1):
            print(f"  {rank}. {args.images[i]}  (cos={sim[i, j].item():.4f})")


if __name__ == "__main__":
    main()
