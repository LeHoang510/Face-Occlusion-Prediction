"""Create a quick image/mask overlay grid for quality checking generated masks."""

import argparse
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from data_challenge.preprocessing.generate_masks import mask_output_path


def visualize_masks(
    csv_path: str,
    img_root: str,
    mask_root: str,
    output_path: str,
    num_samples: int,
    seed: int,
    alpha: float,
):
    rng = random.Random(seed)
    df = pd.read_csv(csv_path)
    filenames = df["filename"].tolist()
    rng.shuffle(filenames)

    pairs = []
    for filename in filenames:
        image_path = os.path.join(img_root, filename)
        mask_path = mask_output_path(mask_root, filename)
        if os.path.isfile(image_path) and mask_path.is_file():
            pairs.append((filename, image_path, mask_path))
        if len(pairs) >= num_samples:
            break

    if not pairs:
        raise FileNotFoundError("No image/mask pairs found. Generate masks first or check mask_root.")

    cols = min(4, len(pairs))
    rows = (len(pairs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, (filename, image_path, mask_path) in zip(axes, pairs, strict=False):
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L").resize(image.size)
        ax.imshow(image)
        ax.imshow(mask, cmap="Reds", alpha=alpha)
        ax.set_title(filename, fontsize=8)
        ax.axis("off")

    for ax in axes[len(pairs):]:
        ax.axis("off")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved mask visualization to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize generated segmentation masks")
    parser.add_argument("--csv", default="dataset/DataChallenge2026/occlusion_datasets/train.csv")
    parser.add_argument("--img-root", default="dataset/crops/Crop_224_5fp_100K")
    parser.add_argument("--mask-root", default="dataset/masks/sam_vit_b")
    parser.add_argument("--output", default="outputs/mask_preview.png")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.45)
    return parser.parse_args()


def main():
    args = parse_args()
    visualize_masks(
        csv_path=args.csv,
        img_root=args.img_root,
        mask_root=args.mask_root,
        output_path=args.output,
        num_samples=args.num_samples,
        seed=args.seed,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
