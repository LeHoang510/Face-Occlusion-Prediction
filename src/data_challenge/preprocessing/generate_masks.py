"""Generate pseudo-masks for the face occlusion dataset.

Supported backends:
- SAM: prompt-based masks using a central box around the crop.
- DeepLabV3: semantic masks using the person class by default.

The resulting masks are not ground truth; they are auxiliary features for the
occlusion regressor. Existing valid masks are skipped by default, so interrupted
runs can be resumed.
"""

import argparse
import os
import urllib.request
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


DEEPLAB_PERSON_CLASS_INDEX = 15

SAM_CHECKPOINT_URLS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}

SAM_CHECKPOINT_FILENAMES = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_sam_checkpoint(model_type: str, checkpoint: str | None, cache_dir: str) -> str:
    if checkpoint:
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"Missing SAM checkpoint: {checkpoint}")
        return checkpoint

    if model_type not in SAM_CHECKPOINT_URLS:
        choices = ", ".join(sorted(SAM_CHECKPOINT_URLS))
        raise ValueError(f"No automatic checkpoint URL for '{model_type}'. Choose from: {choices}")

    cache_path = Path(cache_dir, SAM_CHECKPOINT_FILENAMES[model_type])
    if cache_path.is_file():
        return str(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SAM checkpoint to: {cache_path}")
    print(f"Source: {SAM_CHECKPOINT_URLS[model_type]}")
    urllib.request.urlretrieve(SAM_CHECKPOINT_URLS[model_type], cache_path)
    return str(cache_path)


def build_sam_predictor(model_type: str, checkpoint: str | None, cache_dir: str, device: torch.device):
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise ImportError(
            "SAM support requires the official segment-anything package. "
            "Install it with: pip install git+https://github.com/facebookresearch/segment-anything.git"
        ) from exc

    if model_type not in sam_model_registry:
        choices = ", ".join(sorted(sam_model_registry))
        raise ValueError(f"Unknown SAM model type '{model_type}'. Choose from: {choices}")

    checkpoint = resolve_sam_checkpoint(model_type, checkpoint, cache_dir)

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    sam.eval()
    return SamPredictor(sam)


def build_deeplab(device: torch.device, weights_path: str | None = None):
    from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    if weights_path:
        model = deeplabv3_resnet50(weights=None, weights_backbone=None)
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state)
    else:
        model = deeplabv3_resnet50(weights=weights)

    model = model.to(device)
    model.eval()
    return model, weights.transforms()


def mask_output_path(mask_root: str, filename: str) -> Path:
    return Path(mask_root, filename).with_suffix(".png")


def mask_is_done(mask_path: Path) -> bool:
    if not mask_path.is_file():
        return False
    try:
        with Image.open(mask_path) as image:
            image.verify()
        return True
    except Exception:
        return False


def save_mask_atomic(mask: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=out_path.parent, suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        Image.fromarray(mask, mode="L").save(tmp_path)
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def central_box(width: int, height: int, margin: float) -> np.ndarray:
    margin = min(max(margin, 0.0), 0.49)
    x0 = int(width * margin)
    y0 = int(height * margin)
    x1 = int(width * (1.0 - margin))
    y1 = int(height * (1.0 - margin))
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def collect_pending_masks(filenames: list[str], mask_root: str, skip_existing: bool) -> tuple[list[str], int]:
    pending = []
    skipped = 0
    for filename in filenames:
        out_path = mask_output_path(mask_root, filename)
        if skip_existing and mask_is_done(out_path):
            skipped += 1
            continue
        pending.append(filename)
    return pending, skipped


def print_generation_summary(
    device: torch.device,
    backend: str,
    mask_root: str,
    pending_count: int,
    total_count: int,
    skipped_count: int,
):
    print(f"Device: {device}")
    print(f"Backend: {backend}")
    print(f"Mask root: {mask_root}")
    print(f"Already done: {skipped_count}")
    print(f"Images to process: {pending_count} / {total_count}")


def generate_sam_masks(
    csv_paths: list[str],
    img_root: str,
    mask_root: str,
    sam_checkpoint: str | None,
    sam_model_type: str,
    sam_cache_dir: str,
    box_margin: float,
    skip_existing: bool,
    multimask_output: bool,
):
    device = get_device()
    predictor = build_sam_predictor(
        model_type=sam_model_type,
        checkpoint=sam_checkpoint,
        cache_dir=sam_cache_dir,
        device=device,
    )
    filenames = collect_filenames(csv_paths)

    Path(mask_root).mkdir(parents=True, exist_ok=True)
    pending, skipped = collect_pending_masks(filenames, mask_root, skip_existing)

    print_generation_summary(
        device=device,
        backend="sam",
        mask_root=mask_root,
        pending_count=len(pending),
        total_count=len(filenames),
        skipped_count=skipped,
    )
    print(f"SAM model: {sam_model_type}")
    print(f"SAM checkpoint: {sam_checkpoint or 'auto-download'}")
    print(f"Box margin: {box_margin}")
    if not pending:
        return

    with torch.no_grad():
        for filename in tqdm(pending, desc="Generating SAM masks", unit="image"):
            image = Image.open(os.path.join(img_root, filename)).convert("RGB")
            image_array = np.asarray(image)
            height, width = image_array.shape[:2]
            predictor.set_image(image_array)

            masks, scores, _logits = predictor.predict(
                box=central_box(width=width, height=height, margin=box_margin),
                multimask_output=multimask_output,
            )
            best_idx = int(np.argmax(scores))
            mask = masks[best_idx].astype(np.uint8) * 255

            out_path = mask_output_path(mask_root, filename)
            save_mask_atomic(mask, out_path)


def generate_deeplab_masks(
    csv_paths: list[str],
    img_root: str,
    mask_root: str,
    batch_size: int,
    threshold: float,
    class_index: int,
    skip_existing: bool,
    deeplab_weights_path: str | None = None,
):
    device = get_device()
    model, preprocess = build_deeplab(device, weights_path=deeplab_weights_path)
    filenames = collect_filenames(csv_paths)

    Path(mask_root).mkdir(parents=True, exist_ok=True)
    pending, skipped = collect_pending_masks(filenames, mask_root, skip_existing)

    print_generation_summary(
        device=device,
        backend="deeplabv3",
        mask_root=mask_root,
        pending_count=len(pending),
        total_count=len(filenames),
        skipped_count=skipped,
    )
    print(f"DeepLab class index: {class_index}")
    print(f"Threshold: {threshold}")
    if deeplab_weights_path:
        print(f"DeepLab weights: {deeplab_weights_path}")
    if not pending:
        return

    with torch.no_grad():
        for start in tqdm(range(0, len(pending), batch_size), desc="Generating DeepLab masks", unit="batch"):
            batch_filenames = pending[start:start + batch_size]
            images = [Image.open(os.path.join(img_root, filename)).convert("RGB") for filename in batch_filenames]
            original_sizes = [image.size[::-1] for image in images]
            inputs = torch.stack([preprocess(image) for image in images]).to(device)

            logits = model(inputs)["out"]
            probs = torch.softmax(logits, dim=1)[:, class_index:class_index + 1]

            for filename, prob, original_size in zip(batch_filenames, probs, original_sizes, strict=True):
                prob = F.interpolate(
                    prob.unsqueeze(0),
                    size=original_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
                mask = (prob >= threshold).to(torch.uint8).cpu().numpy() * 255
                save_mask_atomic(mask, mask_output_path(mask_root, filename))


def collect_filenames(csv_paths: list[str]) -> list[str]:
    filenames = []
    seen = set()
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        for filename in df["filename"].tolist():
            if filename not in seen:
                filenames.append(filename)
                seen.add(filename)
    return filenames


def parse_args():
    parser = argparse.ArgumentParser(description="Generate pseudo segmentation masks")
    parser.add_argument("--backend", default="sam", choices=["sam", "deeplabv3"])
    parser.add_argument("--train-csv", required=True, help="Path to train.csv")
    parser.add_argument("--test-csv", default=None, help="Optional path to test_students.csv")
    parser.add_argument("--img-root", required=True, help="Root folder containing crop images")
    parser.add_argument("--mask-root", required=True, help="Output root for generated .png masks")

    parser.add_argument("--sam-checkpoint", default=None, help="Optional path to a SAM checkpoint, e.g. sam_vit_b_01ec64.pth")
    parser.add_argument("--sam-model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--sam-cache-dir", default="checkpoints/sam", help="Folder used for automatically downloaded SAM checkpoints")
    parser.add_argument("--box-margin", type=float, default=0.03, help="Fraction trimmed from each image edge for the box prompt")
    parser.add_argument("--multimask-output", action="store_true", help="Ask SAM for multiple masks and save the highest-scoring one")

    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for DeepLabV3 mask generation")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for DeepLabV3 masks")
    parser.add_argument("--class-index", type=int, default=DEEPLAB_PERSON_CLASS_INDEX, help="Semantic class index for DeepLabV3")
    parser.add_argument("--deeplab-weights-path", default=None, help="Optional local DeepLabV3 state dict path")

    parser.add_argument("--overwrite", action="store_true", help="Regenerate masks that already exist; default resumes by skipping valid masks")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_paths = [args.train_csv]
    if args.test_csv:
        csv_paths.append(args.test_csv)

    if args.backend == "sam":
        generate_sam_masks(
            csv_paths=csv_paths,
            img_root=args.img_root,
            mask_root=args.mask_root,
            sam_checkpoint=args.sam_checkpoint,
            sam_model_type=args.sam_model_type,
            sam_cache_dir=args.sam_cache_dir,
            box_margin=args.box_margin,
            skip_existing=not args.overwrite,
            multimask_output=args.multimask_output,
        )
    elif args.backend == "deeplabv3":
        generate_deeplab_masks(
            csv_paths=csv_paths,
            img_root=args.img_root,
            mask_root=args.mask_root,
            batch_size=args.batch_size,
            threshold=args.threshold,
            class_index=args.class_index,
            skip_existing=not args.overwrite,
            deeplab_weights_path=args.deeplab_weights_path,
        )
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")


if __name__ == "__main__":
    main()
