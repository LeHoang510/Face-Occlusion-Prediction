"""Evaluation and inference script."""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from data_challenge.data.dataset import OcclusionDataset, get_transforms
from data_challenge.models.cnn_baseline import CNNBaseline
from data_challenge.utils.logger import setup_logger
from data_challenge.utils.metrics import compute_score


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(config_path: str, checkpoint: str | None = None, predict_test: bool = False):
    cfg = load_config(config_path)
    logger = setup_logger("eval")

    device = get_device()
    logger.info("Using device: %s", device)

    # Model
    model_cfg = cfg["model"]
    model = CNNBaseline(
        backbone=model_cfg["backbone"],
        pretrained=False,
        dropout=0.0,
        img_size=cfg["data"]["img_size"],
    ).to(device)

    ckpt_path = checkpoint or os.path.join(cfg["output"]["dir"], "best_model.pt")
    logger.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info("Checkpoint epoch=%d | saved_score=%.5f", ckpt.get("epoch", -1), ckpt.get("score", float("nan")))

    data_cfg = cfg["data"]
    transform = get_transforms(train=False, img_size=data_cfg["img_size"])

    if predict_test:
        _predict_test(model, cfg, data_cfg, transform, device, logger)
        return

    # Evaluate on validation split
    from torch.utils.data import random_split
    import torch as _torch

    full_ds = OcclusionDataset(
        csv_path=data_cfg["train_csv"],
        img_root=data_cfg["img_root"],
        transform=transform,
    )
    val_size = int(len(full_ds) * data_cfg["val_split"])
    train_size = len(full_ds) - val_size
    _, val_ds = random_split(
        full_ds,
        [train_size, val_size],
        generator=_torch.Generator().manual_seed(cfg["training"]["seed"]),
    )

    loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    logger.info("Evaluating on %d validation samples...", val_size)

    all_preds, all_labels, all_genders = [], [], []
    with torch.no_grad():
        for images, labels, genders in loader:
            preds = model(images.to(device)).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_genders.extend(genders.numpy())

    score, err_f, err_m = compute_score(
        np.array(all_preds), np.array(all_labels), np.array(all_genders)
    )
    logger.info("Val score=%.5f (lower is better) | err_female=%.5f | err_male=%.5f", score, err_f, err_m)


def _predict_test(model, cfg, data_cfg, transform, device, logger):
    test_ds = OcclusionDataset(
        csv_path=data_cfg["test_csv"],
        img_root=data_cfg["img_root"],
        transform=transform,
        is_test=True,
    )
    loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    logger.info("Generating predictions for %d test samples...", len(test_ds))

    all_filenames, all_preds = [], []
    with torch.no_grad():
        for images, filenames in loader:
            preds = model(images.to(device)).cpu().numpy()
            all_preds.extend(preds)
            all_filenames.extend(filenames)

    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test_predictions.csv")
    pd.DataFrame({"filename": all_filenames, "FaceOcclusion": all_preds, "gender": "x"}).to_csv(out_path, index=False)
    logger.info("Predictions saved to %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate face occlusion model")
    parser.add_argument("--config", default="src/data_challenge/configs/base_config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint (default: output/best_model.pt)")
    parser.add_argument("--predict-test", action="store_true", help="Generate test set predictions instead of val eval")
    args = parser.parse_args()
    evaluate(args.config, checkpoint=args.checkpoint, predict_test=args.predict_test)
