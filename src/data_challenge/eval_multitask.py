"""Evaluation and inference script for multitask occlusion models."""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from data_challenge.data.dataset import OcclusionDataset, get_transforms
from data_challenge.models.multitask import MultiTaskOcclusionModel
from data_challenge.train import get_device, load_config
from data_challenge.utils.logger import setup_logger
from data_challenge.utils.metrics import compute_score


def evaluate(config_path: str, checkpoint: str | None = None, predict_test: bool = False):
    cfg = load_config(config_path)
    logger = setup_logger("eval_multitask")
    device = get_device()
    logger.info("Using device: %s", device)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    model = MultiTaskOcclusionModel(
        backbone=model_cfg["backbone"],
        pretrained=False,
        freeze_backbone=model_cfg.get("freeze_backbone", False),
        regression_head=model_cfg.get("regression_head", {}),
        gender_head=model_cfg.get("gender_head", {}),
    ).to(device)

    ckpt_path = checkpoint or os.path.join(cfg["output"]["dir"], "best_model.pt")
    logger.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info("Checkpoint epoch=%d | saved_score=%.5f", ckpt.get("epoch", -1), ckpt.get("score", float("nan")))

    transform = get_transforms(train=False, img_size=data_cfg["img_size"])
    if predict_test:
        _predict_test(model, cfg, data_cfg, transform, device, logger)
        return

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
        generator=torch.Generator().manual_seed(cfg["training"]["seed"]),
    )

    loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    logger.info("Evaluating on %d validation samples...", val_size)

    all_preds, all_labels, all_genders, all_gender_preds = [], [], [], []
    with torch.no_grad():
        for images, labels, genders in loader:
            outputs = model(images.to(device))
            gender_probs = torch.sigmoid(outputs["gender_logits"]).cpu().numpy()
            all_preds.extend(outputs["occlusion"].cpu().numpy())
            all_labels.extend(labels.numpy())
            all_genders.extend(genders.numpy())
            all_gender_preds.extend((gender_probs >= 0.5).astype(np.float32))

    genders_np = np.array(all_genders)
    score, err_f, err_m = compute_score(np.array(all_preds), np.array(all_labels), genders_np)
    gender_acc = float((np.array(all_gender_preds) == genders_np).mean())
    logger.info(
        "Val score=%.5f (lower is better) | err_female=%.5f | err_male=%.5f | gender_acc=%.4f",
        score,
        err_f,
        err_m,
        gender_acc,
    )


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

    all_filenames, all_preds, all_gender_preds = [], [], []
    with torch.no_grad():
        for images, filenames in loader:
            outputs = model(images.to(device))
            gender_probs = torch.sigmoid(outputs["gender_logits"]).cpu().numpy()
            all_preds.extend(outputs["occlusion"].cpu().numpy())
            all_gender_preds.extend((gender_probs >= 0.5).astype(int))
            all_filenames.extend(filenames)

    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test_predictions.csv")
    pd.DataFrame({
        "filename": all_filenames,
        "FaceOcclusion": all_preds,
        "gender": all_gender_preds,
    }).to_csv(out_path, index=False)
    logger.info("Predictions saved to %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multitask face occlusion model")
    parser.add_argument("--config", default="src/data_challenge/configs/phase4_multitask_heads/swin_t.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint")
    parser.add_argument("--predict-test", action="store_true", help="Generate test set predictions instead of val eval")
    args = parser.parse_args()
    evaluate(args.config, checkpoint=args.checkpoint, predict_test=args.predict_test)
