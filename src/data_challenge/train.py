"""Training script for face occlusion prediction."""

import argparse
import json
import os
import random
import shutil
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data_challenge.data.dataset import OcclusionDataset, get_transforms
from data_challenge.models.cnn_baseline import CNNBaseline
from data_challenge.utils.logger import setup_logger
from data_challenge.utils.metrics import compute_score


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def warmup_lr_lambda(epoch: int, warmup_epochs: int):
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    return 1.0


def evaluate(model, loader, device) -> tuple[float, float, float]:
    """Run model on loader and return (score, err_female, err_male)."""
    model.eval()
    all_preds, all_labels, all_genders = [], [], []

    with torch.no_grad():
        for images, labels, genders in tqdm(loader, desc="  Validating", leave=False, unit="batch"):
            images = images.to(device)
            preds = model(images).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_genders.extend(genders.numpy())

    return compute_score(
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_genders),
    )


def train(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["training"]["seed"])

    logger = setup_logger("train")

    # WandB
    wandb_run = None
    if cfg["wandb"]["enabled"]:
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg["wandb"]["project"],
                entity=cfg["wandb"].get("entity") or None,
                name=cfg["run_name"],
                config=cfg,
            )
            logger.info("WandB run initialized: %s", wandb_run.url)
        except Exception as e:
            logger.warning("WandB init failed, continuing without it: %s", e)

    device = get_device()
    logger.info("Using device: %s", device)

    # Data
    data_cfg = cfg["data"]
    full_dataset = OcclusionDataset(
        csv_path=data_cfg["train_csv"],
        img_root=data_cfg["img_root"],
        transform=get_transforms(train=True, img_size=data_cfg["img_size"]),
    )

    val_size = int(len(full_dataset) * data_cfg["val_split"])
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg["training"]["seed"]),
    )
    # Val uses test-time transforms
    val_ds.dataset = OcclusionDataset(
        csv_path=data_cfg["train_csv"],
        img_root=data_cfg["img_root"],
        transform=get_transforms(train=False, img_size=data_cfg["img_size"]),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    logger.info("Train: %d samples | Val: %d samples", train_size, val_size)

    # Model
    model_cfg = cfg["model"]
    model = CNNBaseline(
        backbone=model_cfg["backbone"],
        pretrained=model_cfg["pretrained"],
        dropout=model_cfg["dropout"],
    ).to(device)
    logger.info("Model: %s (pretrained=%s)", model_cfg["backbone"], model_cfg["pretrained"])

    # Optimizer & scheduler
    train_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    if train_cfg["scheduler"] == "cosine":
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg["epochs"] - train_cfg["warmup_epochs"]
        )
    else:
        base_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda e: warmup_lr_lambda(e, train_cfg["warmup_epochs"]),
    )

    criterion = nn.MSELoss()

    # Output dir: outputs/<run_name>_<timestamp>/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg["output"]["dir"], f"{cfg['run_name']}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(run_dir, "config.yaml"))
    logger.info("Run output dir: %s", run_dir)

    best_score = float("inf")
    best_ckpt = os.path.join(run_dir, "best_model.pt")
    last_ckpt = os.path.join(run_dir, "last_model.pt")

    epoch_bar = tqdm(range(1, train_cfg["epochs"] + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0

        batch_bar = tqdm(train_loader, desc=f"  Epoch {epoch:02d}", leave=False, unit="batch")
        for images, labels, _genders in batch_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            if train_cfg.get("gradient_clip"):
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip"])
            optimizer.step()
            running_loss += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.5f}")

        train_loss = running_loss / len(train_loader)

        if epoch <= train_cfg["warmup_epochs"]:
            warmup_scheduler.step()
        else:
            base_scheduler.step()

        score, err_f, err_m = evaluate(model, val_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_bar.set_postfix(loss=f"{train_loss:.5f}", score=f"{score:.5f}", lr=f"{current_lr:.1e}")
        logger.info(
            "Epoch %02d/%02d | loss=%.5f | val_score=%.5f | err_F=%.5f | err_M=%.5f | lr=%.2e",
            epoch, train_cfg["epochs"], train_loss, score, err_f, err_m, current_lr,
        )

        if wandb_run:
            wandb_run.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/score": score,
                "val/err_female": err_f,
                "val/err_male": err_m,
                "val/err_diff": err_f - err_m,  # positive = female worse, negative = male worse
                "val/err_diff_abs": abs(err_f - err_m),
                "lr": current_lr,
            })

        ckpt = {"epoch": epoch, "model_state": model.state_dict(), "score": score}
        torch.save(ckpt, last_ckpt)

        if score < best_score:
            best_score = score
            torch.save(ckpt, best_ckpt)
            logger.info("  -> New best score %.5f (lower is better), checkpoint saved.", best_score)

    summary = {
        "run_name": cfg["run_name"],
        "backbone": cfg["model"]["backbone"],
        "best_val_score": best_score,
        "epochs": train_cfg["epochs"],
        "wandb_run_id": wandb_run.id if wandb_run else None,
        "wandb_url": wandb_run.url if wandb_run else None,
    }
    with open(os.path.join(run_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Training complete. Best val score: %.5f", best_score)
    logger.info("Artifacts saved to: %s", run_dir)
    if wandb_run:
        wandb_run.summary["best_val_score"] = best_score
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train face occlusion model")
    parser.add_argument("--config", default="src/data_challenge/configs/base_config.yaml")
    args = parser.parse_args()
    train(args.config)
