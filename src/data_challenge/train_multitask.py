"""Training script for multitask occlusion regression and gender classification."""

import argparse
import json
import os
import shutil
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data_challenge.data.dataset import OcclusionDataset, get_transforms
from data_challenge.models.multitask import MultiTaskOcclusionModel
from data_challenge.train import (
    count_trainable_parameters,
    create_train_loader,
    get_device,
    load_config,
    set_seed,
    warmup_lr_lambda,
)
from data_challenge.utils.logger import setup_logger
from data_challenge.utils.losses import WeightedMSELoss
from data_challenge.utils.metrics import compute_score


def evaluate(model, loader, device) -> tuple[float, float, float, float]:
    model.eval()
    all_preds, all_labels, all_genders, all_gender_preds = [], [], [], []

    with torch.no_grad():
        for images, labels, genders in tqdm(loader, desc="  Validating", leave=False, unit="batch"):
            images = images.to(device)
            outputs = model(images)
            gender_probs = torch.sigmoid(outputs["gender_logits"]).cpu().numpy()
            all_preds.extend(outputs["occlusion"].cpu().numpy())
            all_labels.extend(labels.numpy())
            all_genders.extend(genders.numpy())
            all_gender_preds.extend((gender_probs >= 0.5).astype(np.float32))

    genders_np = np.array(all_genders)
    score, err_f, err_m = compute_score(np.array(all_preds), np.array(all_labels), genders_np)
    gender_acc = float((np.array(all_gender_preds) == genders_np).mean())
    return score, err_f, err_m, gender_acc


def train(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["training"]["seed"])
    logger = setup_logger("train_multitask")

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
    val_ds.dataset = OcclusionDataset(
        csv_path=data_cfg["train_csv"],
        img_root=data_cfg["img_root"],
        transform=get_transforms(train=False, img_size=data_cfg["img_size"]),
    )

    train_loader = create_train_loader(train_ds, full_dataset, cfg)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    logger.info("Train: %d samples | Val: %d samples", train_size, val_size)
    logger.info("Train batching strategy: %s", cfg["training"].get("batching", {}).get("strategy", "random"))

    model_cfg = cfg["model"]
    model = MultiTaskOcclusionModel(
        backbone=model_cfg["backbone"],
        pretrained=model_cfg["pretrained"],
        freeze_backbone=model_cfg.get("freeze_backbone", False),
        regression_head=model_cfg.get("regression_head", {}),
        gender_head=model_cfg.get("gender_head", {}),
    ).to(device)
    trainable_params, total_params = count_trainable_parameters(model)
    logger.info(
        "Model: multitask %s (pretrained=%s, freeze_backbone=%s)",
        model_cfg["backbone"],
        model_cfg["pretrained"],
        model_cfg.get("freeze_backbone", False),
    )
    logger.info("Regression head: %s", model_cfg.get("regression_head", {}))
    logger.info("Gender head: %s", model_cfg.get("gender_head", {}))
    logger.info("Trainable params: %d / %d", trainable_params, total_params)

    train_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
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
    occlusion_criterion = WeightedMSELoss()
    gender_criterion = nn.BCEWithLogitsLoss()
    gender_loss_weight = train_cfg.get("gender_loss_weight", 0.1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg["output"]["dir"], f"{cfg['run_name']}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(run_dir, "config.yaml"))
    logger.info("Run output dir: %s", run_dir)

    best_score = float("inf")
    best_epoch = -1
    best_ckpt = os.path.join(run_dir, "best_model.pt")
    last_ckpt = os.path.join(run_dir, "last_model.pt")

    epoch_bar = tqdm(range(1, train_cfg["epochs"] + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0
        running_occ_loss = 0.0
        running_gender_loss = 0.0

        batch_bar = tqdm(train_loader, desc=f"  Epoch {epoch:02d}", leave=False, unit="batch")
        for images, labels, genders in batch_bar:
            images = images.to(device)
            labels = labels.to(device)
            genders = genders.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            occlusion_loss = occlusion_criterion(outputs["occlusion"], labels)
            gender_loss = gender_criterion(outputs["gender_logits"], genders)
            loss = occlusion_loss + gender_loss_weight * gender_loss
            loss.backward()
            if train_cfg.get("gradient_clip"):
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip"])
            optimizer.step()

            running_loss += loss.item()
            running_occ_loss += occlusion_loss.item()
            running_gender_loss += gender_loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.5f}", occ=f"{occlusion_loss.item():.5f}")

        train_loss = running_loss / len(train_loader)
        train_occ_loss = running_occ_loss / len(train_loader)
        train_gender_loss = running_gender_loss / len(train_loader)

        if epoch <= train_cfg["warmup_epochs"]:
            warmup_scheduler.step()
        else:
            base_scheduler.step()

        score, err_f, err_m, gender_acc = evaluate(model, val_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_bar.set_postfix(loss=f"{train_loss:.5f}", score=f"{score:.5f}", lr=f"{current_lr:.1e}")
        logger.info(
            "Epoch %02d/%02d | loss=%.5f | occ_loss=%.5f | gender_loss=%.5f | "
            "val_score=%.5f | err_F=%.5f | err_M=%.5f | gender_acc=%.4f | lr=%.2e",
            epoch,
            train_cfg["epochs"],
            train_loss,
            train_occ_loss,
            train_gender_loss,
            score,
            err_f,
            err_m,
            gender_acc,
            current_lr,
        )

        if wandb_run:
            wandb_run.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/occlusion_loss": train_occ_loss,
                "train/gender_loss": train_gender_loss,
                "val/score": score,
                "val/err_female": err_f,
                "val/err_male": err_m,
                "val/err_diff": err_f - err_m,
                "val/err_diff_abs": abs(err_f - err_m),
                "val/gender_acc": gender_acc,
                "lr": current_lr,
            })

        ckpt = {"epoch": epoch, "model_state": model.state_dict(), "score": score}
        torch.save(ckpt, last_ckpt)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            torch.save(ckpt, best_ckpt)
            logger.info("  -> New best score %.5f (lower is better), checkpoint saved.", best_score)

    summary = {
        "run_name": cfg["run_name"],
        "model_type": "multitask",
        "backbone": cfg["model"]["backbone"],
        "freeze_backbone": cfg["model"].get("freeze_backbone", False),
        "best_val_score": best_score,
        "best_epoch": best_epoch,
        "epochs": train_cfg["epochs"],
        "gender_loss_weight": gender_loss_weight,
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
    parser = argparse.ArgumentParser(description="Train multitask face occlusion model")
    parser.add_argument("--config", default="src/data_challenge/configs/phase4_multitask_heads/swin_t.yaml")
    args = parser.parse_args()
    train(args.config)
