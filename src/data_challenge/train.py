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
from data_challenge.data.samplers import BalancedGenderBatchSampler
from data_challenge.models import build_model
from data_challenge.utils.logger import setup_logger
from data_challenge.utils.losses import build_criterion
from data_challenge.utils.metrics import compute_score


_AMP_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def build_optimizer(model: nn.Module, cfg: dict, trainable_params: list[nn.Parameter]):
    """Return AdamW. Uses head/backbone LR split + LLRD when configured for DINOv3.

    Activated when both `training.learning_rate_head` and `training.learning_rate_backbone`
    are set AND the model is dinov3. Otherwise falls back to the single-LR AdamW.
    """
    train_cfg = cfg["training"]
    model_name = cfg["model"].get("name", "cnn_baseline")
    lr_head = train_cfg.get("learning_rate_head")
    lr_bb = train_cfg.get("learning_rate_backbone")

    if model_name == "dinov3" and lr_head is not None and lr_bb is not None:
        from data_challenge.models.dinov3 import build_param_groups

        groups = build_param_groups(
            model,
            lr_backbone=float(lr_bb),
            lr_head=float(lr_head),
            weight_decay=float(train_cfg["weight_decay"]),
            llrd_decay=float(train_cfg.get("llrd_decay", 1.0)),
            n_layers=int(cfg["model"].get("n_layers", 24)),
        )
        return torch.optim.AdamW(groups), True

    return (
        torch.optim.AdamW(
            trainable_params,
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        ),
        False,
    )


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


def create_train_loader(train_ds, full_dataset, cfg):
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    batching_cfg = train_cfg.get("batching", {})
    strategy = batching_cfg.get("strategy", "random")

    if strategy == "balanced_gender":
        subset_genders = full_dataset.df.iloc[train_ds.indices]["gender"].tolist()
        batch_sampler = BalancedGenderBatchSampler(
            genders=subset_genders,
            batch_size=train_cfg["batch_size"],
            generator=torch.Generator().manual_seed(train_cfg["seed"]),
            drop_last=batching_cfg.get("drop_last", False),
        )
        return DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=data_cfg["num_workers"],
            pin_memory=True,
        )

    return DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )


def train(config_path: str, resume: str | None = None):
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

    # Model (factory: cnn_baseline | dinov3)
    model = build_model(cfg).to(device)

    # Optional gradient checkpointing (essential for full FT of ViT-L on 24G GPUs)
    if cfg["model"].get("gradient_checkpointing", False):
        if hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()
            logger.info("Gradient checkpointing: ENABLED")
        else:
            logger.warning("gradient_checkpointing requested but model has no such method")

    n_total = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    logger.info(
        "Model: %s | trainable=%s / total=%s (%.2f%%)",
        cfg["model"].get("name", "cnn_baseline"),
        f"{n_trainable:,}", f"{n_total:,}", 100.0 * n_trainable / max(n_total, 1),
    )

    # Optimizer & scheduler — head/backbone LR split + LLRD when configured.
    train_cfg = cfg["training"]
    optimizer, used_split_lr = build_optimizer(model, cfg, trainable_params)
    if used_split_lr:
        logger.info(
            "Optimizer: AdamW with %d param groups (head_lr=%.2e, backbone_lr=%.2e, llrd=%.2f)",
            len(optimizer.param_groups),
            float(train_cfg["learning_rate_head"]),
            float(train_cfg["learning_rate_backbone"]),
            float(train_cfg.get("llrd_decay", 1.0)),
        )
    else:
        logger.info("Optimizer: AdamW single group (lr=%.2e)", float(train_cfg["learning_rate"]))

    # Mixed precision setup
    amp_dtype_str = str(train_cfg.get("amp_dtype", "fp32")).lower()
    if amp_dtype_str not in _AMP_DTYPES:
        raise ValueError(f"amp_dtype must be one of {list(_AMP_DTYPES)}, got {amp_dtype_str!r}")
    amp_dtype = _AMP_DTYPES[amp_dtype_str]
    use_amp = amp_dtype != torch.float32 and device.type == "cuda"
    # GradScaler is only needed for fp16; bf16 has fp32 dynamic range
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    grad_accum = max(int(train_cfg.get("gradient_accumulation_steps", 1)), 1)
    effective_bs = train_cfg["batch_size"] * grad_accum
    logger.info(
        "Precision: amp_dtype=%s use_amp=%s scaler=%s | grad_accum=%d (effective batch=%d)",
        amp_dtype_str, use_amp, scaler.is_enabled(), grad_accum, effective_bs,
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

    criterion = build_criterion(cfg).to(device)
    loss_name = cfg["training"].get("loss", "weighted_mse")
    logger.info("Loss: %s", loss_name)

    # Output dir: outputs/<run_name>_<timestamp>/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg["output"]["dir"], f"{cfg['run_name']}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(run_dir, "config.yaml"))
    logger.info("Run output dir: %s", run_dir)

    best_score = float("inf")
    best_epoch = 0
    best_ckpt = os.path.join(run_dir, "best_model.pt")
    last_ckpt = os.path.join(run_dir, "last_model.pt")

    # -----------------------------------------------------------------------
    # Resume: load weights (+ optimizer state if present), fast-forward
    # schedulers so the cosine curve continues from the right point.
    # -----------------------------------------------------------------------
    start_epoch = 1
    if resume is not None:
        logger.info("Resuming from: %s", resume)
        ckpt_in = torch.load(resume, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt_in["model_state"], strict=False)
        if missing:
            logger.warning("  %d missing keys", len(missing))
        if unexpected:
            logger.warning("  %d unexpected keys", len(unexpected))
        if "optimizer_state" in ckpt_in:
            try:
                optimizer.load_state_dict(ckpt_in["optimizer_state"])
                logger.info("  optimizer state restored")
            except Exception as e:
                logger.warning("  optimizer state restore failed: %s", e)
        else:
            logger.info("  optimizer state not in checkpoint — using fresh AdamW state (warm-start)")
        last_epoch = int(ckpt_in.get("epoch", 0))
        start_epoch = last_epoch + 1
        best_score = float(ckpt_in.get("best_score", ckpt_in.get("score", float("inf"))))
        best_epoch = int(ckpt_in.get("best_epoch", last_epoch))
        # Fast-forward schedulers to start_epoch (replays the per-epoch step pattern)
        for ep in range(1, start_epoch):
            if ep <= train_cfg["warmup_epochs"]:
                warmup_scheduler.step()
            else:
                base_scheduler.step()
        logger.info(
            "  resumed: start_epoch=%d  prev_best=%.5f@ep%d  current_lr=%.2e",
            start_epoch, best_score, best_epoch, optimizer.param_groups[0]["lr"],
        )

    if start_epoch > train_cfg["epochs"]:
        logger.warning(
            "Nothing to do: checkpoint already at epoch %d, config.epochs=%d. "
            "Bump training.epochs in the config to continue.",
            start_epoch - 1, train_cfg["epochs"],
        )
        return

    epoch_bar = tqdm(range(start_epoch, train_cfg["epochs"] + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0
        n_batches = len(train_loader)

        optimizer.zero_grad(set_to_none=True)
        batch_bar = tqdm(train_loader, desc=f"  Epoch {epoch:02d}", leave=False, unit="batch")
        for step, (images, labels, genders) in enumerate(batch_bar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            genders = genders.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                preds = model(images)
                loss = criterion(preds, labels, genders) / grad_accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            is_boundary = (step + 1) % grad_accum == 0 or (step + 1) == n_batches
            if is_boundary:
                if train_cfg.get("gradient_clip"):
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(trainable_params, train_cfg["gradient_clip"])
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # un-scale for logging the raw per-batch loss
            unscaled = loss.item() * grad_accum
            running_loss += unscaled
            batch_bar.set_postfix(loss=f"{unscaled:.5f}")

        train_loss = running_loss / n_batches

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

        # last_model.pt carries the full state (model + optimizer) so it is
        # resumable. best_model.pt stays lightweight (model only) for inference.
        last_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "score": score,
            "best_score": best_score if score >= best_score else score,
            "best_epoch": best_epoch if score >= best_score else epoch,
        }
        torch.save(last_payload, last_ckpt)

        if score < best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "score": score},
                best_ckpt,
            )
            logger.info("  -> New best score %.5f (lower is better), checkpoint saved.", best_score)

    summary = {
        "run_name": cfg["run_name"],
        "model_name": cfg["model"].get("name", "cnn_baseline"),
        "backbone": cfg["model"].get("backbone") or cfg["model"].get("model_id"),
        "best_val_score": best_score,
        "best_epoch": best_epoch,
        "epochs": train_cfg["epochs"],
        "trainable_params": n_trainable,
        "total_params": n_total,
        "amp_dtype": amp_dtype_str,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size": effective_bs,
        "split_lr": used_split_lr,
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
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint to resume from (best_model.pt or last_model.pt)",
    )
    args = parser.parse_args()
    train(args.config, resume=args.resume)
