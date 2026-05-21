"""Post-training bias analysis script.

Loads a checkpoint, runs on the validation set, and logs a detailed
gender-bias breakdown to wandb (and prints it to the console).

Usage:
    python -m data_challenge.analyze
    python -m data_challenge.analyze --config src/data_challenge/configs/base_config.yaml
    python -m data_challenge.analyze --checkpoint outputs/best_model.pt
"""

import argparse
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, random_split

from data_challenge.data.dataset import OcclusionDataset, get_transforms
from data_challenge.models.cnn_baseline import CNNBaseline
from data_challenge.utils.logger import setup_logger
from data_challenge.utils.metrics import weighted_mse


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def analyze(config_path: str, checkpoint: str | None = None):
    cfg = load_config(config_path)
    logger = setup_logger("analyze")
    device = get_device()

    # Load model
    model_cfg = cfg["model"]
    model = CNNBaseline(backbone=model_cfg["backbone"], pretrained=False, dropout=0.0).to(device)
    ckpt_path = checkpoint or os.path.join(cfg["output"]["dir"], "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info("Loaded checkpoint: %s (epoch %d)", ckpt_path, ckpt.get("epoch", -1))

    # Val split (same seed as training so it's the same split)
    data_cfg = cfg["data"]
    transform = get_transforms(train=False, img_size=data_cfg["img_size"])
    full_ds = OcclusionDataset(csv_path=data_cfg["train_csv"], img_root=data_cfg["img_root"], transform=transform)
    val_size = int(len(full_ds) * data_cfg["val_split"])
    train_size = len(full_ds) - val_size
    _, val_ds = random_split(
        full_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg["training"]["seed"]),
    )
    loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"] * 2, num_workers=data_cfg["num_workers"])

    all_preds, all_labels, all_genders = [], [], []
    with torch.no_grad():
        for images, labels, genders in loader:
            preds = model(images.to(device)).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_genders.extend(genders.numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    genders = np.array(all_genders)

    female_mask = genders == 0.0
    male_mask = genders == 1.0

    err_f = weighted_mse(preds[female_mask], labels[female_mask])
    err_m = weighted_mse(preds[male_mask], labels[male_mask])
    err_diff = err_f - err_m  # positive = model is worse on females
    score = (err_f + err_m) / 2 + abs(err_diff)

    # Per-occlusion-bucket breakdown
    buckets = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.01)]
    bucket_stats = {}
    for lo, hi in buckets:
        name = f"{int(lo*100)}-{int(hi*100)}%"
        mask = (labels >= lo) & (labels < hi)
        if mask.sum() == 0:
            continue
        bucket_stats[name] = {
            "n": int(mask.sum()),
            "err_all": float(np.mean((preds[mask] - labels[mask]) ** 2)),
            "err_female": float(np.mean((preds[mask & female_mask] - labels[mask & female_mask]) ** 2)) if (mask & female_mask).sum() else float("nan"),
            "err_male": float(np.mean((preds[mask & male_mask] - labels[mask & male_mask]) ** 2)) if (mask & male_mask).sum() else float("nan"),
        }

    # Print summary
    bias_toward = "females" if err_diff > 0 else "males"
    logger.info("=" * 60)
    logger.info("Gender bias analysis")
    logger.info("  err_female : %.5f", err_f)
    logger.info("  err_male   : %.5f", err_m)
    logger.info("  diff (F-M) : %+.5f  (model is worse on %s)", err_diff, bias_toward)
    logger.info("  score      : %.5f", score)
    logger.info("Occlusion bucket breakdown (MSE):")
    for name, s in bucket_stats.items():
        logger.info("  [%s] n=%d | all=%.5f | F=%.5f | M=%.5f", name, s["n"], s["err_all"], s["err_female"], s["err_male"])
    logger.info("=" * 60)

    # WandB
    if cfg["wandb"]["enabled"]:
        try:
            import wandb
            run = wandb.init(
                project=cfg["wandb"]["project"],
                entity=cfg["wandb"].get("entity") or None,
                name=f"{cfg['run_name']}_analysis",
                config=cfg,
                job_type="analysis",
            )
            run.log({
                "analysis/err_female": err_f,
                "analysis/err_male": err_m,
                "analysis/err_diff": err_diff,
                "analysis/err_diff_abs": abs(err_diff),
                "analysis/score": score,
            })
            # Per-bucket table
            table = wandb.Table(columns=["bucket", "n", "err_all", "err_female", "err_male", "err_diff"])
            for name, s in bucket_stats.items():
                table.add_data(name, s["n"], s["err_all"], s["err_female"], s["err_male"], s["err_female"] - s["err_male"])
            run.log({"analysis/bucket_breakdown": table})
            run.finish()
            logger.info("Results logged to wandb.")
        except Exception as e:
            logger.warning("WandB logging failed: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-training gender bias analysis")
    parser.add_argument("--config", default="src/data_challenge/configs/base_config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    analyze(args.config, checkpoint=args.checkpoint)
