"""Evaluation and inference script.

Three modes (combinable):
  - (default)              val pass only, prints metrics.
  - --save-val-preds       val pass, writes val_predictions.csv (for ensembling).
  - --predict-test         test pass, writes test_predictions.csv (submission).

The val and test passes can be combined in one call.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset, random_split

from data_challenge.data.dataset import OcclusionDataset, get_transforms
from data_challenge.models import build_model
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


def _run_val(model, cfg, data_cfg, transform, device, logger, save_val_preds: bool, run_dir: str | None):
    full_ds = OcclusionDataset(
        csv_path=data_cfg["train_csv"],
        img_root=data_cfg["img_root"],
        transform=transform,
    )
    val_strategy = str(data_cfg.get("val_split_strategy", "random")).lower()
    if val_strategy == "random":
        val_size = int(len(full_ds) * data_cfg["val_split"])
        train_size = len(full_ds) - val_size
        _, val_ds = random_split(
            full_ds,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(cfg["training"]["seed"]),
        )
    else:
        # Must match train.py exactly so the reported val score lines up with
        # the score the checkpoint was selected on.
        from data_challenge.data.splits import make_val_indices

        _, val_idx = make_val_indices(full_ds.df, data_cfg, cfg["training"]["seed"])
        val_ds = Subset(full_ds, val_idx)
    val_size = len(val_ds)

    loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    logger.info("Evaluating on %d validation samples...", val_size)

    # Recover the filename for each val sample (needed for save_val_preds).
    # random_split keeps a `.indices` field on the Subset.
    val_indices = val_ds.indices
    val_filenames = full_ds.df.iloc[val_indices]["filename"].tolist()

    all_preds, all_labels, all_genders = [], [], []
    with torch.no_grad():
        for images, labels, genders in loader:
            preds = model(images.to(device)).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_genders.extend(genders.numpy())

    preds_arr = np.array(all_preds, dtype=np.float64)
    labels_arr = np.array(all_labels, dtype=np.float64)
    genders_arr = np.array(all_genders, dtype=np.float64)

    score, err_f, err_m = compute_score(preds_arr, labels_arr, genders_arr)
    logger.info(
        "Val score=%.5f (lower is better) | err_female=%.5f | err_male=%.5f | err_diff=%.5f",
        score, err_f, err_m, err_f - err_m,
    )

    if save_val_preds:
        out_dir = run_dir or cfg["output"]["dir"]
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "val_predictions.csv")
        pd.DataFrame({
            "filename": val_filenames,
            "GT": labels_arr,
            "FaceOcclusion": preds_arr,
            "gender": genders_arr,
        }).to_csv(out_path, index=False)
        logger.info("Val predictions saved to %s", out_path)


def evaluate(
    config_path: str,
    checkpoint: str | None = None,
    predict_test: bool = False,
    save_val_preds: bool = False,
):
    cfg = load_config(config_path)
    logger = setup_logger("eval")

    device = get_device()
    logger.info("Using device: %s", device)

    # Model (factory). For cnn_baseline we skip the pretrained download since the
    # checkpoint will overwrite everything anyway; for dinov3 we keep pretrained=True
    # because the HF config + weights are needed to construct the architecture.
    if cfg["model"].get("name", "cnn_baseline") == "cnn_baseline":
        cfg["model"]["pretrained"] = False
        cfg["model"]["dropout"] = 0.0
    model = build_model(cfg).to(device)

    ckpt_path = checkpoint or os.path.join(cfg["output"]["dir"], "best_model.pt")
    logger.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info(
        "Checkpoint epoch=%d | saved_score=%.5f",
        ckpt.get("epoch", -1), ckpt.get("score", float("nan")),
    )

    data_cfg = cfg["data"]
    transform = get_transforms(train=False, img_size=data_cfg["img_size"])

    # When the checkpoint lives under a run dir, drop val/test predictions there too.
    run_dir = os.path.dirname(os.path.abspath(ckpt_path)) if checkpoint else None

    # Decide which passes to run. Default (no flags) = val only.
    run_val = save_val_preds or not predict_test
    run_test = predict_test

    if run_val:
        _run_val(model, cfg, data_cfg, transform, device, logger, save_val_preds, run_dir)
    if run_test:
        _predict_test(model, cfg, data_cfg, transform, device, logger, run_dir)


def _predict_test(model, cfg, data_cfg, transform, device, logger, run_dir: str | None):
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

    out_dir = run_dir or cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test_predictions.csv")
    pd.DataFrame({"filename": all_filenames, "FaceOcclusion": all_preds, "gender": "x"}).to_csv(out_path, index=False)
    logger.info("Test predictions saved to %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate face occlusion model")
    parser.add_argument("--config", default="src/data_challenge/configs/base_config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint (default: output/best_model.pt)")
    parser.add_argument("--predict-test", action="store_true", help="Generate test set predictions")
    parser.add_argument(
        "--save-val-preds",
        action="store_true",
        help="Also write val_predictions.csv (with GT + gender) for ensembling",
    )
    args = parser.parse_args()
    evaluate(
        args.config,
        checkpoint=args.checkpoint,
        predict_test=args.predict_test,
        save_val_preds=args.save_val_preds,
    )
