# Face Occlusion Prediction — Data Challenge 2026

**Group:** OccluVision (Group 8)

Predict the face occlusion percentage from 224×224 face crops. This repository contains the code to reproduce our leaderboard submission (DINOv3-L + LoRA).

## Requirements

- Python ≥ 3.11
- CUDA GPU (recommended: ≥24 GB VRAM for DINOv3-L, batch size 64)
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

```bash
git clone https://github.com/LeHoang510/Face-Occlusion-Prediction.git
cd Face-Occlusion-Prediction
uv sync
```

## Data

1. Download metadata from the challenge zip (train/test CSVs).
2. Download images from [IMT share](https://partage.imt.fr/index.php/s/ntYk27ZFCbeKGqW).
3. Place files as follows:

```
dataset/
  DataChallenge2026/occlusion_datasets/
    train.csv
    test_students.csv
  crops/Crop_224_5fp_100K/
    <image files>
```

## DINOv3 access

The best model uses `facebook/dinov3-vitl16-pretrain-lvd1689m` (gated on Hugging Face). Request access, then:

```bash
export HF_TOKEN=<your_huggingface_token>
```

## Train (submitted model)

```bash
uv run python src/data_challenge/train.py \
  --config src/data_challenge/configs/dinov3_l_lora.yaml
```

Checkpoints and logs are saved under `outputs/dinov3_l_lora_<timestamp>/`.

## Evaluate & predict

```bash
CKPT=outputs/dinov3_l_lora_<timestamp>/best_model.pt

# Validation score (weighted MSE + gender fairness)
uv run python src/data_challenge/eval.py \
  --config src/data_challenge/configs/dinov3_l_lora.yaml \
  --checkpoint "$CKPT"

# Test-set predictions for leaderboard submission
uv run python src/data_challenge/eval.py \
  --config src/data_challenge/configs/dinov3_l_lora.yaml \
  --checkpoint "$CKPT" \
  --predict-test
```

Predictions are written to `outputs/test_predictions.csv`.

## Cluster (SLURM)

```bash
sbatch --export=ALL,CONFIG=src/data_challenge/configs/dinov3_l_lora.yaml \
  scripts/slurm/train.sh
```

## Other experiments

Phase-1 CNN/ViT baselines live under `src/data_challenge/configs/archive/phase1_baselines/`.
Phase-2 balanced-gender batching configs are in `src/data_challenge/configs/phase2_balanced_batching/`.
