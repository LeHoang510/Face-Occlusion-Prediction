#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# vast_setup.sh — bootstrap a fresh vast.ai instance for training
#
# Assumes:
#   - the repo has been rsync'd to the instance (e.g. /workspace/Face-Occlusion-Prediction)
#     -> from local:  CLUSTER_HOST=vast REMOTE_DIR=/workspace/Face-Occlusion-Prediction \
#                     ./scripts/cluster_sync_push.sh --with-data
#   - HF_TOKEN and (optionally) WANDB_API_KEY are in .env at the repo root
#
# Usage on the instance:
#   cd /workspace/Face-Occlusion-Prediction
#   bash scripts/vast_setup.sh
# ---------------------------------------------------------------------------
set -euo pipefail

# Pre-flight
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python --version || true

# Install uv (one-liner, no sudo)
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
grep -q '.local/bin' ~/.bashrc 2>/dev/null || \
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Sync deps. uv resolves torch with the CUDA wheels available in the container.
echo "[setup] uv sync"
uv sync

# Load .env if present (HF_TOKEN, WANDB_API_KEY, ...)
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# HuggingFace login (gated DINOv3 weights require it)
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "[setup] huggingface login"
  uv run python -c "from huggingface_hub import login; login(token='${HF_TOKEN}', add_to_git_credential=False)"
fi

# WandB login
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "[setup] wandb login"
  uv run wandb login --relogin "$WANDB_API_KEY" >/dev/null || true
fi

# Pre-warm HF cache for DINOv3-L (optional; saves time on first epoch)
if [[ "${PREWARM_HF:-0}" == "1" ]]; then
  echo "[setup] pre-downloading facebook/dinov3-vitl16-pretrain-lvd1689m"
  uv run python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m', trust_remote_code=True)"
fi

echo "[setup] done. Next: bash scripts/vast_run.sh src/data_challenge/configs/dinov3_l_full_ft.yaml"
