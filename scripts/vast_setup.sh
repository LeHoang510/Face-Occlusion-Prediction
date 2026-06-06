#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# vast_setup.sh — bootstrap a fresh vast.ai instance for training
#
# Strategy: reuse the container's pre-installed torch (via uv venv
# --system-site-packages) instead of reinstalling it. The pytorch/pytorch:*
# images ship a working torch+CUDA+NCCL bundle, and reinstalling from PyPI
# kept hitting wheel/NCCL ABI mismatches (nvidia-nccl-cu12 wheels missing
# ncclCommResume despite being recent).
#
# Assumes:
#   - the repo has been rsync'd to /workspace/Face-Occlusion-Prediction
#     (from local:  CLUSTER_HOST=vast REMOTE_DIR=/workspace/Face-Occlusion-Prediction \
#                   ./scripts/cluster_sync_push.sh --with-data)
#   - HF_TOKEN and (optionally) WANDB_API_KEY are in .env at the repo root
#
# Usage:
#   cd /workspace/Face-Occlusion-Prediction
#   bash scripts/vast_setup.sh
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "$PROJECT_ROOT"

# Pre-flight
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python --version 2>/dev/null || true

# Install uv (needed for venv creation and our pip)
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
grep -q '.local/bin' ~/.bashrc 2>/dev/null || \
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# ---------------------------------------------------------------------------
# Locate a system Python that ALREADY has a working torch + CUDA.
# Walking the PyTorch-image conventions first, then standard paths.
# ---------------------------------------------------------------------------
SYS_PYTHON=""
for cand in /opt/conda/bin/python python3.12 python3.11 python3.10 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c "import torch; assert torch.cuda.is_available() and torch.cuda.device_count() > 0" 2>/dev/null; then
      SYS_PYTHON=$(command -v "$cand")
      break
    fi
  fi
done

if [[ -z "$SYS_PYTHON" ]]; then
  cat >&2 <<EOF
[setup] ERROR: no system Python with a working torch + CUDA was found.
[setup] This image probably doesn't have torch pre-installed.
[setup] Pick a vast.ai image with torch baked in, e.g.:
[setup]   pytorch/pytorch:2.6.0-cuda12.8-cudnn9-devel
[setup]   pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
[setup] Override with: ./scripts/cluster_exec.sh vast launch <ask> --image <img>
EOF
  exit 3
fi

SYS_TORCH=$("$SYS_PYTHON" -c "import torch; print(torch.__version__)")
SYS_TORCH_CUDA=$("$SYS_PYTHON" -c "import torch; print(torch.version.cuda)")
echo "[setup] container python: $SYS_PYTHON"
echo "[setup] container torch:  $SYS_TORCH (built for CUDA $SYS_TORCH_CUDA)"

# transformers>=4.50 imports torch.float8_e8m0fnu at module load (PyTorch >= 2.6).
# Older pytorch/* images still work via our FP8 shim, but 2.6+ is preferred.
if ! "$SYS_PYTHON" -c "import torch; assert hasattr(torch, 'float8_e8m0fnu')" 2>/dev/null; then
  echo "[setup] WARN: torch $SYS_TORCH lacks float8_e8m0fnu (need 2.6+ for native FP8)."
  echo "[setup]        Training will use an FP8 compat shim; prefer image pytorch/pytorch:2.6.0-cuda12.8-cudnn9-devel"
fi

# ---------------------------------------------------------------------------
# Create venv that INHERITS the container's site-packages.
# This means `import torch` in the venv resolves to the container's torch.
# Wipe any partial venv from previous failed runs so uv builds cleanly.
# ---------------------------------------------------------------------------
echo "[setup] (re)creating venv with --system-site-packages"
rm -rf .venv
uv venv --system-site-packages --python "$SYS_PYTHON" .venv
# shellcheck disable=SC1091
source .venv/bin/activate

# ---------------------------------------------------------------------------
# Install ONLY our extras. We use plain `pip` (not `uv pip`) here because pip
# respects --system-site-packages when checking what's already satisfied — so
# transformers' transitive `torch` dep is considered installed (it is, via
# /opt/conda) and pip will not pull a fresh torch wheel into the venv.
# ---------------------------------------------------------------------------
echo "[setup] installing project (no deps) + extras"
pip install --quiet --no-deps -e .

# Each of these may depend on torch — pip sees torch in the system path and
# skips reinstalling it. Their other (lightweight) deps go into the venv.
pip install --quiet \
  "transformers>=4.50.0" \
  "peft>=0.11.0" \
  "timm>=1.0.0" \
  "wandb>=0.16.0" \
  "pyyaml>=6.0" \
  "pandas>=2.0.0" \
  "pillow>=10.0.0" \
  "tqdm>=4.67.3" \
  "matplotlib>=3.10.9" \
  accelerate \
  "huggingface-hub>=0.24.0"

# ---------------------------------------------------------------------------
# Sanity: torch must still see CUDA after our pip installs.
# ---------------------------------------------------------------------------
CUDA_OK=$(python -c "import torch; print(int(torch.cuda.is_available()))" 2>/dev/null || echo 0)
if [[ "$CUDA_OK" != "1" ]]; then
  {
    echo "[setup] ERROR: torch lost CUDA after our installs (we likely shadowed system torch)."
    echo "  which python : $(command -v python)"
    python -c "import torch, sys; print(f'  torch={torch.__version__} from {torch.__file__}'); print(f'  python={sys.version}')" || true
  } >&2
  exit 3
fi
echo "[setup] CUDA OK ($(python -c 'import torch; print(torch.cuda.get_device_name(0))'))"

# ---------------------------------------------------------------------------
# Secrets: load .env and log into HF / W&B if tokens are present.
# ---------------------------------------------------------------------------
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "[setup] huggingface login"
  python -c "from huggingface_hub import login; login(token='${HF_TOKEN}', add_to_git_credential=False)"
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "[setup] wandb login"
  wandb login --relogin "$WANDB_API_KEY" >/dev/null || true
fi

# Optional pre-warm of DINOv3-L weights to save time on the first epoch.
if [[ "${PREWARM_HF:-0}" == "1" ]]; then
  echo "[setup] pre-downloading facebook/dinov3-vitl16-pretrain-lvd1689m"
  python -c "from data_challenge.utils import torch_compat; from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m', trust_remote_code=True)"
fi

# Smoke-test: DINOv3 class import (catches transformers/torch ABI mismatches early).
echo "[setup] smoke-test: DINOv3ViTModel import"
python -c "from data_challenge.utils import torch_compat; from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel; print('DINOv3ViTModel OK')" \
  || { echo "[setup] ERROR: DINOv3 import failed — check torch/transformers versions above" >&2; exit 3; }

echo "[setup] done. Next: bash scripts/vast_run.sh src/data_challenge/configs/dinov3_l_full_ft.yaml"
