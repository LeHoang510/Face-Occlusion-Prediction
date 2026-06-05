#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# vast_run.sh — SLURM-less equivalent of scripts/slurm/train.sh
#
# Designed for vast.ai (or any plain container). On success, auto-launches
# val eval + test prediction. Tees stdout/stderr to logs/vast/<ts>.log so the
# run is recoverable if SSH disconnects.
#
# Usage:
#   bash scripts/vast_run.sh src/data_challenge/configs/dinov3_l_full_ft.yaml
#   SKIP_EVAL=1 bash scripts/vast_run.sh <config>
#
# Tip: run inside `nohup`, `tmux`, or `screen` so the job survives SSH drop:
#   tmux new -s train -d "bash scripts/vast_run.sh src/data_challenge/configs/dinov3_l_full_ft.yaml"
# ---------------------------------------------------------------------------
set -euo pipefail

CONFIG="${1:?usage: vast_run.sh <config-yaml> ; e.g. src/data_challenge/configs/dinov3_l_full_ft.yaml}"
SKIP_EVAL="${SKIP_EVAL:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "$PROJECT_ROOT"

# Load .env
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Caches & paths
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/.torch_cache}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$HF_HOME" "$TORCH_HOME" logs/vast outputs

LOG="logs/vast/run_$(date +%Y%m%d_%H%M%S).log"
echo "[vast-run] config=$CONFIG | log=$LOG"
echo "[vast-run] host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true

# Pick runner. vast_setup.sh creates a venv with --system-site-packages so
# torch comes from /opt/conda. Activating .venv prepends its site-packages
# but still inherits the system ones — that's the whole trick.
export PATH="$HOME/.local/bin:$PATH"
if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PY="python"
else
  echo "[vast-run] ERROR: .venv missing — run 'bash scripts/vast_setup.sh' first" >&2
  exit 1
fi

set -o pipefail

# Force the venv's bundled NVIDIA libs (NCCL, cuDNN, cuBLAS…) to take priority
# over the container's /usr/lib/libnccl (which may be too old). Recursive find
# handles both classic (nvidia/nccl/lib/) and cu-namespaced (nvidia/cu13/lib/)
# wheel layouts.
NVIDIA_LIB_DIRS=$(find .venv -type d -name 'lib' -path '*/nvidia/*' 2>/dev/null | sort -u)
for d in $NVIDIA_LIB_DIRS; do
  LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
export LD_LIBRARY_PATH

# Pre-flight CUDA check. Full-FT of ViT-L on CPU = ~55h/epoch and would burn
# vast.ai $$ for nothing. Refuse to start if torch can't see the GPU.
CUDA_OK=$($PY -c "import torch; print(int(torch.cuda.is_available()))" 2>/dev/null || echo 0)
if [[ "$CUDA_OK" != "1" ]]; then
  $PY - <<'PY' || true
import torch
print(f"[vast-run] torch={torch.__version__}  built_for_cuda={torch.version.cuda}  device_count={torch.cuda.device_count()}", flush=True)
PY
  echo "[vast-run] ERROR: torch.cuda.is_available() = False. Refusing to run on CPU." >&2
  echo "[vast-run] Re-run 'bash scripts/vast_setup.sh' (it now auto-pins the right torch wheel)" >&2
  echo "[vast-run] or destroy this instance and pick one with newer drivers (CUDA >= 12.6)." >&2
  exit 3
fi
echo "[vast-run] CUDA OK ($($PY -c 'import torch; print(torch.cuda.get_device_name(0))'))" | tee -a "$LOG"

START_TS=$(date +%s)

if $PY src/data_challenge/train.py --config "$CONFIG" 2>&1 | tee -a "$LOG"; then
  ELAPSED=$(( $(date +%s) - START_TS ))
  echo "[vast-run] training success in ${ELAPSED}s" | tee -a "$LOG"

  if [[ "$SKIP_EVAL" == "1" ]]; then
    echo "[vast-run] SKIP_EVAL=1, done." | tee -a "$LOG"
    exit 0
  fi

  LATEST_RUN=$(ls -td outputs/*/ 2>/dev/null | head -n 1 || true)
  if [[ -z "$LATEST_RUN" || ! -f "${LATEST_RUN}best_model.pt" ]]; then
    echo "[vast-run] no best_model.pt under outputs/, skipping eval" >&2 | tee -a "$LOG"
    exit 0
  fi
  CKPT="${LATEST_RUN%/}/best_model.pt"
  echo "[vast-run] eval checkpoint: $CKPT" | tee -a "$LOG"

  $PY src/data_challenge/eval.py --config "$CONFIG" --checkpoint "$CKPT" 2>&1 | tee -a "$LOG" || \
    echo "[vast-run] val pass failed" >&2 | tee -a "$LOG"

  $PY src/data_challenge/eval.py --config "$CONFIG" --checkpoint "$CKPT" --predict-test 2>&1 | tee -a "$LOG" || \
    echo "[vast-run] test pass failed" >&2 | tee -a "$LOG"

  echo "[vast-run] done." | tee -a "$LOG"
else
  echo "[vast-run] training FAILED — see $LOG" >&2
  exit 1
fi
