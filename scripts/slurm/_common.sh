# ---------------------------------------------------------------------------
# _common.sh — shared environment setup for SLURM jobs on the ENST cluster
#
# Sourced (not executed) by train.sh / eval.sh after their #SBATCH directives.
# Responsibilities:
#   - cd into the project root
#   - load .env if present
#   - activate the uv-managed venv if present
#   - point HF / Torch caches into the project (so they persist across jobs)
#   - print a small context banner (host, GPU, config)
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/Face-Occlusion-Prediction}"
cd "$PROJECT_ROOT"

# Load .env (KEY=VALUE format) into the environment.
if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Ensure `import data_challenge.*` resolves when the editable install is missing/broken.
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

# Activate uv-managed venv if it exists; otherwise we'll rely on `uv run`.
if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  RUN_PREFIX=""
elif command -v uv >/dev/null 2>&1; then
  RUN_PREFIX="uv run"
else
  echo "[common] WARNING: no .venv and no uv on PATH; relying on PYTHONPATH only" >&2
  RUN_PREFIX=""
fi
export RUN_PREFIX

# Cache placement (persists across jobs, kept out of rsync via excludes).
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/.torch_cache}"
mkdir -p "$HF_HOME" "$TORCH_HOME"

# Sane defaults for shared nodes.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

echo "[common] PROJECT_ROOT=$PROJECT_ROOT"
echo "[common] HOST=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[common] CONFIG=${CONFIG:-<unset>}  CHECKPOINT=${CHECKPOINT:-<unset>}"
echo "[common] RUN_PREFIX='$RUN_PREFIX'"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
fi
