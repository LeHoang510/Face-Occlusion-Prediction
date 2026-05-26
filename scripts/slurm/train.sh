#!/usr/bin/env bash
#SBATCH --job-name=focc-train
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# ---------------------------------------------------------------------------
# Expected env (passed via `sbatch --export=ALL,CONFIG=...`):
#   CONFIG          path (relative to project root) to a yaml config — required
#   SKIP_EVAL       if set to 1, do not chain eval after training (default: chain)
#
# On training success, the script auto-launches:
#   - val eval on the best checkpoint
#   - test prediction (writes outputs/test_predictions.csv)
# ---------------------------------------------------------------------------
set -euo pipefail
mkdir -p logs/slurm

# sbatch copies this script into /var/spool/slurmd/...; BASH_SOURCE would point there.
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/Face-Occlusion-Prediction}"
# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/slurm/_common.sh"

CONFIG="${CONFIG:?train.sh: CONFIG env var required (path to yaml)}"
SKIP_EVAL="${SKIP_EVAL:-0}"

echo "[train] starting: config=$CONFIG"
START_TS=$(date +%s)

if $RUN_PREFIX python src/data_challenge/train.py --config "$CONFIG"; then
  ELAPSED=$(( $(date +%s) - START_TS ))
  echo "[train] success in ${ELAPSED}s"

  if [[ "$SKIP_EVAL" == "1" ]]; then
    echo "[train] SKIP_EVAL=1, skipping eval"
    exit 0
  fi

  # Pick the most recently-modified run directory under outputs/.
  LATEST_RUN=$(ls -td outputs/*/ 2>/dev/null | head -n 1 || true)
  if [[ -z "$LATEST_RUN" || ! -f "${LATEST_RUN}best_model.pt" ]]; then
    echo "[train] could not locate a best_model.pt under outputs/, skipping eval" >&2
    exit 0
  fi
  CKPT="${LATEST_RUN%/}/best_model.pt"
  echo "[eval]  using checkpoint: $CKPT"

  echo "[eval]  validation pass"
  $RUN_PREFIX python src/data_challenge/eval.py --config "$CONFIG" --checkpoint "$CKPT" || \
    echo "[eval]  validation pass failed" >&2

  echo "[eval]  test prediction pass"
  $RUN_PREFIX python src/data_challenge/eval.py --config "$CONFIG" --checkpoint "$CKPT" --predict-test || \
    echo "[eval]  test prediction failed" >&2

  echo "[done]  job complete."
else
  echo "[train] FAILED — see logs/slurm/${SLURM_JOB_NAME:-focc-train}-${SLURM_JOB_ID:-?}.err" >&2
  exit 1
fi
