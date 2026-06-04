#!/usr/bin/env bash
#SBATCH --job-name=focc-eval
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# ---------------------------------------------------------------------------
# Expected env (passed via `sbatch --export=ALL,...`):
#   CONFIG       path to yaml config — required
#   CHECKPOINT   path to .pt checkpoint — required
#   MODE         "val" (default) | "test" | "both"
# ---------------------------------------------------------------------------
set -euo pipefail
mkdir -p logs/slurm

# sbatch copies this script into /var/spool/slurmd/...; BASH_SOURCE would point there.
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/Face-Occlusion-Prediction}"
# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/slurm/_common.sh"

CONFIG="${CONFIG:?eval.sh: CONFIG required}"
CHECKPOINT="${CHECKPOINT:?eval.sh: CHECKPOINT required}"
MODE="${MODE:-val}"

run_val()  { $RUN_PREFIX python src/data_challenge/eval.py --config "$CONFIG" --checkpoint "$CHECKPOINT"; }
run_test() { $RUN_PREFIX python src/data_challenge/eval.py --config "$CONFIG" --checkpoint "$CHECKPOINT" --predict-test; }

case "$MODE" in
  val)   run_val ;;
  test)  run_test ;;
  both)  run_val && run_test ;;
  *)     echo "eval.sh: MODE must be val|test|both (got: $MODE)" >&2; exit 1 ;;
esac

echo "[eval] done (mode=$MODE)."
