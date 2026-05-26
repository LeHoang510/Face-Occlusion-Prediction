#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# cluster_sync_push.sh — local → cluster (rsync over SSH)
#
# Defaults exclude .venv/, caches, .git/, wandb/, and the heavy outputs/
# (alias: runs/) and dataset/ (alias: artifacts/) trees. Use --with-runs or
# --with-data to include them.
#
# Env overrides:
#   CLUSTER_HOST   default: gpu        (must resolve via ~/.ssh/config)
#   REMOTE_DIR     default: ~/Face-Occlusion-Prediction
#
# Usage:
#   ./scripts/cluster_sync_push.sh                  # code only
#   ./scripts/cluster_sync_push.sh --with-runs      # also push outputs/, logs/
#   ./scripts/cluster_sync_push.sh --with-data      # also push dataset/, artifacts/
#   ./scripts/cluster_sync_push.sh --dry-run        # show what would be sent
# ---------------------------------------------------------------------------
set -euo pipefail

CLUSTER_HOST="${CLUSTER_HOST:-gpu}"
REMOTE_DIR="${REMOTE_DIR:-~/Face-Occlusion-Prediction}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

WITH_RUNS=0
WITH_DATA=0
DRY=""
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-runs) WITH_RUNS=1; shift ;;
    --with-data) WITH_DATA=1; shift ;;
    --dry-run)   DRY="--dry-run"; shift ;;
    -h|--help)
      sed -n '2,21p' "$0"; exit 0 ;;
    --)          shift; EXTRA+=("$@"); break ;;
    *)           EXTRA+=("$1"); shift ;;
  esac
done

EXCLUDES=(
  --exclude '.venv/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.pytest_cache/'
  --exclude '.ruff_cache/'
  --exclude '.mypy_cache/'
  --exclude '.git/'
  --exclude '.idea/'
  --exclude '.vscode/'
  --exclude 'wandb/'
  --exclude '.DS_Store'
  --exclude '.hf_cache/'
  --exclude '.torch_cache/'
)

if [[ $WITH_RUNS -eq 0 ]]; then
  EXCLUDES+=(--exclude '/outputs/' --exclude '/runs/' --exclude '/logs/')
fi
if [[ $WITH_DATA -eq 0 ]]; then
  # Anchor to project root: unanchored `data/` also excludes src/data_challenge/data/.
  EXCLUDES+=(--exclude '/dataset/' --exclude '/data/' --exclude '/artifacts/')
fi

echo "[push] ${LOCAL_DIR}/  →  ${CLUSTER_HOST}:${REMOTE_DIR}/"
echo "[push] with-runs=${WITH_RUNS}  with-data=${WITH_DATA}  dry-run=${DRY:-0}"

ssh "$CLUSTER_HOST" "mkdir -p ${REMOTE_DIR}"

rsync -avzP $DRY \
  "${EXCLUDES[@]}" \
  "${EXTRA[@]}" \
  "${LOCAL_DIR}/" \
  "${CLUSTER_HOST}:${REMOTE_DIR}/"

echo "[push] done."
