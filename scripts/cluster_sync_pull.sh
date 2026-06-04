#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# cluster_sync_pull.sh — cluster → local (rsync over SSH)
#
# By default merges outputs/ (runs) and logs/ back. Use --only-runs /
# --only-artifacts / --all to pick what to pull.
#
# Env overrides:
#   CLUSTER_HOST   default: gpu
#   REMOTE_DIR     default: ~/Face-Occlusion-Prediction
#
# Usage:
#   ./scripts/cluster_sync_pull.sh                  # outputs/ + logs/
#   ./scripts/cluster_sync_pull.sh --only-runs      # outputs/ only
#   ./scripts/cluster_sync_pull.sh --only-artifacts # artifacts/ only
#   ./scripts/cluster_sync_pull.sh --all            # outputs/ + logs/ + artifacts/
#   ./scripts/cluster_sync_pull.sh --dry-run
# ---------------------------------------------------------------------------
set -euo pipefail

CLUSTER_HOST="${CLUSTER_HOST:-gpu}"
REMOTE_DIR="${REMOTE_DIR:-~/Face-Occlusion-Prediction}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODE="default"   # default = runs + logs
DRY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only-runs)      MODE="runs"; shift ;;
    --only-artifacts) MODE="artifacts"; shift ;;
    --all)            MODE="all"; shift ;;
    --dry-run)        DRY="--dry-run"; shift ;;
    -h|--help)        sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

declare -a DIRS
case "$MODE" in
  default)    DIRS=("outputs" "logs") ;;
  runs)       DIRS=("outputs") ;;
  artifacts)  DIRS=("artifacts") ;;
  all)        DIRS=("outputs" "logs" "artifacts") ;;
esac

for d in "${DIRS[@]}"; do
  echo "[pull] ${CLUSTER_HOST}:${REMOTE_DIR}/${d}/  →  ${LOCAL_DIR}/${d}/"
  mkdir -p "${LOCAL_DIR}/${d}"
  rsync -avzP $DRY \
    "${CLUSTER_HOST}:${REMOTE_DIR}/${d}/" \
    "${LOCAL_DIR}/${d}/" || echo "[pull] (skipped: remote ${d}/ missing or empty)"
done

echo "[pull] done."
