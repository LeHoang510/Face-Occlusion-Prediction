#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# cluster_exec.sh — local entry point to drive cluster jobs
#
# Subcommands:
#   submit <config> [--script train|eval] [--export K=V] [--export K=V] ...
#                          sync .env then sbatch a SLURM job
#   status [job-id]        squeue (no arg) or sacct -j <id>
#   logs <job-id> [-f]     show the SLURM stdout (-f to follow)
#   cancel <job-id>        scancel
#   shell                  ssh -t into the cluster, cd'd into the repo
#   cmd <command...>       run an arbitrary command on the cluster, in the repo dir
#   sync-env               push the local .env to the cluster
#   vast <command...>      delegate to vast.ai (placeholder, not wired)
#
# Env overrides:
#   CLUSTER_HOST   default: gpu
#   REMOTE_DIR     default: ~/Face-Occlusion-Prediction
#
# Example:
#   ./scripts/cluster_exec.sh submit src/data_challenge/configs/dinov3_l_lora.yaml
#   ./scripts/cluster_exec.sh status
#   ./scripts/cluster_exec.sh logs 12345 -f
#   ./scripts/cluster_exec.sh cancel 12345
#   ./scripts/cluster_exec.sh cmd "nvidia-smi"
# ---------------------------------------------------------------------------
set -euo pipefail

CLUSTER_HOST="${CLUSTER_HOST:-gpu}"
REMOTE_DIR="${REMOTE_DIR:-~/Face-Occlusion-Prediction}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() { sed -n '2,28p' "$0"; }

run_remote() {
  ssh "$CLUSTER_HOST" "cd ${REMOTE_DIR} && $*"
}

sync_env() {
  if [[ -f "${LOCAL_DIR}/.env" ]]; then
    echo "[sync-env] pushing .env → ${CLUSTER_HOST}:${REMOTE_DIR}/.env"
    scp -q "${LOCAL_DIR}/.env" "${CLUSTER_HOST}:${REMOTE_DIR}/.env"
  else
    echo "[sync-env] no local .env, skipping."
  fi
}

cmd_submit() {
  local config="" script="train"
  local -a exports=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --script) script="$2"; shift 2 ;;
      --export) exports+=("$2"); shift 2 ;;
      -h|--help) usage; exit 0 ;;
      *) config="$1"; shift ;;
    esac
  done
  [[ -n "$config" ]] || { echo "submit: need a config path" >&2; exit 1; }

  sync_env

  # Build --export string: CONFIG plus user-supplied KEY=VAL
  local export_str="CONFIG=${config}"
  for kv in "${exports[@]}"; do export_str+=",${kv}"; done

  echo "[submit] host=${CLUSTER_HOST} script=scripts/slurm/${script}.sh"
  echo "[submit] export=${export_str}"
  run_remote "mkdir -p logs/slurm && sbatch --export=ALL,${export_str} scripts/slurm/${script}.sh"
}

cmd_status() {
  if [[ $# -gt 0 ]]; then
    run_remote "sacct -j $1 --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode -P"
  else
    run_remote "squeue -u \$USER -o '%.10i %.12P %.20j %.8u %.2t %.10M %.6D %R'"
  fi
}

cmd_logs() {
  local jid="${1:?logs: job id required}"; shift || true
  local follow="no"
  [[ "${1:-}" == "-f" || "${1:-}" == "--follow" ]] && follow="yes"
  if [[ "$follow" == "yes" ]]; then
    run_remote "f=\$(ls logs/slurm/*-${jid}.out 2>/dev/null | head -1); \
                if [ -n \"\$f\" ]; then echo \"[logs] tailing \$f\"; tail -f \"\$f\"; \
                else echo '[logs] no file yet for job ${jid}'; fi"
  else
    run_remote "f=\$(ls logs/slurm/*-${jid}.out 2>/dev/null | head -1); \
                if [ -n \"\$f\" ]; then echo \"[logs] \$f\"; tail -n 200 \"\$f\"; \
                else echo '[logs] no file for job ${jid}'; fi"
  fi
}

cmd_cancel() {
  local jid="${1:?cancel: job id required}"
  run_remote "scancel ${jid}"
  echo "[cancel] job ${jid} cancelled."
}

cmd_shell() {
  ssh -t "$CLUSTER_HOST" "cd ${REMOTE_DIR} && exec \${SHELL:-bash} -l"
}

cmd_cmd() {
  [[ $# -gt 0 ]] || { echo "cmd: need a command to run" >&2; exit 1; }
  run_remote "$*"
}

cmd_vast() {
  cat <<EOF >&2
[vast] vast.ai delegation is not wired up yet.
[vast] To plug it in, install \`vastai\` CLI and call e.g.:
[vast]   vastai search offers 'gpu_name=RTX_4090 reliability>0.95 num_gpus=1'
[vast]   vastai create instance <offer-id> --image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel --disk 80
[vast]   vastai ssh-url <instance-id>
[vast] Forwarded args: $*
EOF
  exit 2
}

sub="${1:-}"; shift || true
case "$sub" in
  submit)        cmd_submit "$@" ;;
  status)        cmd_status "$@" ;;
  logs)          cmd_logs "$@" ;;
  cancel)        cmd_cancel "$@" ;;
  shell)         cmd_shell ;;
  cmd)           cmd_cmd "$@" ;;
  sync-env)      sync_env ;;
  vast)          cmd_vast "$@" ;;
  ""|-h|--help)  usage ;;
  *) echo "unknown subcommand: ${sub}" >&2; usage; exit 1 ;;
esac
