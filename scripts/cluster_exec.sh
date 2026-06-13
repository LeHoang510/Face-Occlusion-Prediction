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
  # Thin wrapper around the vastai CLI with sensible presets for this project.
  #
  # Subcommands (under `vast`):
  #   offers [gpu]    search for cheap offers (default: RTX_4090, 1 GPU, dph<0.6)
  #   list            show your running instances
  #   launch <ask>    create instance from an ask id, PyTorch image, 80G disk
  #   ssh-url <iid>   print SSH url for an instance
  #   destroy <iid>   destroy an instance (with confirmation)
  #   <other>         passthrough to `vastai <other> ...`
  if ! command -v vastai >/dev/null 2>&1; then
    cat <<EOF >&2
[vast] vastai CLI not found. Install it once:
[vast]   pipx install vastai      # or: uv tool install vastai
[vast]   vastai set api-key <YOUR_KEY>
EOF
    exit 2
  fi

  local vsub="${1:-}"; shift || true
  case "$vsub" in
    offers)
      # cuda_max_good is the max CUDA version the host driver supports.
      # We require >=12.6 so the default PyPI torch wheel (cu126/cu128) works
      # out of the box and we don't end up in CPU fallback like on driver 12.4.
      local gpu="${1:-RTX_4090}"
      vastai search offers \
        "gpu_name=${gpu} num_gpus=1 dph<0.6 reliability>0.97 inet_down>200 cuda_max_good>=12.6" \
        --order "dph"
      ;;
    list)
      vastai show instances
      ;;
    launch)
      # Usage: vast launch <ask-id> [--image <img>] [--disk <GB>]
      # VAST_IMAGE env var overrides --image. Defaults to torch 2.6 + CUDA 12.6
      # (no 2.6.0-cuda12.8 tag on Docker Hub; CUDA 12.8 images start at torch 2.7+).
      # Driver filter cuda_max_good>=12.6 still matches hosts that run this image.
      local ask="${1:?vast launch: need an ask/offer id}"; shift || true
      local image="${VAST_IMAGE:-pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel}"
      local disk="80"
      while [[ $# -gt 0 ]]; do
        case "$1" in
          --image) image="$2"; shift 2 ;;
          --disk)  disk="$2";  shift 2 ;;
          *) echo "vast launch: unknown arg '$1'" >&2; exit 1 ;;
        esac
      done
      echo "[vast] launch ask=$ask image=$image disk=${disk}G"
      vastai create instance "$ask" \
        --image "$image" \
        --disk "$disk" \
        --onstart-cmd "touch /workspace/.vast-ready"
      ;;
    ssh-url)
      local iid="${1:?vast ssh-url: need an instance id}"
      vastai ssh-url "$iid"
      ;;
    destroy)
      local iid="${1:?vast destroy: need an instance id}"
      read -rp "[vast] really destroy instance $iid? [y/N] " ans
      [[ "$ans" =~ ^[yY]$ ]] || { echo "[vast] aborted."; exit 0; }
      vastai destroy instance "$iid"
      ;;
    ""|-h|--help)
      sed -n '/^cmd_vast()/,/^  esac$/p' "$0" | head -n 20
      ;;
    *)
      # Passthrough so anything not listed still works.
      vastai "$vsub" "$@"
      ;;
  esac
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
