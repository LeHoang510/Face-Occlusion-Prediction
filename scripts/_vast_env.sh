# Shared NVIDIA / CUDA paths for vast.ai PyTorch containers.
# Sourced by vast_setup.sh and vast_run.sh — do not execute directly.

export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

# Driver-native libcuda FIRST. NEVER prepend /usr/local/cuda*/compat:
# the "CUDA Forward Compatibility" package is restricted to data-center GPUs
# (A100, H100, ...). On consumer cards (RTX 4090, RTX 3090, GTX, GeForce) it
# triggers `cuda error 804: forward compatibility was attempted on non
# supported HW` and CUDA becomes unusable.
#
# Loop appends to the FRONT, so the last entry in this list ends up at the
# HIGHEST priority in LD_LIBRARY_PATH.
_vast_ld=""
for _libdir in \
  /opt/conda/lib \
  /usr/local/cuda/lib64 \
  /usr/local/cuda/targets/x86_64-linux/lib \
  /usr/local/cuda/extras/CUPTI/lib64 \
  /usr/local/nvidia/lib \
  /usr/local/nvidia/lib64 \
  /usr/lib/x86_64-linux-gnu; do
  if [[ -d "$_libdir" ]]; then
    _vast_ld="${_libdir}:${_vast_ld}"
  fi
done
if [[ -n "$_vast_ld" ]]; then
  export LD_LIBRARY_PATH="${_vast_ld}${LD_LIBRARY_PATH:-}"
fi

# Belt-and-suspenders: many vast.ai PyTorch images ship a
# /etc/ld.so.conf.d/000_cuda-compat.conf that pins ldconfig to the compat
# libcuda. Even with LD_LIBRARY_PATH set, downstream programs (torch loader,
# python C-extensions) sometimes still hit ldconfig first. On a consumer GPU
# this is fatal. We neuter the compat entry when we detect such a GPU.
_vast_gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)"
case "$_vast_gpu_name" in
  *RTX*|*GTX*|*GeForce*)
    if [[ "$(id -u 2>/dev/null)" == "0" ]]; then
      _vast_neutered=0
      shopt -s nullglob
      for _f in /etc/ld.so.conf.d/000_cuda-compat.conf /etc/ld.so.conf.d/*cuda*compat*.conf; do
        if [[ -f "$_f" ]]; then
          echo "[vast] disabling $_f (consumer GPU '$_vast_gpu_name' — forward compat not supported)"
          mv "$_f" "${_f}.disabled-by-vast-env"
          _vast_neutered=1
        fi
      done
      shopt -u nullglob
      if [[ "$_vast_neutered" == "1" ]]; then
        ldconfig 2>/dev/null || true
      fi
      unset _vast_neutered _f
    fi
    ;;
esac
unset _vast_gpu_name
if [[ -d /usr/local/cuda ]]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi
if [[ -d /opt/conda/bin ]]; then
  export PATH="/opt/conda/bin:${PATH}"
fi
unset _vast_ld _libdir

# Strip /opt/conda/lib from LD_LIBRARY_PATH (conda's libcurl breaks curl SSL).
vast_ld_without_conda() {
  local _ld=":${LD_LIBRARY_PATH:-}:"
  _ld="${_ld//:\/opt\/conda\/lib:/:}"
  _ld="${_ld#:}"
  _ld="${_ld%:}"
  printf '%s' "${_ld}"
}

# Some vast.ai hosts MITM HTTPS with a cert missing from the container trust store.
vast_ssl_probe() {
  local _ld
  _ld="$(vast_ld_without_conda)"
  LD_LIBRARY_PATH="${_ld}" /usr/bin/curl -fsS --max-time 15 https://pypi.org/simple/ >/dev/null 2>&1
}

vast_enable_insecure_ssl() {
  export VAST_INSECURE_SSL=1
  export UV_INSECURE_HOST="${UV_INSECURE_HOST:-pypi.org files.pythonhosted.org download.pytorch.org github.com raw.githubusercontent.com objects.githubusercontent.com huggingface.co cdn-lfs.huggingface.co astral.sh}"
}

vast_configure_ssl() {
  if [[ -n "${VAST_SSL_CONFIGURED:-}" ]]; then
    return 0
  fi
  VAST_SSL_CONFIGURED=1

  local ca="/etc/ssl/certs/ca-certificates.crt"
  if [[ -f "${ca}" ]]; then
    export SSL_CERT_FILE="${SSL_CERT_FILE:-${ca}}"
  fi
  export UV_NATIVE_TLS="${UV_NATIVE_TLS:-1}"

  if [[ "${VAST_INSECURE_SSL:-}" == "1" ]]; then
    echo "[vast] VAST_INSECURE_SSL=1 — skipping HTTPS cert verification for known hosts"
    vast_enable_insecure_ssl
    return 0
  fi
  if [[ "${VAST_INSECURE_SSL:-}" == "0" ]]; then
    return 0
  fi

  if ! vast_ssl_probe; then
    echo "[vast] HTTPS cert verify failing; enabling trusted-host fallbacks (export VAST_INSECURE_SSL=0 to disable)"
    vast_enable_insecure_ssl
  fi
}

vast_pip() {
  local py="${1:?python required}"
  shift
  local -a trusted=()
  if [[ "${VAST_INSECURE_SSL:-0}" == "1" ]]; then
    trusted=(
      --trusted-host pypi.org
      --trusted-host pypi.python.org
      --trusted-host files.pythonhosted.org
      --trusted-host download.pytorch.org
    )
  fi
  "${py}" -m pip "${trusted[@]}" "$@"
}

# Install uv without sudo. Handles broken SSL on some vast.ai networks.
vast_install_uv() {
  local py="${1:?python required}"
  local _ld dest tmpdir
  local -a curl_k=()
  _ld="$(vast_ld_without_conda)"
  dest="${HOME}/.local/bin"
  mkdir -p "${dest}"
  if [[ "${VAST_INSECURE_SSL:-0}" == "1" ]]; then
    curl_k=(-k)
  fi

  echo "[setup] installing uv"
  if [[ "${VAST_INSECURE_SSL:-0}" == "1" ]]; then
    echo "[setup] fetching uv binary from GitHub (insecure SSL mode)"
    tmpdir="$(mktemp -d)"
    if LD_LIBRARY_PATH="${_ld}" /usr/bin/curl "${curl_k[@]}" -L \
        "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz" \
        | tar xz -C "${tmpdir}" 2>/dev/null \
        && install -m 755 "${tmpdir}"/uv-x86_64-unknown-linux-gnu/uv "${dest}/uv"; then
      rm -rf "${tmpdir}"
      return 0
    fi
    rm -rf "${tmpdir}" 2>/dev/null || true
  elif LD_LIBRARY_PATH="${_ld}" /usr/bin/curl -LsSf https://astral.sh/uv/install.sh | sh; then
    return 0
  fi

  echo "[setup] install script failed; fetching uv binary from GitHub"
  tmpdir="$(mktemp -d)"
  if LD_LIBRARY_PATH="${_ld}" /usr/bin/curl "${curl_k[@]}" -L \
      "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz" \
      | tar xz -C "${tmpdir}" 2>/dev/null \
      && install -m 755 "${tmpdir}"/uv-x86_64-unknown-linux-gnu/uv "${dest}/uv"; then
    rm -rf "${tmpdir}"
    return 0
  fi
  rm -rf "${tmpdir}" 2>/dev/null || true

  echo "[setup] curl failed; falling back to pip"
  vast_pip "${py}" install --user uv
}

vast_configure_ssl

# Print torch/CUDA diagnostics for a given python executable.
# Exit 0 = CUDA OK, 1 = torch present but no CUDA, 2 = torch import failed.
vast_cuda_probe() {
  local py="${1:?python required}"
  "${py}" - <<'PY'
import ctypes.util
import os
import sys

print(f"[probe] python={sys.executable}")
print(f"[probe] LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}")
print(f"[probe] libcuda={ctypes.util.find_library('cuda')}")
try:
    import torch
except Exception as exc:
    print(f"[probe] import torch FAILED: {exc}")
    sys.exit(2)

print(f"[probe] torch={torch.__version__} cuda_build={torch.version.cuda}")
print(f"[probe] torch_file={torch.__file__}")
if torch.cuda.is_available():
    print(f"[probe] cuda_available=True device={torch.cuda.get_device_name(0)}")
    sys.exit(0)

print("[probe] cuda_available=False")
sys.exit(1)
PY
}

vast_cuda_diagnose() {
  echo "[diag] nvidia-smi:"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true
  echo "[diag] libcuda on system:"
  ldconfig -p 2>/dev/null | grep libcuda || true
  ls -la /usr/local/nvidia/lib64/libcuda.so* 2>/dev/null || true
  if [[ -x /opt/conda/bin/python ]]; then
    echo "[diag] pip nvidia-* in conda:"
    /opt/conda/bin/python -m pip list 2>/dev/null | grep -i '^nvidia-' || echo "(none)"
  fi
}

# Remove pip-shipped NVIDIA runtimes; they shadow the container driver (error 804).
vast_strip_pip_cuda_libs() {
  local py="${1:?python required}"
  local site_pkgs pip_pkgs

  site_pkgs="$("${py}" -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)"
  if [[ -n "${site_pkgs}" && -d "${site_pkgs}/nvidia" ]]; then
    echo "[vast] removing ${site_pkgs}/nvidia (pip CUDA libs)"
    rm -rf "${site_pkgs}/nvidia"
  fi

  pip_pkgs="$("${py}" -m pip list --format=freeze 2>/dev/null | sed -n 's/^\(nvidia-[^=]*\)==.*/\1/p' || true)"
  if [[ -n "${pip_pkgs}" ]]; then
    echo "[vast] pip uninstall nvidia-* from ${py}"
    # shellcheck disable=SC2086
    "${py}" -m pip uninstall -y ${pip_pkgs} >/dev/null 2>&1 || true
  fi
  "${py}" -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
}

# Install torch+CUDA wheels WITH their nvidia-* runtime deps.
#
# Rationale: the pytorch/* container only ships libcudart in /usr/local/cuda;
# libcublas, libcudnn, libcusolver, … are NOT there. torch's loader looks for
# them inside the `nvidia` site-packages tree (nvidia-cublas-cu12, etc.). So
# `--no-deps` fails immediately with "libcublas.so.*[0-9] not found" or
# "libcudnn.so.9: cannot open shared object file".
#
# The historical cuda error 804 was NOT caused by these wheels per se but by
# wheels whose CUDA major was too new for the driver (cu126 wheels on R535).
# cu124 wheels are R525+ compatible via /usr/local/cuda-12.4/compat/libcuda.so.1
# (now prepended to LD_LIBRARY_PATH above).
#
# Args: <python> [cuda_tag] [torch_ver] [tv_ver]
#   cuda_tag: cu124 (default — driver R525+ via /usr/local/cuda-12.4/compat)
#             cu118 (fallback — driver R470+; uses pip nvidia-*-cu11 wheels)
#             cu126 (driver R555+)
vast_install_torch() {
  local py="${1:?python required}"
  local tag="${2:-cu124}"
  local tver="${3:-2.4.1}"
  local tvver="${4:-0.19.1}"
  echo "[setup] installing torch==${tver}+${tag} torchvision==${tvver}+${tag} (with nvidia-* deps)"
  vast_pip "${py}" install --force-reinstall \
    "torch==${tver}+${tag}" \
    "torchvision==${tvver}+${tag}" \
    --index-url "https://download.pytorch.org/whl/${tag}"
}

# Soft uninstall: drop torch/torchvision but KEEP nvidia-* libs so the next
# torch reinstall can reuse them. The torch wheels alone are ~800 MB — no
# point re-downloading just to install identical nvidia deps each time.
vast_uninstall_torch() {
  local py="${1:?python required}"
  "${py}" -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
}

# Restore a working torch stack. Strategy: cu124 first (matches driver R535 via
# container compat libs), cu118 fallback (older drivers). Conda path removed —
# on the pytorch/* image, conda's `pytorch-cuda=12.4` metapackage often resolves
# without actually installing torch, leaving the env worse than before.
#
# We intentionally do NOT call vast_strip_pip_cuda_libs here: the nvidia-* libs
# are required by torch (see comment on vast_install_torch). Only torch itself
# is removed before the reinstall.
vast_repair_torch() {
  local py="${1:?python required}"

  echo "[setup] repairing torch CUDA stack"
  vast_uninstall_torch "${py}"

  if vast_install_torch "${py}" cu124 && vast_cuda_probe "${py}"; then
    return 0
  fi

  echo "[setup] cu124 failed, trying cu118 fallback"
  vast_uninstall_torch "${py}"
  # cu11 uses different nvidia-*-cu11 packages; force a clean strip of cu12 libs
  vast_strip_pip_cuda_libs "${py}"
  if vast_install_torch "${py}" cu118 && vast_cuda_probe "${py}"; then
    return 0
  fi

  vast_cuda_diagnose
  return 1
}

# Pick the first python on PATH whose torch sees CUDA. Prints path to stdout.
vast_pick_python() {
  local candidate
  for candidate in \
    "${VAST_PYTHON:-}" \
    "${PROJECT_ROOT:-.}/.venv/bin/python" \
    "/opt/conda/bin/python" \
    "$(command -v python3 2>/dev/null || true)" \
    "$(command -v python 2>/dev/null || true)"; do
    [[ -n "${candidate}" && -x "${candidate}" ]] || continue
    if vast_cuda_probe "${candidate}"; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}
