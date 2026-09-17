#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 /path/to/detectron2/source" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DETECTRON2_SRC="$1"
PYENV="${UNSTEP_PYENV:-$REPO_ROOT/.venv}"
CUDA_ROOT="${UNSTEP_CUDA:-${CUDA_HOME:-}}"
CUDA_OVERLAY="${UNSTEP_CUDA_OVERLAY:-}"
TARGET="${UNSTEP_VBENCH_DEPS:-$REPO_ROOT/.vbench_deps}"
EXTRA_PYTHONPATH="${UNSTEP_EXTRA_PYTHONPATH:-}"

if [ ! -x "$PYENV/bin/python" ]; then
  echo "Missing Python env: $PYENV/bin/python" >&2
  exit 2
fi
if [ ! -d "$DETECTRON2_SRC" ]; then
  echo "Missing detectron2 source dir: $DETECTRON2_SRC" >&2
  exit 2
fi
if [ -z "$CUDA_ROOT" ]; then
  echo "Missing CUDA root. Set CUDA_HOME or UNSTEP_CUDA." >&2
  exit 2
fi

mkdir -p "$TARGET"

if [ ! -e "$CUDA_ROOT/lib/libcudart.so" ] && [ -e "$CUDA_ROOT/lib/libcudart.so.13" ]; then
  ln -s libcudart.so.13 "$CUDA_ROOT/lib/libcudart.so"
fi

LD_LIBRARY_PATH_VALUE="$CUDA_ROOT/lib64:$CUDA_ROOT/lib:${CUDA_OVERLAY:+$CUDA_OVERLAY/nvidia/cu13/lib:$CUDA_OVERLAY/nvidia/cudnn/lib:$CUDA_OVERLAY/nvidia/nccl/lib:$CUDA_OVERLAY/nvidia/cusparselt/lib:$CUDA_OVERLAY/nvidia/nvshmem/lib:$CUDA_OVERLAY/nvidia/cusparse/lib:$CUDA_OVERLAY/nvidia/cufft/lib:$CUDA_OVERLAY/nvidia/nvjitlink/lib:}$PYENV/lib:${LD_LIBRARY_PATH:-}"

# Do not include runtime/sitecustomize.py while building Detectron2. The
# generation runtime knobs intentionally import torch early, which can break
# setup.py metadata generation after a failed CUDA-library preload.
PYTHONPATH_VALUE="$TARGET"
if [ -n "$CUDA_OVERLAY" ]; then
  PYTHONPATH_VALUE="$PYTHONPATH_VALUE:$CUDA_OVERLAY"
fi
if [ -n "$EXTRA_PYTHONPATH" ]; then
  PYTHONPATH_VALUE="$PYTHONPATH_VALUE:$EXTRA_PYTHONPATH"
fi

env \
  PYTHONNOUSERSITE=1 \
  CUDA_HOME="$CUDA_ROOT" \
  PATH="$CUDA_ROOT/bin:$PYENV/bin:/usr/bin:/bin:${PATH:-}" \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH_VALUE" \
  LIBRARY_PATH="$CUDA_ROOT/lib64:$CUDA_ROOT/lib:${LIBRARY_PATH:-}" \
  C_INCLUDE_PATH="$CUDA_ROOT/include:${C_INCLUDE_PATH:-}" \
  CPLUS_INCLUDE_PATH="$CUDA_ROOT/include:${CPLUS_INCLUDE_PATH:-}" \
  PYTHONPATH="$PYTHONPATH_VALUE" \
  MAX_JOBS="${MAX_JOBS:-4}" \
  FORCE_CUDA=1 \
  TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}" \
  "$PYENV/bin/python" -m pip install \
    --target "$TARGET" \
    --no-deps \
    --no-build-isolation \
    "$DETECTRON2_SRC"

env \
  PYTHONNOUSERSITE=1 \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH_VALUE" \
  PYTHONPATH="$PYTHONPATH_VALUE" \
  "$PYENV/bin/python" - <<'PY'
import detectron2
print(detectron2.__file__)
PY
