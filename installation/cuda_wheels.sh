#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Source after installing requirements-cuda13.txt in the active virtualenv.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Activate the setup venv, then source installation/cuda_wheels.sh." >&2
  exit 2
fi

_unstep_cuda_wheels() {
  local site cuda cudnn library link library_paths directory
  if [[ -z "${VIRTUAL_ENV:-}" || ! -x "$VIRTUAL_ENV/bin/python" ]]; then
    echo "Activate the setup virtualenv before sourcing cuda_wheels.sh." >&2
    return 2
  fi
  site="$("$VIRTUAL_ENV/bin/python" -I -c 'import sysconfig; print(sysconfig.get_path("platlib"))')" || return
  cuda="$site/nvidia/cu13"
  cudnn="$site/nvidia/cudnn"
  for library in "$cuda/bin/nvcc" "$cuda/include/cuda.h" \
      "$cuda/include/curand.h" "$cuda/lib/libcurand.so.10" \
      "$cudnn/include/cudnn.h" "$cudnn/lib/libcudnn.so.9"; do
    if [[ ! -f "$library" ]]; then
      echo "Missing CUDA build prerequisite: $library" >&2
      return 2
    fi
  done

  # NVIDIA wheels supply versioned libraries; CMake/linkers also need .so names.
  for library in "$cuda"/lib/lib*.so.*; do
    [[ -f "$library" ]] || continue
    link="${library##*/}"
    link="$cuda/lib/${link%%.so.*}.so"
    if [[ ! -e "$link" && ! -L "$link" ]]; then
      ln -s "${library##*/}" "$link" || return
    fi
  done
  library_paths="$cuda/lib:$cudnn/lib"
  for directory in "$site"/nvidia/*/lib; do
    [[ -d "$directory" ]] || continue
    case "$directory" in "$cuda/lib"|"$cudnn/lib") continue ;; esac
    library_paths="$library_paths:$directory"
  done

  export UNSTEP_CUDA_OVERLAY="$site"
  export CUDA_HOME="$cuda" UNSTEP_CUDA="$cuda"
  export CUDNN_INCLUDE_DIR="$cudnn/include"
  export CUDNN_LIBRARY="$cudnn/lib/libcudnn.so.9"
  export PATH="$cuda/bin:$PATH"
  export LD_LIBRARY_PATH="$library_paths${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export CPLUS_INCLUDE_PATH="$cuda/include/cccl${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
  export C_INCLUDE_PATH="$cuda/include/cccl${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
}

_unstep_cuda_wheels
