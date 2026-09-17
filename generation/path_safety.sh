#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Sourced by the launchers before creating or clearing output/work directories.
unstep_safe_directory() {
  local kind="$1" target="$2" resolved protected base repo
  local -a dependency_dirs=()
  if [ -z "$target" ]; then
    echo "Refusing empty $kind directory." >&2
    return 2
  fi
  resolved="$(realpath -m -- "$target")" || return 2
  repo="$(realpath -m -- "$REPO_ROOT")" || return 2
  for protected in / "$HOME" /tmp /var/tmp "$repo" /opt /mnt /media /srv; do
    base="$(realpath -m -- "$protected")" || return 2
    if [[ "$resolved" == "$base" || "$base" == "$resolved/"* ]]; then
      echo "Refusing unsafe $kind directory: $resolved (contains $base)" >&2
      return 2
    fi
  done
  if [[ "$resolved" == "$repo/"* ]]; then
    case "$kind:$resolved" in
      output:"$repo/results/"*|video:"$repo/results/"*) ;;
      cache:"$repo/results/"*|cache:"$repo/.runtime_state/"*) ;;
      *) echo "Refusing $kind directory in the source tree: $resolved" >&2; return 2 ;;
    esac
  fi
  IFS=: read -r -a dependency_dirs <<< "${EXTRA_PYTHONPATH:-}:${VBSCORE:-}"
  for protected in /usr /etc /bin /sbin /lib /lib64 /dev /proc /sys /boot /run \
    "${ASSETS_ROOT:-}" "${SOURCE_URI:-}" "${WEIGHTS_URI:-}" "${CHECKPOINT_URI:-}" \
    "${VAE_URI:-}" "${PROMPT_URI:-}" "${SITE:-}" "${VBSRC:-}" "${VBCACHE:-}" \
    "${PYENV:-}" "${CUDA_ROOT:-}" "${CUDA_OVERLAY:-}" "${FA3:-}" "${dependency_dirs[@]}"; do
    [ -n "$protected" ] || continue
    base="$(realpath -m -- "$protected")" || return 2
    if [[ "$resolved" == "$base" || "$resolved" == "$base/"* || "$base" == "$resolved/"* ]]; then
      echo "Refusing $kind directory overlapping protected input: $resolved ($base)" >&2
      return 2
    fi
  done
  base="$(realpath -m -- "${SRC:-$REPO_ROOT}")" || return 2
  if [[ "$base" != "$repo" ]] &&
     [[ "$resolved" == "$base" || "$resolved" == "$base/"* || "$base" == "$resolved/"* ]]; then
    echo "Refusing $kind directory overlapping UNSTEP_SRC: $resolved" >&2
    return 2
  fi
  if [ "$kind" = work ]; then
    for protected in "$OUT_DIR" "${RUN_STATE_ROOT:-}" "${SHARED_RUNTIME_STATE_DIR:-}"; do
      [ -n "$protected" ] || continue
      base="$(realpath -m -- "$protected")" || return 2
      if [[ "$resolved" == "$base" || "$resolved" == "$base/"* || "$base" == "$resolved/"* ]]; then
        echo "Refusing work directory overlapping outputs/caches: $resolved" >&2
        return 2
      fi
    done
  fi
  if [ "$kind" = cache ] && [ -n "${OUT_DIR:-}" ]; then
    base="$(realpath -m -- "$OUT_DIR")" || return 2
    if [[ "$resolved" == "$base" || "$base" == "$resolved/"* ]]; then
      echo "Refusing cache directory containing outputs: $resolved" >&2
      return 2
    fi
  fi
  printf '%s\n' "$resolved"
}
