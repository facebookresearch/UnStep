#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

# Default production full-946 generation launcher.
#
# Usage:
#   ./generation/run.sh RUN_NAME
#
# Required external inputs:
#   UNSTEP_PYENV        Python 3.12.14 env with PyTorch 2.15/CUDA 13.3
#   UNSTEP_CUDA         CUDA 13.3 prefix
#   UNSTEP_FA3          optional Hopper FA3 overlay path
#   UNSTEP_ASSETS_ROOT  assets directory containing SF source, weights, VAE, prompts
#
# Default clean sigma: 0.02 for both input re-noising and the emitted prediction.
#
# The launcher isolates run outputs by refusing to reuse an existing output
# directory unless UNSTEP_OVERWRITE_OUTPUT=1 is set. Runtime compiler caches are
# fresh per run by default; each GPU warms its first assigned prompt in a
# separate process before measured generation, then reuses that staged source.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=generation/path_safety.sh
source "$SCRIPT_DIR/path_safety.sh"

if [ "$#" -lt 1 ]; then
  echo "usage: $0 RUN_NAME" >&2
  exit 2
fi

RUN="$1"
if [[ ! "$RUN" =~ ^[[:alnum:]][[:alnum:]_.-]*$ ]]; then
  echo "RUN_NAME must be a single name using letters, digits, dot, underscore or hyphen." >&2
  exit 2
fi
NUM_SHARDS="${UNSTEP_NUM_SHARDS:-8}"
NUM_GPUS="${UNSTEP_NUM_GPUS:-2}"
PROMPT_INDICES="${UNSTEP_PROMPT_INDICES:-all}"
ATTENTION_BACKEND="${UNSTEP_ATTENTION_BACKEND:-auto}"
CPU_THREADS_PER_SHARD="${UNSTEP_CPU_THREADS_PER_SHARD:-32}"
TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-32}"
UNSTEP_EXTRA_ARGS="${UNSTEP_EXTRA_ARGS:---clean-sigma-override 0.02}"
FULL_WARMUP_RUNS="${UNSTEP_FULL_WARMUP_RUNS:-1}"
FULL_WARMUP_PROMPT_INDICES="${UNSTEP_FULL_WARMUP_PROMPT_INDICES:-shard_first}"
FULL_WARMUP_PROCESS="${UNSTEP_FULL_WARMUP_PROCESS:-separate}"
CLEAR_GLOBAL_RUNTIME_CACHES="${UNSTEP_CLEAR_GLOBAL_RUNTIME_CACHES:-0}"

if [[ ! "$NUM_SHARDS" =~ ^[1-9][0-9]*$ || ! "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "UNSTEP_NUM_SHARDS and UNSTEP_NUM_GPUS must be positive integers." >&2
  exit 2
fi
case "$ATTENTION_BACKEND" in
  auto|fa2|h100|flash_attn_3) ;;
  *) echo "Unsupported UNSTEP_ATTENTION_BACKEND=$ATTENTION_BACKEND" >&2; exit 2 ;;
esac
case "$FULL_WARMUP_PROCESS" in
  separate|inline) ;;
  *) echo "UNSTEP_FULL_WARMUP_PROCESS must be separate or inline." >&2; exit 2 ;;
esac
if [[ ! "$FULL_WARMUP_RUNS" =~ ^(0|[1-9][0-9]*)$ ]] ||
   { [ "$FULL_WARMUP_PROCESS" = separate ] && [ "$FULL_WARMUP_RUNS" -gt 1 ]; }; then
  echo "Separate warmup supports 0 or 1 runs; inline supports non-negative counts." >&2
  exit 2
fi
if [ "$PROMPT_INDICES" != all ] && [ "$PROMPT_INDICES" != shard_first ] && [ "$NUM_SHARDS" != 1 ]; then
  echo "Explicit UNSTEP_PROMPT_INDICES subsets require UNSTEP_NUM_SHARDS=1; not the full VBench protocol." >&2
  exit 2
fi

GPU_IDS=()
if [ "${CUDA_VISIBLE_DEVICES+x}" ]; then
  IFS=, read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else
  for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do GPU_IDS+=("$gpu"); done
fi
if [ "${#GPU_IDS[@]}" -lt "$NUM_GPUS" ]; then
  echo "CUDA_VISIBLE_DEVICES must expose at least UNSTEP_NUM_GPUS=$NUM_GPUS devices." >&2
  exit 2
fi
for gpu in "${GPU_IDS[@]}"; do
  if [[ ! "$gpu" =~ ^([0-9]+|GPU-[[:alnum:]-]+|MIG-[[:alnum:]/-]+)$ ]]; then
    echo "Invalid CUDA_VISIBLE_DEVICES entry: $gpu" >&2
    exit 2
  fi
done

EXTRA_ARGS=()
CHECKPOINT_URI=""
read -r -a EXTRA_ARGS <<< "$UNSTEP_EXTRA_ARGS"
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  case "${EXTRA_ARGS[i]}" in
    --clean-sigma-override|--checkpoint-uri)
      if [ "$((i + 1))" -ge "${#EXTRA_ARGS[@]}" ] || [[ "${EXTRA_ARGS[i+1]}" == --* ]]; then
        echo "Missing value for ${EXTRA_ARGS[i]}" >&2; exit 2
      fi
      if [ "${EXTRA_ARGS[i]}" = --checkpoint-uri ]; then CHECKPOINT_URI="${EXTRA_ARGS[i+1]}"; fi
      i=$((i + 1)) ;;
    --checkpoint-uri=*) CHECKPOINT_URI="${EXTRA_ARGS[i]#*=}" ;;
    --clean-sigma-override=*|--disable-block-schedule-overrides) ;;
    *) echo "Unsupported or launcher-owned UNSTEP_EXTRA_ARGS option: ${EXTRA_ARGS[i]}" >&2; exit 2 ;;
  esac
done

PYENV="${UNSTEP_PYENV:-$REPO_ROOT/.venv}"
SRC="${UNSTEP_SRC:-$REPO_ROOT}"
FA3="${UNSTEP_FA3:-}"
EXTRA_PYTHONPATH="${UNSTEP_EXTRA_PYTHONPATH:-}"
CUDA_ROOT="${UNSTEP_CUDA:-${CUDA_HOME:-}}"
CUDA_OVERLAY="${UNSTEP_CUDA_OVERLAY:-}"
ASSETS_ROOT="${UNSTEP_ASSETS_ROOT:-$REPO_ROOT/assets}"
SITE="${UNSTEP_SITE:-$REPO_ROOT/runtime}"
SOURCE_URI="${UNSTEP_SOURCE_URI:-$ASSETS_ROOT/source/Self-Forcing-main}"
WEIGHTS_URI="${UNSTEP_WEIGHTS_URI:-$ASSETS_ROOT/weights/self_forcing_weights.tar}"
VAE_URI="${UNSTEP_VAE_URI:-$ASSETS_ROOT/weights/Wan2.1_VAE.pth}"
PROMPT_URI="${UNSTEP_PROMPT_URI:-$ASSETS_ROOT/data/vbench_all_dimension_extended.txt}"

if [ ! -x "$PYENV/bin/python" ]; then
  echo "Missing Python env: $PYENV/bin/python" >&2
  exit 2
fi

if [ -z "$CUDA_ROOT" ]; then
  echo "Missing CUDA root. Set CUDA_HOME or UNSTEP_CUDA to the CUDA 13.3 prefix." >&2
  exit 2
fi

OUT_DIR="${UNSTEP_OUT_DIR:-$REPO_ROOT/results/$RUN}"
OUT_DIR="$(unstep_safe_directory output "$OUT_DIR")"
VIDEO_DIR="$OUT_DIR/indexed_videos"
GEN_JSON_DIR="$OUT_DIR/generation_jsons"
LOG_DIR="$OUT_DIR/generation_logs"
RUNTIME_STATE_DIR="$OUT_DIR/runtime_state"
CACHE_MODE="${UNSTEP_CACHE_MODE:-isolated}"
CACHE_SCOPE="${UNSTEP_CACHE_SCOPE:-run_gpu}"
SHARED_RUNTIME_STATE_DIR="${UNSTEP_SHARED_RUNTIME_STATE_DIR:-$REPO_ROOT/.runtime_state/full946_generation}"
SHARED_WORK_ROOT="${UNSTEP_WORK_ROOT:-/tmp/unstep_full946_generation}"

case "$CACHE_MODE" in
  shared|isolated) ;;
  *)
    echo "Unsupported UNSTEP_CACHE_MODE=$CACHE_MODE; expected shared or isolated." >&2
    exit 2
    ;;
esac

if [ "$CACHE_MODE" = shared ]; then
  SHARED_RUNTIME_STATE_DIR="$(unstep_safe_directory cache "$SHARED_RUNTIME_STATE_DIR")"
  SHARED_WORK_ROOT="$(unstep_safe_directory work "$SHARED_WORK_ROOT")"
fi
case "$CACHE_SCOPE" in
  run_gpu|shard) ;;
  *)
    echo "Unsupported UNSTEP_CACHE_SCOPE=$CACHE_SCOPE; expected run_gpu or shard." >&2
    exit 2
    ;;
esac

if [ -e "$OUT_DIR" ] && [ "${UNSTEP_OVERWRITE_OUTPUT:-0}" != "1" ]; then
  echo "Refusing to reuse output directory: $OUT_DIR" >&2
  echo "Use a new RUN_NAME, delete the directory, or set UNSTEP_OVERWRITE_OUTPUT=1." >&2
  exit 2
fi

if [ -e "$OUT_DIR" ]; then
  find "$OUT_DIR" -mindepth 1 -delete
fi
mkdir -p "$VIDEO_DIR" "$GEN_JSON_DIR" "$LOG_DIR" "$RUNTIME_STATE_DIR"

if [ "$CLEAR_GLOBAL_RUNTIME_CACHES" = "1" ]; then
  CACHE_USER="${USER:-$(id -un)}"
  for stale_cache in \
    "/tmp/torchinductor_${CACHE_USER}" \
    "/var/tmp/torchinductor_${CACHE_USER}" \
    "$HOME/.triton" \
    "$HOME/.cache/triton" \
    "$HOME/.cache/torch/inductor" \
    "$HOME/.nv/ComputeCache"; do
    if [ -d "$stale_cache" ]; then
      unstep_safe_directory cache "$stale_cache" >/dev/null
      find "$stale_cache" -mindepth 1 -delete
      rmdir "$stale_cache" 2>/dev/null || true
    fi
  done
fi

PYTHONPATH_VALUE="$SITE:$SRC"
if [ -n "$FA3" ]; then
  PYTHONPATH_VALUE="$PYTHONPATH_VALUE:$FA3"
fi
if [ -n "$CUDA_OVERLAY" ]; then
  PYTHONPATH_VALUE="$PYTHONPATH_VALUE:$CUDA_OVERLAY"
fi
if [ -n "$EXTRA_PYTHONPATH" ]; then
  PYTHONPATH_VALUE="$PYTHONPATH_VALUE:$EXTRA_PYTHONPATH"
fi

LD_LIBRARY_PATH_VALUE="$CUDA_ROOT/lib64:$CUDA_ROOT/lib:${CUDA_OVERLAY:+$CUDA_OVERLAY/nvidia/cu13/lib:$CUDA_OVERLAY/nvidia/cudnn/lib:$CUDA_OVERLAY/nvidia/nccl/lib:$CUDA_OVERLAY/nvidia/cusparselt/lib:$CUDA_OVERLAY/nvidia/nvshmem/lib:$CUDA_OVERLAY/nvidia/cusparse/lib:$CUDA_OVERLAY/nvidia/cufft/lib:$CUDA_OVERLAY/nvidia/nvjitlink/lib:}$PYENV/lib:${LD_LIBRARY_PATH:-}"
if [ -n "${UNSTEP_FFMPEG:-}" ]; then
  FFMPEG_EXE="$UNSTEP_FFMPEG"
else
  FFMPEG_EXE="$(
    env \
      PYTHONNOUSERSITE=1 \
      LD_LIBRARY_PATH="$LD_LIBRARY_PATH_VALUE" \
      PYTHONPATH="$PYTHONPATH_VALUE" \
      "$PYENV/bin/python" - <<'PY'
import imageio_ffmpeg
print(imageio_ffmpeg.get_ffmpeg_exe())
PY
  )"
fi
if [ ! -x "$FFMPEG_EXE" ]; then
  echo "Missing executable ffmpeg. Set UNSTEP_FFMPEG or install imageio-ffmpeg in the env." >&2
  exit 2
fi

RUNNER="$SRC/unstep/run_wrapper_generation.py"
COMMON_ARGS=(
    --num-shards "$NUM_SHARDS"
    --shard-mode contiguous
    --seed-mode sequential_global
    --rng-skip-mode initial_only
    --fa3-import-backend "$ATTENTION_BACKEND"
    --source-uri "$SOURCE_URI"
    --weights-uri "$WEIGHTS_URI"
    --vae-uri "$VAE_URI"
    --prompt-uri "$PROMPT_URI"
    --disable-gc-during-measurement
)

if [ "$FULL_WARMUP_RUNS" != "0" ] && [ "$FULL_WARMUP_PROCESS" = "inline" ]; then
  COMMON_ARGS+=(
    --warmup-runs "$FULL_WARMUP_RUNS"
    --warmup-prompt-indices "$FULL_WARMUP_PROMPT_INDICES"
  )
fi

launch_shard() {
  trap - EXIT INT TERM
  local shard="$1"
  local gpu="$2"
  local shard_name="${RUN}_shard${shard}"
  local shard_work
  local cache
  if [ "$CACHE_MODE" = "shared" ]; then
    shard_work="$SHARED_WORK_ROOT/shard${shard}"
    cache="$SHARED_RUNTIME_STATE_DIR/shard${shard}"
  elif [ "$CACHE_SCOPE" = "run_gpu" ]; then
    shard_work="/tmp/unstep_${RUN}_gpu${gpu}_work"
    cache="$RUNTIME_STATE_DIR/gpu${gpu}"
  else
    shard_work="/tmp/unstep_${RUN}_shard${shard}_work"
    cache="$RUNTIME_STATE_DIR/shard${shard}"
  fi
  cache="$(unstep_safe_directory cache "$cache")"
  shard_work="$(unstep_safe_directory work "$shard_work")"
  if [ -e "$shard_work" ] && [ ! -f "$cache/warmup.done" ]; then
    find "$shard_work" -mindepth 1 -delete
  fi
  mkdir -p \
    "$shard_work" \
    "$cache/tmp" \
    "$cache/xdg" \
    "$cache/torch_home" \
    "$cache/pycache" \
    "$cache/torchinductor" \
    "$cache/triton" \
    "$cache/cuda"
  {
    echo "production_cache_mode=$CACHE_MODE"
    echo "production_cache_scope=$CACHE_SCOPE"
    echo "full_warmup_runs=$FULL_WARMUP_RUNS"
    echo "full_warmup_prompt_indices=$FULL_WARMUP_PROMPT_INDICES"
    echo "full_warmup_process=$FULL_WARMUP_PROCESS"
    echo "clear_global_runtime_caches=$CLEAR_GLOBAL_RUNTIME_CACHES"
    echo "workdir=$shard_work"
    echo "TMPDIR=$cache/tmp"
    echo "XDG_CACHE_HOME=$cache/xdg"
    echo "TORCHINDUCTOR_CACHE_DIR=$cache/torchinductor"
    echo "TRITON_CACHE_DIR=$cache/triton"
    echo "TRITON_HOME=$cache/triton"
    echo "CUDA_CACHE_PATH=$cache/cuda"
    common_env=(
      CUDA_VISIBLE_DEVICES="${GPU_IDS[gpu]}" \
      OMP_NUM_THREADS="$CPU_THREADS_PER_SHARD" \
      OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_SHARD" \
      MKL_NUM_THREADS="$CPU_THREADS_PER_SHARD" \
      NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_SHARD" \
      VECLIB_MAXIMUM_THREADS="$CPU_THREADS_PER_SHARD" \
      MALLOC_ARENA_MAX=4 \
      PYTHONDONTWRITEBYTECODE=1 \
      PYTHONNOUSERSITE=1 \
      PYTHONPYCACHEPREFIX="$cache/pycache" \
      TMPDIR="$cache/tmp" \
      XDG_CACHE_HOME="$cache/xdg" \
      TORCH_HOME="$cache/torch_home" \
      LD_LIBRARY_PATH="$LD_LIBRARY_PATH_VALUE" \
      PYTHONPATH="$PYTHONPATH_VALUE" \
      PATH="$(dirname "$FFMPEG_EXE"):$PYENV/bin:/usr/bin:/bin:${PATH:-}" \
      IMAGEIO_FFMPEG_EXE="$FFMPEG_EXE" \
      FFMPEG_BINARY="$FFMPEG_EXE" \
      TORCHINDUCTOR_CACHE_DIR="$cache/torchinductor" \
      TRITON_CACHE_DIR="$cache/triton" \
      TRITON_HOME="$cache/triton" \
      CUDA_CACHE_PATH="$cache/cuda" \
      TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE=0 \
      TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE=0 \
      TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE=0 \
      TRITON_REMOTE_CACHE_ENABLE=0 \
      TORCH_COMPILE_CACHE_KEY_TAG="unstep_full946_shard${shard}" \
      TORCHINDUCTOR_COMPILE_THREADS="$TORCHINDUCTOR_COMPILE_THREADS" \
      TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 \
    )
    if [ "$FULL_WARMUP_RUNS" != "0" ] && [ "$FULL_WARMUP_PROCESS" = "separate" ] && [ ! -f "$cache/warmup.done" ]; then
      env "${common_env[@]}" \
        "$PYENV/bin/python" "$RUNNER" \
          --run-name "${shard_name}_warmup" \
          --shard-index "$shard" \
          --out-json "$GEN_JSON_DIR/shard${shard}.warmup.json" \
          --workdir "$shard_work" \
          --workdir-name source \
          --keep-workdir \
          "${COMMON_ARGS[@]}" \
          --prompt-indices "$FULL_WARMUP_PROMPT_INDICES" \
          --speed-only \
          "${EXTRA_ARGS[@]}"
      touch "$cache/warmup.done"
    fi
    measure_reuse_args=()
    if [ "$FULL_WARMUP_RUNS" != "0" ] && [ "$FULL_WARMUP_PROCESS" = "separate" ]; then
      measure_reuse_args+=(--reuse-workdir --workdir-name source --keep-workdir)
    fi
    env "${common_env[@]}" \
      "$PYENV/bin/python" "$RUNNER" \
        --run-name "$shard_name" \
        --shard-index "$shard" \
        --out-json "$GEN_JSON_DIR/shard${shard}.json" \
        --workdir "$shard_work" \
        "${measure_reuse_args[@]}" \
        "${COMMON_ARGS[@]}" \
        --prompt-indices "$PROMPT_INDICES" \
        --save-videos \
        --video-dir "$VIDEO_DIR" \
        "${EXTRA_ARGS[@]}"
  } > "$LOG_DIR/shard${shard}.log" 2>&1
}

# Job control gives each worker and its compiler children a separate process group.
set -m
pids=()
cleanup_workers() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  if [ "${#pids[@]}" -gt 0 ]; then sleep 1; fi
  for pid in "${pids[@]}"; do kill -KILL -- "-$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  exit "$status"
}
trap cleanup_workers EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for ((start = 0; start < NUM_SHARDS; start += NUM_GPUS)); do
  pids=()
  for ((offset = 0; offset < NUM_GPUS; offset++)); do
    shard=$((start + offset))
    if [ "$shard" -ge "$NUM_SHARDS" ]; then
      continue
    fi
    gpu=$((offset % NUM_GPUS))
    launch_shard "$shard" "$gpu" &
    pids+=("$!")
  done
  while [ "${#pids[@]}" -gt 0 ]; do
    for i in "${!pids[@]}"; do
      if ! kill -0 "${pids[i]}" 2>/dev/null; then
        wait "${pids[i]}"
        unset 'pids[i]'
      fi
    done
    if [ "${#pids[@]}" -gt 0 ]; then sleep 0.1; fi
  done
done

"$PYENV/bin/python" - "$GEN_JSON_DIR" "$VIDEO_DIR" <<'PY'
import json
import sys
from pathlib import Path

json_dir = Path(sys.argv[1])
video_dir = Path(sys.argv[2])
rows = []
for path in sorted(json_dir.glob("shard*.json")):
    if path.name.endswith(".warmup.json"):
        continue
    data = json.loads(path.read_text())
    rows.extend(data["measurements"])
fps = [float(row["finite_video_fps"]) for row in rows]
healthy = [value for value in fps if value >= 30.0]
print(f"generation_json_dir={json_dir}")
print(f"video_dir={video_dir}")
print(f"videos={len(list(video_dir.glob('*.mp4')))}")
print(f"measurements={len(fps)}")
print("raw_mean_fps=" + (f"{sum(fps) / len(fps):.4f}" if fps else "n/a"))
print("healthy_ge30_fps=" + (f"{sum(healthy) / len(healthy):.4f}" if healthy else "n/a"))
print(f"healthy_ge30_count={len(healthy)}/{len(fps)}")
PY

if [ "$CACHE_MODE" = "isolated" ] && [ "$CACHE_SCOPE" = "run_gpu" ]; then
  for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    gpu_work="/tmp/unstep_${RUN}_gpu${gpu}_work"
    gpu_work="$(unstep_safe_directory work "$gpu_work")"
    if [ -d "$gpu_work" ]; then
      find "$gpu_work" -mindepth 1 -delete
      rmdir "$gpu_work" 2>/dev/null || true
    fi
  done
fi
