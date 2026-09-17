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
#   ./scripts/run_h100_full946_generation.sh RUN_NAME
#
# Required external inputs:
#   UNSTEP_PYENV        Python 3.12.14 env with PyTorch 2.15/CUDA 13.3
#   UNSTEP_CUDA         CUDA 13.3 prefix
#   UNSTEP_FA3          FA3 overlay path
#   UNSTEP_ASSETS_ROOT  assets directory containing SF source, weights, VAE, prompts
#
# Default clean sigma: 0.02 for both input re-noising and the emitted prediction.
#
# The launcher isolates run outputs by refusing to reuse an existing output
# directory unless UNSTEP_OVERWRITE_OUTPUT=1 is set. Runtime compiler caches are
# fresh per run by default; each shard then does an untimed first-prompt warmup
# before the measured full-shard generation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ "$#" -lt 1 ]; then
  echo "usage: $0 RUN_NAME" >&2
  exit 2
fi

RUN="$1"
NUM_SHARDS="${UNSTEP_NUM_SHARDS:-8}"
NUM_GPUS="${UNSTEP_NUM_GPUS:-2}"
CPU_THREADS_PER_SHARD="${UNSTEP_CPU_THREADS_PER_SHARD:-32}"
TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-32}"
UNSTEP_EXTRA_ARGS="${UNSTEP_EXTRA_ARGS:---clean-sigma-override 0.02}"
FULL_WARMUP_RUNS="${UNSTEP_FULL_WARMUP_RUNS:-1}"
FULL_WARMUP_PROMPT_INDICES="${UNSTEP_FULL_WARMUP_PROMPT_INDICES:-shard_first}"
FULL_WARMUP_PROCESS="${UNSTEP_FULL_WARMUP_PROCESS:-separate}"
CLEAR_GLOBAL_RUNTIME_CACHES="${UNSTEP_CLEAR_GLOBAL_RUNTIME_CACHES:-1}"

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
VIDEO_DIR="$OUT_DIR/indexed_videos"
GEN_JSON_DIR="$OUT_DIR/generation_jsons"
LOG_DIR="$OUT_DIR/generation_logs"
RUNTIME_STATE_DIR="$OUT_DIR/runtime_state"
CACHE_MODE="${UNSTEP_CACHE_MODE:-isolated}"
CACHE_SCOPE="${UNSTEP_CACHE_SCOPE:-run_gpu}"
SHARED_RUNTIME_STATE_DIR="${UNSTEP_SHARED_RUNTIME_STATE_DIR:-$REPO_ROOT/.runtime_state/h100_full946_generation}"
SHARED_WORK_ROOT="${UNSTEP_WORK_ROOT:-/tmp/unstep_h100_full946_generation}"

case "$CACHE_MODE" in
  shared|isolated) ;;
  *)
    echo "Unsupported UNSTEP_CACHE_MODE=$CACHE_MODE; expected shared or isolated." >&2
    exit 2
    ;;
esac
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

EXTRA_ARGS=()
if [ -n "$UNSTEP_EXTRA_ARGS" ]; then
  read -r -a EXTRA_ARGS <<< "$UNSTEP_EXTRA_ARGS"
fi

RUNNER="$SRC/unstep/run_wrapper_generation.py"
COMMON_ARGS=(
    --num-shards "$NUM_SHARDS"
    --shard-mode contiguous
    --seed-mode sequential_global
    --rng-skip-mode initial_only
    --fa3-import-backend h100
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
      CUDA_VISIBLE_DEVICES="$gpu" \
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
      TORCH_COMPILE_CACHE_KEY_TAG="unstep_h100_full946_shard${shard}" \
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
        --prompt-indices all \
        --save-videos \
        --video-dir "$VIDEO_DIR" \
        "${EXTRA_ARGS[@]}"
  } > "$LOG_DIR/shard${shard}.log" 2>&1
}

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
  for pid in "${pids[@]}"; do
    wait "$pid"
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
print(f"raw_mean_fps={sum(fps) / len(fps):.4f}")
print(f"healthy_ge30_fps={sum(healthy) / len(healthy):.4f}")
print(f"healthy_ge30_count={len(healthy)}/{len(fps)}")
PY

if [ "$CACHE_MODE" = "isolated" ] && [ "$CACHE_SCOPE" = "run_gpu" ]; then
  for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    gpu_work="/tmp/unstep_${RUN}_gpu${gpu}_work"
    if [ -d "$gpu_work" ]; then
      find "$gpu_work" -mindepth 1 -delete
      rmdir "$gpu_work" 2>/dev/null || true
    fi
  done
fi
