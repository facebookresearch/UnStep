#!/usr/bin/env bash
set -euo pipefail

# Default one-H100 speed probe for the production configuration.
#
# Usage:
#   ./scripts/run_h100_speed_probe.sh RUN_NAME
#
# The default method setting matches the full-946 launcher:
#   --clean-sigma-override 0.02

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN="${1:-H100_SPEED_PROBE_$(date +%Y%m%d_%H%M%S)}"
PROMPT_INDICES="${PROMPT_INDICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}"
WARMUP_PROMPT_INDICES="${WARMUP_PROMPT_INDICES:-$PROMPT_INDICES}"
WARMUP_REPEATS="${WARMUP_REPEATS:-1}"
DISABLE_GC_DURING_MEASUREMENT="${DISABLE_GC_DURING_MEASUREMENT:-1}"
EXCLUDE_FINAL_OUTPUT_LAYOUT_TIMING="${EXCLUDE_FINAL_OUTPUT_LAYOUT_TIMING:-0}"
CPU_THREADS_PER_PROCESS="${UNSTEP_CPU_THREADS_PER_PROCESS:-32}"
TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-32}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
UNSTEP_EXTRA_ARGS="${UNSTEP_EXTRA_ARGS:---clean-sigma-override 0.02}"

PYENV="${UNSTEP_PYENV:-$REPO_ROOT/.venv}"
SRC="${UNSTEP_SRC:-$REPO_ROOT}"
FA3="${UNSTEP_FA3:-}"
EXTRA_PYTHONPATH="${UNSTEP_EXTRA_PYTHONPATH:-}"
CUDA_ROOT="${UNSTEP_CUDA:-${CUDA_HOME:-}}"
CUDA_OVERLAY="${UNSTEP_CUDA_OVERLAY:-}"
ASSETS_ROOT="${UNSTEP_ASSETS_ROOT:-$REPO_ROOT/assets}"
SITE="${UNSTEP_SITE:-$REPO_ROOT/runtime}"
TMP_ROOT="${TMPDIR:-/tmp}"
SOURCE_URI="${UNSTEP_SOURCE_URI:-$ASSETS_ROOT/source/Self-Forcing-main}"
WEIGHTS_URI="${UNSTEP_WEIGHTS_URI:-$ASSETS_ROOT/weights/self_forcing_weights.tar}"
VAE_URI="${UNSTEP_VAE_URI:-$ASSETS_ROOT/weights/Wan2.1_VAE.pth}"
PROMPT_URI="${UNSTEP_PROMPT_URI:-$ASSETS_ROOT/data/vbench_all_dimension_extended.txt}"

if [ ! -x "$PYENV/bin/python" ]; then
  echo "Missing Python env: $PYENV/bin/python" >&2
  echo "Set UNSTEP_PYENV or create .venv from SETUP_H100.md." >&2
  exit 2
fi

if [ -z "$CUDA_ROOT" ]; then
  echo "Missing CUDA root. Set CUDA_HOME or UNSTEP_CUDA to the CUDA 13.3 prefix." >&2
  exit 2
fi

OUT_DIR="${UNSTEP_OUT_DIR:-$REPO_ROOT/results/$RUN}"
RUN_STATE_ROOT="${UNSTEP_RUN_STATE_ROOT:-$OUT_DIR/runtime_state}"
CACHE="$RUN_STATE_ROOT/cache"
WORK_ROOT="${UNSTEP_WORK_ROOT:-$TMP_ROOT/unstep_h100_speed_probe_stage}"
OUT_JSON="$OUT_DIR/${RUN}.json"
LOG="$OUT_DIR/${RUN}.log"

if [ -e "$RUN_STATE_ROOT" ] && [ "${UNSTEP_REUSE_RUNTIME_STATE:-0}" != "1" ]; then
  echo "Refusing to reuse runtime state: $RUN_STATE_ROOT" >&2
  echo "Use a new RUN_NAME, delete that directory, or set UNSTEP_REUSE_RUNTIME_STATE=1." >&2
  exit 2
fi

if [ -e "$WORK_ROOT" ]; then
  find "$WORK_ROOT" -mindepth 1 -delete
fi

mkdir -p \
  "$OUT_DIR" \
  "$CACHE/torchinductor" \
  "$CACHE/triton" \
  "$CACHE/cuda" \
  "$CACHE/xdg" \
  "$CACHE/pycache" \
  "$CACHE/torch_home" \
  "$CACHE/tmp" \
  "$WORK_ROOT"

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

COMMON_ENV=(
  CUDA_VISIBLE_DEVICES="$GPU"
  OMP_NUM_THREADS="$CPU_THREADS_PER_PROCESS"
  OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_PROCESS"
  MKL_NUM_THREADS="$CPU_THREADS_PER_PROCESS"
  NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_PROCESS"
  VECLIB_MAXIMUM_THREADS="$CPU_THREADS_PER_PROCESS"
  MALLOC_ARENA_MAX=4
  PYTHONDONTWRITEBYTECODE=1
  PYTHONNOUSERSITE=1
  PYTHONPYCACHEPREFIX="$CACHE/pycache"
  TMPDIR="$CACHE/tmp"
  XDG_CACHE_HOME="$CACHE/xdg"
  TORCH_HOME="$CACHE/torch_home"
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH_VALUE"
  PYTHONPATH="$PYTHONPATH_VALUE"
  PATH="$(dirname "$FFMPEG_EXE"):$PYENV/bin:/usr/bin:/bin:${PATH:-}"
  IMAGEIO_FFMPEG_EXE="$FFMPEG_EXE"
  FFMPEG_BINARY="$FFMPEG_EXE"
  TORCHINDUCTOR_CACHE_DIR="$CACHE/torchinductor"
  TRITON_CACHE_DIR="$CACHE/triton"
  TRITON_HOME="$CACHE/triton"
  CUDA_CACHE_PATH="$CACHE/cuda"
  TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE=0
  TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE=0
  TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE=0
  TORCHINDUCTOR_COMPILE_THREADS="$TORCHINDUCTOR_COMPILE_THREADS"
  TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0
  TRITON_REMOTE_CACHE_ENABLE=0
  TORCH_COMPILE_CACHE_KEY_TAG="unstep_h100_speed_probe"
)

RUNNER="$SRC/unstep/run_wrapper_generation.py"
COMMON_ARGS=(
    --num-shards 1 \
    --shard-index 0 \
    --shard-mode contiguous \
    --seed-mode sequential_global \
    --rng-skip-mode initial_only \
    --speed-only \
    --fa3-import-backend h100 \
    --source-uri "$SOURCE_URI" \
    --weights-uri "$WEIGHTS_URI" \
    --vae-uri "$VAE_URI" \
    --prompt-uri "$PROMPT_URI"
)

EXTRA_ARGS=()
if [ -n "$UNSTEP_EXTRA_ARGS" ]; then
  read -r -a EXTRA_ARGS <<< "$UNSTEP_EXTRA_ARGS"
fi

if [ "$EXCLUDE_FINAL_OUTPUT_LAYOUT_TIMING" = "1" ]; then
  COMMON_ARGS+=(--exclude-final-output-layout-timing)
fi
if [ "$DISABLE_GC_DURING_MEASUREMENT" = "1" ]; then
  COMMON_ARGS+=(--disable-gc-during-measurement)
fi

for ((i = 0; i < WARMUP_REPEATS; i++)); do
  WARMUP_RUN="${RUN}_WARMUP${i}"
  REUSE_ARGS=()
  if [ "$i" -gt 0 ]; then
    REUSE_ARGS+=(--reuse-workdir)
  fi
  {
    echo "UNSTEP_RUN_STATE_ROOT=$RUN_STATE_ROOT"
    echo "UNSTEP_WORK_ROOT=$WORK_ROOT"
    echo "TMPDIR=$CACHE/tmp"
    echo "XDG_CACHE_HOME=$CACHE/xdg"
    echo "TORCHINDUCTOR_CACHE_DIR=$CACHE/torchinductor"
    echo "TRITON_CACHE_DIR=$CACHE/triton"
    echo "TRITON_HOME=$CACHE/triton"
    echo "CUDA_CACHE_PATH=$CACHE/cuda"
    echo "TORCH_COMPILE_CACHE_KEY_TAG=unstep_h100_speed_probe"
    env "${COMMON_ENV[@]}" \
      "$PYENV/bin/python" "$RUNNER" \
        --run-name "$RUN" \
        --prompt-indices "$WARMUP_PROMPT_INDICES" \
        --out-json "$OUT_DIR/${WARMUP_RUN}.json" \
        --workdir "$WORK_ROOT" \
        --workdir-name source \
        --keep-workdir \
        "${REUSE_ARGS[@]}" \
        "${COMMON_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
  } >"$OUT_DIR/${WARMUP_RUN}.log" 2>&1
done

MEASURE_REUSE_ARGS=()
if [ "$WARMUP_REPEATS" -gt 0 ]; then
  MEASURE_REUSE_ARGS+=(--reuse-workdir)
fi

{
  echo "UNSTEP_RUN_STATE_ROOT=$RUN_STATE_ROOT"
  echo "UNSTEP_WORK_ROOT=$WORK_ROOT"
  echo "TMPDIR=$CACHE/tmp"
  echo "XDG_CACHE_HOME=$CACHE/xdg"
  echo "TORCHINDUCTOR_CACHE_DIR=$CACHE/torchinductor"
  echo "TRITON_CACHE_DIR=$CACHE/triton"
  echo "TRITON_HOME=$CACHE/triton"
  echo "CUDA_CACHE_PATH=$CACHE/cuda"
  echo "TORCH_COMPILE_CACHE_KEY_TAG=unstep_h100_speed_probe"
  env "${COMMON_ENV[@]}" \
    "$PYENV/bin/python" "$RUNNER" \
      --run-name "$RUN" \
      --prompt-indices "$PROMPT_INDICES" \
      --out-json "$OUT_JSON" \
      --workdir "$WORK_ROOT" \
      --workdir-name source \
      "${MEASURE_REUSE_ARGS[@]}" \
      "${COMMON_ARGS[@]}" \
      "${EXTRA_ARGS[@]}"
} >"$LOG" 2>&1

"$PYENV/bin/python" - "$OUT_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
fps = [float(row["finite_video_fps"]) for row in data["measurements"]]
warm = fps[1:]
steady_tail = fps[-3:] if len(fps) >= 3 else fps
print(f"json={path}")
print("fps=" + ",".join(f"{x:.4f}" for x in fps))
print(f"warm_mean_excluding_first={sum(warm) / len(warm):.4f}")
print(f"min_after_first={min(warm):.4f}")
print(f"steady_tail_mean={sum(steady_tail) / len(steady_tail):.4f}")
print(f"raw_mean={sum(fps) / len(fps):.4f}")
print(f"single_transformer={data.get('config', {}).get('single_transformer')}")
print(f"separate_method_transformer={data.get('config', {}).get('separate_method_transformer')}")
print(f"clean_sigma_override={data.get('config', {}).get('clean_sigma_override')}")
print("runtime=" + json.dumps(data.get("runtime_info", {}), sort_keys=True))
PY
