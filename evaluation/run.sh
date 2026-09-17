#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 RUN_NAME" >&2
  exit 2
fi

RUN="$1"
if [[ ! "$RUN" =~ ^[[:alnum:]][[:alnum:]_.-]*$ ]]; then
  echo "RUN_NAME must be a single name using letters, digits, dot, underscore or hyphen." >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=generation/path_safety.sh
source "$REPO_ROOT/generation/path_safety.sh"

PYENV="${UNSTEP_PYENV:-$REPO_ROOT/.venv}"
SRC="${UNSTEP_SRC:-$REPO_ROOT}"
CUDA_ROOT="${UNSTEP_CUDA:-${CUDA_HOME:-}}"
CUDA_OVERLAY="${UNSTEP_CUDA_OVERLAY:-}"
ASSETS_ROOT="${UNSTEP_ASSETS_ROOT:-$REPO_ROOT/assets}"
VBSRC="${UNSTEP_VBENCH_SRC:-$ASSETS_ROOT/source/VBench}"
VBCACHE="${UNSTEP_VBENCH_CACHE:-$ASSETS_ROOT/weights/vbench}"
VBSCORE="${UNSTEP_VBENCH_PYTHONPATH:-}"
EXTRA_PYTHONPATH="${UNSTEP_EXTRA_PYTHONPATH:-}"
OUT_DIR="${UNSTEP_OUT_DIR:-$REPO_ROOT/results/$RUN}"
OUT_DIR="$(unstep_safe_directory output "$OUT_DIR")"
VIDEO_DIR="${UNSTEP_VIDEO_DIR:-$OUT_DIR/indexed_videos}"
VIDEO_DIR="$(unstep_safe_directory video "$VIDEO_DIR")"
SCORE_DIR="$OUT_DIR/vbench_score"
LINK_DIR="$SCORE_DIR/standard_video_links"
LOG_DIR="$SCORE_DIR/logs"
DIM0_OUT="$SCORE_DIR/dim0"
DIM1_OUT="$SCORE_DIR/dim1"
DELETE_VIDEOS_AFTER_SCORE="${UNSTEP_DELETE_VIDEOS_AFTER_SCORE:-1}"

GPU_IDS=(0 1)
if [ "${CUDA_VISIBLE_DEVICES+x}" ]; then
  IFS=, read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
fi
if [ "${#GPU_IDS[@]}" -lt 2 ]; then
  echo "Evaluation requires two devices in CUDA_VISIBLE_DEVICES." >&2
  exit 2
fi
for gpu in "${GPU_IDS[@]}"; do
  if [[ ! "$gpu" =~ ^([0-9]+|GPU-[[:alnum:]-]+|MIG-[[:alnum:]/-]+)$ ]]; then
    echo "Invalid CUDA_VISIBLE_DEVICES entry: $gpu" >&2
    exit 2
  fi
done

if [ ! -x "$PYENV/bin/python" ]; then
  echo "Missing Python env: $PYENV/bin/python" >&2
  exit 2
fi
if [ -z "$CUDA_ROOT" ]; then
  echo "Missing CUDA root. Set CUDA_HOME or UNSTEP_CUDA." >&2
  exit 2
fi
if [ ! -d "$VIDEO_DIR" ]; then
  echo "Missing video dir: $VIDEO_DIR" >&2
  exit 2
fi

if [ -e "$SCORE_DIR" ] || [ -L "$SCORE_DIR" ]; then
  echo "Refusing to reuse score directory: $SCORE_DIR" >&2
  echo "Archive it explicitly before retrying; scoring never overwrites prior results." >&2
  exit 2
fi
mkdir -p "$OUT_DIR"
mkdir "$SCORE_DIR"
mkdir -p "$LINK_DIR" "$LOG_DIR" "$DIM0_OUT" "$DIM1_OUT"

PYTHONPATH_VALUE="$SRC:$VBSRC"
if [ -n "$VBSCORE" ]; then
  PYTHONPATH_VALUE="$PYTHONPATH_VALUE:$VBSCORE"
fi
if [ -n "$CUDA_OVERLAY" ]; then
  PYTHONPATH_VALUE="$PYTHONPATH_VALUE:$CUDA_OVERLAY"
fi
if [ -n "$EXTRA_PYTHONPATH" ]; then
  PYTHONPATH_VALUE="$PYTHONPATH_VALUE:$EXTRA_PYTHONPATH"
fi
LD_LIBRARY_PATH_VALUE="$CUDA_ROOT/lib64:$CUDA_ROOT/lib:${CUDA_OVERLAY:+$CUDA_OVERLAY/nvidia/cu13/lib:$CUDA_OVERLAY/nvidia/cudnn/lib:$CUDA_OVERLAY/nvidia/nccl/lib:$CUDA_OVERLAY/nvidia/cusparselt/lib:$CUDA_OVERLAY/nvidia/nvshmem/lib:$CUDA_OVERLAY/nvidia/cusparse/lib:$CUDA_OVERLAY/nvidia/cufft/lib:$CUDA_OVERLAY/nvidia/nvjitlink/lib:}$PYENV/lib:${LD_LIBRARY_PATH:-}"

"$PYENV/bin/python" - "$VBSRC/vbench/VBench_full_info.json" "$VIDEO_DIR" "$LINK_DIR" <<'PY'
import json
import os
import sys
from pathlib import Path

info = json.loads(Path(sys.argv[1]).read_text())
video_dir = Path(sys.argv[2]).resolve()
link_dir = Path(sys.argv[3]).resolve()
for old in link_dir.glob("*.mp4"):
    old.unlink()
missing = []
for idx, item in enumerate(info):
    src = video_dir / f"{idx}-0_ema.mp4"
    dst = link_dir / f"{item['prompt_en']}-0.mp4"
    if not src.exists():
        missing.append(str(src))
        continue
    if dst.exists() or dst.is_symlink():
        continue
    os.symlink(src, dst)
if missing:
    raise SystemExit(f"missing {len(missing)} generated videos; first missing: {missing[0]}")
print(f"linked {len(info)} VBench-standard video names into {link_dir}")
PY

DIM0=(
  subject_consistency
  background_consistency
  aesthetic_quality
  imaging_quality
  temporal_flickering
  motion_smoothness
  dynamic_degree
  overall_consistency
)
DIM1=(
  object_class
  multiple_objects
  color
  spatial_relationship
  scene
  temporal_style
  human_action
  appearance_style
)

pick_port() {
  "$PYENV/bin/python" - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

run_dim() {
  trap - EXIT INT TERM
  local gpu="$1"
  local out="$2"
  local port="$3"
  shift 3
  exec env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT="$port" \
    RANK=0 \
    LOCAL_RANK=0 \
    WORLD_SIZE=1 \
    SSH_CONNECTION="${SSH_CONNECTION:-}" \
    PYTHONNOUSERSITE=1 \
    TORCH_HOME="$VBCACHE/torch_hub" \
    VBENCH_CACHE_DIR="$VBCACHE" \
    HF_HOME="$VBCACHE" \
    HUGGINGFACE_HUB_CACHE="$VBCACHE" \
    TRANSFORMERS_CACHE="$VBCACHE" \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH_VALUE" \
    PYTHONPATH="$PYTHONPATH_VALUE" \
    PATH="$PYENV/bin:${PATH:-}" \
    "$PYENV/bin/python" "$SCRIPT_DIR/vbench_eval_entry.py" "$VBSRC/evaluate.py" \
      --videos_path "$LINK_DIR" \
      --dimension "$@" \
      --output_path "$out" \
      --full_json_dir "$VBSRC/vbench/VBench_full_info.json" \
      --load_ckpt_from_local True \
      --mode vbench_standard
}

PORT0="$(pick_port)"
PORT1="$(pick_port)"

# Keep each scoring worker and its children in a separately terminable group.
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
run_dim "${GPU_IDS[0]}" "$DIM0_OUT" "$PORT0" "${DIM0[@]}" > "$LOG_DIR/dim0.log" 2>&1 &
pid0="$!"
pids+=("$pid0")
run_dim "${GPU_IDS[1]}" "$DIM1_OUT" "$PORT1" "${DIM1[@]}" > "$LOG_DIR/dim1.log" 2>&1 &
pid1="$!"
pids+=("$pid1")
while [ "${#pids[@]}" -gt 0 ]; do
  for i in "${!pids[@]}"; do
    if ! kill -0 "${pids[i]}" 2>/dev/null; then
      wait "${pids[i]}"
      unset 'pids[i]'
    fi
  done
  if [ "${#pids[@]}" -gt 0 ]; then sleep 0.1; fi
done

"$PYENV/bin/python" "$SCRIPT_DIR/merge_vbench_scores.py" \
  "$RUN" \
  "$SCORE_DIR/${RUN}_table1_score.json" \
  "$SCORE_DIR/${RUN}_table1_score.md" \
  "$DIM0_OUT" "$DIM1_OUT"

if [ "$DELETE_VIDEOS_AFTER_SCORE" = "1" ]; then
  find "$VIDEO_DIR" -type f -name '*.mp4' -delete
fi
