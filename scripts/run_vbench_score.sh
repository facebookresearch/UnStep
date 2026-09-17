#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 RUN_NAME" >&2
  exit 2
fi

RUN="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

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
VIDEO_DIR="${UNSTEP_VIDEO_DIR:-$OUT_DIR/indexed_videos}"
SCORE_DIR="$OUT_DIR/vbench_score"
LINK_DIR="$SCORE_DIR/standard_video_links"
LOG_DIR="$SCORE_DIR/logs"
DIM0_OUT="$SCORE_DIR/dim0"
DIM1_OUT="$SCORE_DIR/dim1"
DELETE_VIDEOS_AFTER_SCORE="${UNSTEP_DELETE_VIDEOS_AFTER_SCORE:-1}"

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

if [ -e "$SCORE_DIR" ]; then
  find "$SCORE_DIR" -mindepth 1 -delete
fi
mkdir -p "$LINK_DIR" "$LOG_DIR" "$DIM0_OUT" "$DIM1_OUT"

PYTHONPATH_VALUE="$SRC:$VBSRC"
if [ -n "$CUDA_OVERLAY" ]; then
  PYTHONPATH_VALUE="$PYTHONPATH_VALUE:$CUDA_OVERLAY"
fi
if [ -n "$VBSCORE" ]; then
  PYTHONPATH_VALUE="$PYTHONPATH_VALUE:$VBSCORE"
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
  local gpu="$1"
  local out="$2"
  local port="$3"
  shift 3
  env \
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

run_dim 0 "$DIM0_OUT" "$PORT0" "${DIM0[@]}" > "$LOG_DIR/dim0.log" 2>&1 &
pid0="$!"
run_dim 1 "$DIM1_OUT" "$PORT1" "${DIM1[@]}" > "$LOG_DIR/dim1.log" 2>&1 &
pid1="$!"
wait "$pid0"
wait "$pid1"

"$PYENV/bin/python" "$SCRIPT_DIR/merge_vbench_scores.py" \
  "$RUN" \
  "$SCORE_DIR/${RUN}_table1_score.json" \
  "$SCORE_DIR/${RUN}_table1_score.md" \
  "$DIM0_OUT" "$DIM1_OUT"

if [ "$DELETE_VIDEOS_AFTER_SCORE" = "1" ]; then
  find "$VIDEO_DIR" -type f -name '*.mp4' -delete
fi
