# H100 Generation Environment

This is the H100 environment recipe for the 50+ FPS H100 generation path. It
only covers dependencies and launch isolation; method settings are defined by
the wrapper code.

## Exact Stack

- Python 3.12.14.
- PyTorch from `https://github.com/pytorch/pytorch` commit
  `65cd5f09ea45cf4c7e55821b840e2bbaf45a7ec5`.
- Torch runtime reports CUDA `13.3`.
- cuDNN runtime reports `9.25.1`.
- Triton `3.8.0+gitdf3f91dd`.
- torchvision `0.30.0.dev20260906+cu134`.
- FA3 overlay providing `flash_attn_interface` and `flash_attn_3`.
- Required runtime knob: `torch.backends.cudnn.benchmark_limit = 32`.
- Required Dynamo limits: `torch._dynamo.config.recompile_limit = 64` and
  `torch._dynamo.config.accumulated_recompile_limit = 1024`.

The PyTorch wheel currently used here is:

```text
torch-2.15.0a0+git65cd5f0-cp312-cp312-linux_x86_64.whl
sha256=54c6c4cb23ed41b41a0387cb617272d0b24999d07e82bc6918a0557d8a937bbb
```

This PyTorch build is identified by both a version and a source commit. The
version `2.15.0a0+git65cd5f0` is what the installed wheel reports at runtime.
The full commit `65cd5f09ea45cf4c7e55821b840e2bbaf45a7ec5` is the exact
PyTorch source revision to check out before compiling the wheel. This is needed
because the verified stack uses a development build, not a standard released
`pip install torch==...` package.

## Build From Scratch

Clone this repository and use a fresh Python 3.12.14 environment:

```bash
git clone <unstep-repo-url> unstep
cd unstep
export UNSTEP_REPO="$PWD"
python3.12 -m venv .venv
. .venv/bin/activate
```

Install the build tools first:

```bash
with-proxy python -m pip install -r requirements-build.txt
```

Install CUDA 13.3 and cuDNN 9.25.1. A system install is fine. If using NVIDIA
Python packages, use the versions in `requirements-cuda13.txt` and point
`CUDA_HOME` at the resulting CUDA 13.3 tree.

```bash
with-proxy python -m pip install -r requirements-cuda13.txt
```

Build/install PyTorch `2.15.0a0+git65cd5f0` from source at the locked
commit. In concrete terms: clone PyTorch, check out the exact commit, build the
wheel, then install that wheel into the environment:

```bash
mkdir -p .build
with-proxy git clone --recursive https://github.com/pytorch/pytorch .build/pytorch
cd .build/pytorch
git checkout 65cd5f09ea45cf4c7e55821b840e2bbaf45a7ec5
git submodule sync
with-proxy git submodule update --init --recursive
export CUDA_HOME=<cuda-13.3-prefix>
export CUDNN_INCLUDE_DIR="$CUDA_HOME/include"
export CUDNN_LIBRARY="$CUDA_HOME/lib/libcudnn.so"
export USE_CUDA=1
export USE_CUDNN=1
export USE_NCCL=0
export BUILD_TEST=0
export TORCH_CUDA_ARCH_LIST='9.0'
python -m build --wheel --no-isolation
python -m pip install dist/torch-2.15.0a0+git65cd5f0-*.whl
cd "$UNSTEP_REPO"
```

Then install the Python dependencies:

```bash
with-proxy python -m pip install -r requirements-generation.txt
python -m pip install -e .
```

Build/install FA3 against this same Python/Torch/CUDA stack. The verified
local overlay has extension hash:

```text
flash_attn_3/_C.abi3.so sha256=217bce6218eda0a5665b29efe7ef0bdb64584089c1c9703a20368bdaae4584b1
```

Make sure the import check reports FA3:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/runtime:$PWD" python - <<'PY'
import importlib.util
import torch
import triton

print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version())
print("triton", triton.__version__)
print("benchmark_limit", torch.backends.cudnn.benchmark_limit)
print("flash_attn_interface", importlib.util.find_spec("flash_attn_interface") is not None)
print("flash_attn_3", importlib.util.find_spec("flash_attn_3") is not None)
PY
```

## Source Repositories And Assets

UnStep expects the upstream Self-Forcing source tree and VBench source tree to
be available locally, but source checkouts, model weights, prompt copies,
benchmark images, and benchmark model caches are not committed in this folder.
`assets/` is intentionally gitignored and is local-only; create it with the
commands below.

Use this layout:

```text
assets/
  data/
    vbench_all_dimension.txt
    vbench_all_dimension_extended.txt
  source/
    Self-Forcing-main/
    VBench/
  weights/
    Wan2.1_VAE.pth
    self_forcing_weights.tar
    vbench/
```

Clone the upstream source repositories:

```bash
mkdir -p assets/source assets/data assets/weights .build
with-proxy git clone https://github.com/guandeh17/Self-Forcing.git assets/source/Self-Forcing-main
with-proxy git clone https://github.com/Vchitect/VBench.git assets/source/VBench
```

If your machine has normal outbound network access, omit `with-proxy`. The
launchers use these paths through `UNSTEP_ASSETS_ROOT`; the source trees can
also live elsewhere if `UNSTEP_SOURCE_URI` and `UNSTEP_VBENCH_SRC` are set.

Copy the prompt files from Self-Forcing:

```bash
cp assets/source/Self-Forcing-main/prompts/vbench/all_dimension.txt \
  assets/data/vbench_all_dimension.txt
cp assets/source/Self-Forcing-main/prompts/vbench/all_dimension_extended.txt \
  assets/data/vbench_all_dimension_extended.txt
```

Download the generation weights:

```bash
with-proxy huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir .build/Wan2.1-T2V-1.3B \
  --local-dir-use-symlinks False

with-proxy huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt \
  --local-dir .build/Self-Forcing \
  --local-dir-use-symlinks False
```

The launchers expect `assets/weights/Wan2.1_VAE.pth` and
`assets/weights/self_forcing_weights.tar`. The tar is a local packaging of the
Wan 2.1 1.3B model files plus the Self-Forcing DMD checkpoint with this
internal structure:

```text
checkpoints/self_forcing_dmd.pt
wan_models/Wan2.1-T2V-1.3B/config.json
wan_models/Wan2.1-T2V-1.3B/configuration.json
wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors
wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth
wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/*
```

One way to build the expected local files is:

```bash
cp .build/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth assets/weights/Wan2.1_VAE.pth
mkdir -p .build/unstep_weights/checkpoints .build/unstep_weights/wan_models
cp .build/Self-Forcing/checkpoints/self_forcing_dmd.pt \
  .build/unstep_weights/checkpoints/self_forcing_dmd.pt
cp -a .build/Wan2.1-T2V-1.3B \
  .build/unstep_weights/wan_models/Wan2.1-T2V-1.3B
tar -C .build/unstep_weights -cf assets/weights/self_forcing_weights.tar .
```

If you keep the checkpoint or Wan files in a different layout, pass explicit
paths with `UNSTEP_WEIGHTS_URI` and `UNSTEP_VAE_URI`.

Full-946 generation uses the extended prompt file. VBench scoring uses
`assets/source/VBench/vbench/VBench_full_info.json`.

For VBench scoring, install VBench-specific Python dependencies into a separate
overlay so they do not overwrite the generation stack:

```bash
with-proxy python -m pip install --target .vbench_deps -r requirements-vbench.txt
export UNSTEP_VBENCH_PYTHONPATH="$PWD/.vbench_deps"
```

The VBench source tree is not modified; `scripts/vbench_eval_entry.py` supplies
a small runtime compatibility shim for the current `transformers` import
locations.

Download the standard VBench pretrained scoring models using the upstream
files under `assets/source/VBench/pretrained/` and put/cache them under
`assets/weights/vbench`. Required sources include:

```text
pretrained/aesthetic_model/model_path.txt
pretrained/amt_model/download.sh
pretrained/caption_model/model_path.txt
pretrained/clip_model/model_path.txt
pretrained/grit_model/model_path.txt
pretrained/pyiqa_model/model_path.txt
pretrained/raft_model/download.sh
pretrained/umt_model/model_path.txt
pretrained/viclip_model/model_path.txt
```

This cache contains ViCLIP, CLIP, DINO, AMT, RAFT, GRiT/DenseCap, Tag2Text,
pyiqa/aesthetic, CoTracker, and BERT files as required by VBench. For BERT
tokenizer/config files, use:

```bash
with-proxy huggingface-cli download bert-base-uncased \
  --local-dir assets/weights/vbench/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594 \
  --allow-pattern config.json \
  --allow-pattern tokenizer.json \
  --allow-pattern tokenizer_config.json \
  --allow-pattern vocab.txt \
  --allow-pattern special_tokens_map.json \
  --local-dir-use-symlinks False
```

For VBench-I2V experiments, use the upstream data instructions in
`assets/source/VBench/vbench2_beta_i2v/README.md`. The upstream helper is:

```bash
with-proxy python -m pip install gdown
cd assets/source/VBench
with-proxy sh vbench2_beta_i2v/download_data.sh
```

That downloads `i2v-bench-info.json`, `crop.zip`, and `origin.zip` and unpacks
the I2V image suite. These files are external benchmark data and should not be
committed.

For VBench-long experiments, use the upstream instructions in
`assets/source/VBench/vbench2_beta_long/README.md` and keep any long-video
benchmark data or model caches under `assets/weights/vbench` or another local
path passed through the corresponding `UNSTEP_*` environment variable.

Build Detectron2 after PyTorch and CUDA are installed:

```bash
with-proxy git clone --depth 1 https://github.com/facebookresearch/detectron2.git .build/detectron2
UNSTEP_CUDA_OVERLAY=<cuda-wheel-overlay-if-used> \
UNSTEP_VBENCH_DEPS="$PWD/.vbench_deps" \
./scripts/install_detectron2_for_vbench.sh .build/detectron2
```

If CUDA is installed from `requirements-cuda13.txt`, set:

```bash
export UNSTEP_CUDA_OVERLAY=<path-containing-nvidia/cu13>
export UNSTEP_CUDA="$UNSTEP_CUDA_OVERLAY/nvidia/cu13"
```

The `PYTHONPATH="$PWD/runtime:$PWD"` part is required. It loads
`runtime/sitecustomize.py`, which sets:

```text
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark_limit = 32
torch._dynamo.config.recompile_limit = 64
torch._dynamo.config.accumulated_recompile_limit = 1024
torch._inductor.config._use_fp64_for_unbacked_floats = False
```

## Speed Probe

Use `scripts/run_h100_speed_probe.sh`. It creates per-run compiler cache
directories and disables remote compile caches so one run does not reuse another
run's generated artifacts.

The script also runs untimed adaptive prewarm before measurement:

- `PREWARM_REPEATS=6` maximum by default.
- `PREWARM_MIN_FPS=48.0` by default.
- TorchInductor/Triton compile and autotune workers are quiesced after each
  prewarm pass and again immediately before measurement.
- Only the prompts after this stable prewarm are reported as measured prompts.

For a repo-local setup:

```bash
./scripts/run_h100_speed_probe.sh H100_SPEED6
```

For an external env or asset location:

```bash
UNSTEP_PYENV=<python-3.12.14-env> \
UNSTEP_CUDA=<cuda-13.3-prefix> \
UNSTEP_ASSETS_ROOT=<assets-dir> \
UNSTEP_FA3=<fa3-overlay> \
UNSTEP_EXTRA_PYTHONPATH=<optional-dependency-overlay> \
./scripts/run_h100_speed_probe.sh H100_SPEED6
```

## Full-946 Generation

For paper-quality full-946 generation, use the default launcher:

```bash
./scripts/run_h100_full946_generation.sh H100_SIGMA002_FULL946
```

This launcher defaults to `--clean-sigma-override 0.02`, applied to both the
clean-input re-noising and the emitted prediction, with DiT conditioning `t=0`.

Each run is output-isolated. If `results/<RUN_NAME>` already exists, the script
fails instead of appending to it. Use a fresh run name for normal operation. Set
`UNSTEP_OVERWRITE_OUTPUT=1` only when intentionally replacing an existing run.
Per-shard work directories are recreated before generation, so staged source or
temporary files from an older run cannot be reused accidentally.
Each shard uses a run-local TorchInductor/Triton/CUDA cache directory. This
keeps runs independent while preserving the same timed DiT+VAE region.

The canonical generation protocol is fixed by the script:

```text
prompt file: assets/data/vbench_all_dimension_extended.txt
prompt indices: all
num shards: 8
shard mode: contiguous
seed mode: sequential_global
rng skip mode: initial_only
decode mode: full
FA3 backend: h100
clean sigma override: 0.02 for both re-noising and emission
```

Score the generated videos with two local GPUs:

```bash
./scripts/run_vbench_score.sh H100_SIGMA002_FULL946
```

The scorer deletes MP4 files after a successful merge by default. Preserve
videos with:

```bash
UNSTEP_DELETE_VIDEOS_AFTER_SCORE=0 ./scripts/run_vbench_score.sh H100_SIGMA002_FULL946
```
