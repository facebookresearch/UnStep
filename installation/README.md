# Installation

Start in the root of your existing UnStep clone; do not clone it again inside
itself. Use Linux, Python 3.12.14 with development headers and `venv`, Git,
a C/C++ compiler supported by CUDA 13.3, and an NVIDIA driver supporting your GPU
and toolkit. Python packages do not install the compiler, driver, or interpreter.

The commands below reconstruct the recorded source stack; they are not a claim
that a fresh end-to-end build has been tested on every platform. Keep the same
activated environment and CUDA exports throughout these steps. Use absolute
paths for environment and asset overrides.

## 1. Python environment

Create the environment once, or activate an existing dedicated setup environment:

```bash
export UNSTEP_REPO="$PWD"
export UNSTEP_PYENV="${UNSTEP_PYENV:-$UNSTEP_REPO/.venv}"
if [ ! -x "$UNSTEP_PYENV/bin/python" ]; then
  python3.12 -m venv "$UNSTEP_PYENV"
fi
. "$UNSTEP_PYENV/bin/activate"
python -c 'import sys; assert sys.version_info[:3] == (3, 12, 14), sys.version'
python -m pip install -r installation/requirements-build.txt
```

PyYAML is needed during torch code generation, before generation dependencies
are installed. Use a new environment rather than modifying a production one.

## 2. CUDA and cuDNN

Use **one** of the following paths. Both must supply CUDA 13.3 and cuDNN 9.25.1.

### System development toolkit

Use the actual include/library locations of your CUDA and cuDNN development
installations; cuDNN may be separate from CUDA and may use `lib64` rather than `lib`:

```bash
export CUDA_HOME=/path/to/cuda-13.3
export CUDNN_INCLUDE_DIR=/path/to/cudnn/include
export CUDNN_LIBRARY=/path/to/cudnn/lib/libcudnn.so.9
export UNSTEP_CUDA="$CUDA_HOME"
unset UNSTEP_CUDA_OVERLAY
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$(dirname "$CUDNN_LIBRARY"):$CUDA_HOME/lib64:$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

### NVIDIA Python packages

```bash
python -m pip install -r installation/requirements-cuda13.txt
. installation/cuda_wheels.sh
```

The helper derives paths from the active venv: CUDA is under `nvidia/cu13`,
cuDNN under `nvidia/cudnn`, and the cuDNN library is `libcudnn.so.9`. It adds
missing unversioned CUDA linker symlinks inside that venv and exports build and
runtime search paths. cuRAND is explicit because torch's CMake links
`CUDA::curand`; the bare `cuda-toolkit` metapackage does not install all extras.

In a new shell, activate the same venv and repeat the system exports or source
`installation/cuda_wheels.sh` again; reinstalling the packages is unnecessary.

## 3. PyTorch from the pinned public source

The recorded torch build is `2.15.0a0+git65cd5f0`, with CUDA `13.3`.
Use the same commit, not the moving upstream branch:

```bash
mkdir -p .build
git clone https://github.com/pytorch/pytorch.git .build/pytorch
git -C .build/pytorch checkout --detach 65cd5f09ea45cf4c7e55821b840e2bbaf45a7ec5
git -C .build/pytorch submodule sync --recursive
git -C .build/pytorch submodule update --init --recursive
export USE_CUDA=1 USE_CUDNN=1 USE_NCCL=0 BUILD_TEST=0
export MAX_JOBS="${MAX_JOBS:-4}"
export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | paste -sd ';' -)}"
: "${TORCH_CUDA_ARCH_LIST:?Set the deployment GPU compute capability before building}"
(
  cd .build/pytorch
  python - <<'PY'
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from build import ProjectBuilder

constraints = Path(os.environ["UNSTEP_REPO"]) / "installation/requirements-build.txt"
def install(requirements):
    if requirements:
        subprocess.run([sys.executable, "-m", "pip", "install", "--constraint",
                        str(constraints), *sorted(requirements)], check=True)

with open("pyproject.toml", "rb") as f:
    install(tomllib.load(f)["build-system"]["requires"])
install(ProjectBuilder(".").get_requires_for_build("wheel"))
PY
  python -m build --wheel --no-isolation
  python -m pip install --no-deps dist/torch-*.whl
)
```

The pre-build step installs the pinned checkout's declared and backend-reported
build requirements, constrained by our build-tool pins. This is necessary for
`--no-isolation`; do not rely on generation packages installed later to supply
build dependencies. A constraint conflict should stop setup, not silently change
the recorded tool versions.

Build parallelism defaults to four jobs. Set `MAX_JOBS` before this block to
adjust the CPU/RAM budget; CMake uses the same limit. This is a build resource
control, not an inference runtime setting.

Skip a dependency's clone command if you already have that checkout, then select
the stated revision and initialize its submodules. The architecture list must
include the deployment GPU: H100 is `9.0`, GB200 is `10.0`. A build on a machine
without a visible target GPU needs this variable set explicitly. GPU architecture
does not make an x86-64 wheel usable on Grace/aarch64.

## 4. Python dependencies and Triton

On **aarch64**, build [native decord](native_dependencies.md#decord-on-aarch64)
before installing the requirements below. PyPI's decord 0.6.0 release has no
aarch64 wheel or source distribution; `--no-binary` alone cannot supply it.

```bash
python -m pip install -r installation/requirements-generation.txt
python -m pip install --no-deps triton==3.8.0
```

These requirements also provide torch's Python runtime dependencies. OpenCLIP
and timm are not imported by the generation path; install scoring dependencies
separately through [evaluation](../evaluation/).

## 5. Native torchvision

The recorded torchvision binary is `0.30.0.dev20260906+cu134`, from commit
`c8be83eb6a3ce76e67d92caf1f8358ec533cd3d0`. Its wheel metadata requires
`torch==2.15.0.dev20260906`, which differs from the source-built torch version.
Do not let pip replace torch to satisfy that binary's metadata. Build this source
revision against the active torch/CUDA installation, including on Grace/aarch64:

```bash
git clone https://github.com/pytorch/vision.git .build/vision
git -C .build/vision checkout --detach c8be83eb6a3ce76e67d92caf1f8358ec533cd3d0
(
  cd .build/vision
  unset PYTORCH_VERSION BUILD_VERSION
  FORCE_CUDA=1 python -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
  python -m pip install --no-deps dist/torchvision-*.whl
)
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
```

The native build's version suffix can differ from the recorded `+cu134` binary.
Unsetting `PYTORCH_VERSION` avoids inheriting an external nightly dependency
override. Verify the resulting package with `pip check`; if its metadata still
requires a different torch version, stop and inspect that requirement rather
than replacing the source-built torch. This source build has not been run here.
If using that exact binary instead, use `--no-deps` and expect its declared torch
version mismatch in `pip check`; this does not establish binary compatibility.

## 6. Attention, assets, and generation

Follow [attention installation](flash_attention.md): **H100 uses FA3; GB200 uses
FA2**. Then prepare [model files and prompts](assets.md) and follow
[generation](../generation/). Both launchers load `runtime/sitecustomize.py`,
which preserves these required settings:

- `torch.backends.cudnn.benchmark = True` and `benchmark_limit = 32`.
- Dynamo `recompile_limit = 64` and `accumulated_recompile_limit = 1024`.
- Inductor `_use_fp64_for_unbacked_floats = False`.

The expected runtime is CUDA `13.3`, cuDNN `9.25.1` (`92501`), and Triton `3.8.0`.
