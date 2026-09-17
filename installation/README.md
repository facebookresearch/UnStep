# Installation

Run commands from the repository root after cloning it. Install Python, CUDA,
PyTorch, and the Python dependencies below, then follow:

- [FlashAttention installation](flash_attention.md).
- [Model files and prompts](assets.md).
- [Video generation](../generation/).

## Exact Stack

- Python 3.12.14.
- PyTorch from `https://github.com/pytorch/pytorch` commit
  `65cd5f09ea45cf4c7e55821b840e2bbaf45a7ec5`.
- Torch runtime reports CUDA `13.3`.
- cuDNN runtime reports `9.25.1`.
- Triton `3.8.0+gitdf3f91dd`.
- torchvision `0.30.0.dev20260906+cu134`.
- FlashAttention, installed using the [backend build instructions](flash_attention.md).
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
git clone https://github.com/facebookresearch/UnStep.git unstep
cd unstep
export UNSTEP_REPO="$PWD"
python3.12 -m venv .venv
. .venv/bin/activate
```

Install the build tools first:

```bash
python -m pip install -r installation/requirements-build.txt
```

Install CUDA 13.3 and cuDNN 9.25.1. A system install is fine. If using NVIDIA
Python packages, use the versions in `installation/requirements-cuda13.txt` and point
`CUDA_HOME` at the resulting CUDA 13.3 tree.

```bash
python -m pip install -r installation/requirements-cuda13.txt
```

Build/install PyTorch `2.15.0a0+git65cd5f0` from source at the locked
commit. In concrete terms: clone PyTorch, check out the exact commit, build the
wheel, then install that wheel into the environment:

```bash
mkdir -p .build
git clone --recursive https://github.com/pytorch/pytorch .build/pytorch
cd .build/pytorch
git checkout 65cd5f09ea45cf4c7e55821b840e2bbaf45a7ec5
git submodule sync
git submodule update --init --recursive
export CUDA_HOME=/path/to/cuda-13.3
export CUDNN_INCLUDE_DIR="$CUDA_HOME/include"
export CUDNN_LIBRARY="$CUDA_HOME/lib/libcudnn.so"
export USE_CUDA=1
export USE_CUDNN=1
export USE_NCCL=0
export BUILD_TEST=0
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | paste -sd ';' -)"
python -m build --wheel --no-isolation
python -m pip install dist/torch-2.15.0a0+git65cd5f0-*.whl
cd "$UNSTEP_REPO"
```

Then install the Python dependencies:

```bash
python -m pip install -r installation/requirements-generation.txt
python -m pip install -e .
```


The runtime settings are applied by the generation launcher through
`runtime/sitecustomize.py`. The recorded environment and source hashes are in
[benchmarks](../benchmarks/).
