# FlashAttention installation

Complete steps 1-5 of [installation](README.md) first. Run from the repository
root in the same activated environment with its CUDA exports. In a new shell,
activate `UNSTEP_PYENV` (default `.venv`) and restore the selected CUDA setup.

**H100 uses FA3; GB200 uses FA2.** Use one of the two builds below, from the
same public source revision. These are source-based installation instructions,
not a claim that a fresh GB200 build has been tested. On Grace/aarch64, compile
natively; x86-64 wheels are not usable.

```bash
mkdir -p .build
git clone https://github.com/Dao-AILab/flash-attention.git .build/flash-attention
git -C .build/flash-attention checkout --detach c75d019dea9d910312974417bc28f190dfdda6d9
git -C .build/flash-attention submodule update --init --recursive csrc/cutlass
```

Skip the clone command if that checkout already exists.

## H100: FA3

Build from `hopper/`. This build targets `sm_90a`, not GB200's `sm_100`.
The wrapper uses forward attention in BF16 with head dimension 128.
The following flags retain that configuration
and disable unused variants. These flags affect attention, not the FP16 VAE.
The build uses the existing PyTorch installation and CUDA toolkit:

```bash
(
  cd .build/flash-attention/hopper
  export FLASH_ATTENTION_FORCE_BUILD=TRUE
  export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE
  export FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE
  export FLASH_ATTENTION_FORCE_CXX11_ABI=FALSE
  export MAX_JOBS=4 NVCC_THREADS=2
  for feature in BACKWARD SPLIT PAGEDKV APPENDKV SOFTCAP PACKGQA FP16 FP8 \
      VARLEN CLUSTER HDIM64 HDIM96 HDIM192 HDIM256 SM80 HDIMDIFF64 HDIMDIFF192; do
    export "FLASH_ATTENTION_DISABLE_${feature}=TRUE"
  done
  for feature in LOCAL HDIM128; do
    export "FLASH_ATTENTION_DISABLE_${feature}=FALSE"
  done
  export FLASH_ATTENTION_ENABLE_VCOLMAJOR=FALSE
  python -m pip install --no-build-isolation --no-deps .
)
export UNSTEP_ATTENTION_BACKEND=h100
```

This installs `flash_attn_interface`, `flash_attn_config`, and `flash_attn_3`
into the active Python environment. Leave `UNSTEP_FA3` unset for this installation.
That variable is only needed when loading FA3 from a separate directory.

## GB200: FA2

Build the root FA2 package, version `2.8.4` at the pinned commit. FA2 uses
`FLASH_ATTN_CUDA_ARCHS=100`; PyTorch's separate `TORCH_CUDA_ARCH_LIST=10.0` does
not control FA2's explicit compiler flags. Do not reuse the FA3 variant-disable
loop for this build:

```bash
(
  cd .build/flash-attention
  export BUILD_TARGET=cuda
  export FLASH_ATTENTION_FORCE_BUILD=TRUE
  export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE
  export FLASH_ATTENTION_FORCE_CXX11_ABI=FALSE
  export FLASH_ATTN_CUDA_ARCHS=100
  export MAX_JOBS=4 NVCC_THREADS=2
  python -m pip install --no-build-isolation --no-deps .
)
export UNSTEP_ATTENTION_BACKEND=fa2
unset UNSTEP_FA3
```

The ABI flag leaves the existing torch ABI unchanged; it does not force an old
ABI. The root setup emits SM100 code with the selected CUDA toolkit. This source
version is verified locally, but is not evidence of the version used for the
reported historical GB200 timings.

Both launchers accept `UNSTEP_ATTENTION_BACKEND=fa2`; the Python entrypoint uses
`--fa3-import-backend fa2`. The default `auto` selects FA3 on Hopper and FA2 on
GB200. Explicit `fa2` avoids choosing an installed FA3 on GB200; no runtime
precision, compiler, cuDNN, or method settings need to be changed.

## Import and runtime check

This check imports the selected backend without generating a video. It does not
prove that GPU kernels build or execute correctly:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/runtime:$PWD" python - <<'PY'
import os
import torch
import triton
from unstep.wrapper_runtime import resolve_attention_backend

assert torch.version.cuda == "13.3", torch.version.cuda
assert torch.backends.cudnn.version() == 92501, torch.backends.cudnn.version()
assert triton.__version__.split("+")[0] == "3.8.0", triton.__version__
assert torch.backends.cudnn.benchmark
assert torch.backends.cudnn.benchmark_limit == 32
assert torch._dynamo.config.recompile_limit == 64
assert torch._dynamo.config.accumulated_recompile_limit == 1024
assert torch._inductor.config._use_fp64_for_unbacked_floats is False

print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version())
print("triton", triton.__version__)
backend = resolve_attention_backend(os.environ.get("UNSTEP_ATTENTION_BACKEND", "auto"))
print("attention backend", backend)
if backend == "fa2":
    import flash_attn
    print("FA2", flash_attn.__version__, flash_attn.__file__)
else:
    import flash_attn_interface
    import flash_attn_3._C
    from flash_attn_config import CONFIG
    print("flash_attn_interface", flash_attn_interface.__file__)
    print("flash_attn_3", flash_attn_3._C.__file__)
    print("FA3 build flags", CONFIG["build_flags"])
PY
```
