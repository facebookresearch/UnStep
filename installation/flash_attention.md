# FlashAttention installation

Run these commands from the repository root after installing PyTorch and CUDA.
This FA3 build targets compute capability 9.0 (`sm_90a`). Other architectures
require an attention backend compiled for their target.

Build FA3 from the `hopper/` directory of the public FlashAttention repository
at commit `c75d019dea9d910312974417bc28f190dfdda6d9`, using the Python environment
and CUDA 13.3 toolkit from the [installation guide](README.md). Initialize its CUTLASS submodule at the
revision recorded by that commit, `7127592069c2fe01b041e174ba4345ef9b279671`:

```bash
mkdir -p .build
git clone https://github.com/Dao-AILab/flash-attention.git .build/flash-attention
git -C .build/flash-attention checkout c75d019dea9d910312974417bc28f190dfdda6d9
git -C .build/flash-attention submodule update --init --recursive csrc/cutlass
```

The wrapper uses forward attention in BF16 with head dimension 128.
The following flags match the installed FA3 build, retaining that configuration
and disabling unused variants. These flags affect attention, not the FP16 VAE.
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
```

This installs `flash_attn_interface`, `flash_attn_config`, and `flash_attn_3`
into the active Python environment. Leave `UNSTEP_FA3` unset for this installation.
That variable is only needed when loading FA3 from a separate directory.

The recorded binary below identifies the existing validated installation.
Rebuilding can change the binary hash with the compiler and build paths:

```text
flash_attn_3/_C.abi3.so sha256=217bce6218eda0a5665b29efe7ef0bdb64584089c1c9703a20368bdaae4584b1
```

Check that FA3 imports with the installed PyTorch/CUDA stack:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/runtime:$PWD" python - <<'PY'
import torch
import triton
import flash_attn_interface
import flash_attn_3._C
from flash_attn_config import CONFIG

print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version())
print("triton", triton.__version__)
print("benchmark_limit", torch.backends.cudnn.benchmark_limit)
print("flash_attn_interface", flash_attn_interface.__file__)
print("flash_attn_3", flash_attn_3._C.__file__)
print("FA3 build flags", CONFIG["build_flags"])
PY
```
