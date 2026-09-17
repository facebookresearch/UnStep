# Model files and prompts

Run from the repository root with the Python environment from
[installation](README.md) activated.

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

Use existing source directories by setting `UNSTEP_SOURCE_URI` and
`UNSTEP_VBENCH_SRC` first, or clone the missing repositories:

```bash
export UNSTEP_ASSETS_ROOT="${UNSTEP_ASSETS_ROOT:-$PWD/assets}"
export UNSTEP_SOURCE_URI="${UNSTEP_SOURCE_URI:-$UNSTEP_ASSETS_ROOT/source/Self-Forcing-main}"
export UNSTEP_VBENCH_SRC="${UNSTEP_VBENCH_SRC:-$UNSTEP_ASSETS_ROOT/source/VBench}"
mkdir -p "$UNSTEP_ASSETS_ROOT"/{source,data,weights} .build
if [ ! -d "$UNSTEP_SOURCE_URI" ]; then
  git clone https://github.com/guandeh17/Self-Forcing.git "$UNSTEP_SOURCE_URI"
fi
if [ ! -d "$UNSTEP_VBENCH_SRC" ]; then
  git clone https://github.com/Vchitect/VBench.git "$UNSTEP_VBENCH_SRC"
fi
```

The launchers use these paths through `UNSTEP_ASSETS_ROOT`; the source trees can
also live elsewhere if `UNSTEP_SOURCE_URI` and `UNSTEP_VBENCH_SRC` are set.

Copy the prompt files from Self-Forcing:

```bash
cp "$UNSTEP_SOURCE_URI/prompts/vbench/all_dimension.txt" \
  "$UNSTEP_ASSETS_ROOT/data/vbench_all_dimension.txt"
cp "$UNSTEP_SOURCE_URI/prompts/vbench/all_dimension_extended.txt" \
  "$UNSTEP_ASSETS_ROOT/data/vbench_all_dimension_extended.txt"
```

Download the generation weights:

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir .build/Wan2.1-T2V-1.3B

huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt \
  --local-dir .build/Self-Forcing
```

These upstream sources and model downloads are not revision-pinned here. For
reproduction, retain the source checkout SHAs and model revisions with the run;
use `--revision` for a known model revision rather than assuming `main` is fixed.

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
cp .build/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth "$UNSTEP_ASSETS_ROOT/weights/Wan2.1_VAE.pth"
mkdir -p .build/unstep_weights/checkpoints \
  .build/unstep_weights/wan_models/Wan2.1-T2V-1.3B
cp .build/Self-Forcing/checkpoints/self_forcing_dmd.pt \
  .build/unstep_weights/checkpoints/self_forcing_dmd.pt
cp -a .build/Wan2.1-T2V-1.3B/. \
  .build/unstep_weights/wan_models/Wan2.1-T2V-1.3B/
tar -C .build/unstep_weights -cf "$UNSTEP_ASSETS_ROOT/weights/self_forcing_weights.tar" .
```

`UNSTEP_WEIGHTS_URI` and `UNSTEP_VAE_URI` can change the archive and VAE file
locations, not the archive's internal layout. A non-`.tar` weights path is
treated as one checkpoint file, not a Wan directory; in that case the source
tree must already contain the complete `wan_models/Wan2.1-T2V-1.3B` layout.

Full-946 generation uses the extended prompt file. VBench scoring uses
`assets/source/VBench/vbench/VBench_full_info.json`.
