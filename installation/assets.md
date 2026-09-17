# Model files and prompts

Run the commands below from the repository root.

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
git clone https://github.com/guandeh17/Self-Forcing.git assets/source/Self-Forcing-main
git clone https://github.com/Vchitect/VBench.git assets/source/VBench
```

The launchers use these paths through `UNSTEP_ASSETS_ROOT`; the source trees can
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
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir .build/Wan2.1-T2V-1.3B \
  --local-dir-use-symlinks False

huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt \
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
