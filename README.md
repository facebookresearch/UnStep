# UnStep H100 Setup

This folder contains the H100 generation setup for reproducing the H100
generation path.

## Files

- `SETUP_H100.md`: step-by-step setup from a fresh Python 3.12.14
  environment.
- `requirements-build.txt`: install with
  `python -m pip install -r requirements-build.txt` before building PyTorch
  from source.
- `requirements-cuda13.txt`: install with
  `python -m pip install -r requirements-cuda13.txt` only if CUDA/cuDNN are
  installed through Python packages instead of a system CUDA install.
- `requirements-generation.txt`: install with
  `python -m pip install -r requirements-generation.txt` after PyTorch/CUDA are
  ready.
- `requirements-vbench.txt`: install into a separate scoring overlay with
  `python -m pip install --target .vbench_deps -r requirements-vbench.txt`.
- `ENVIRONMENT_LOCK.json`: exact stack, source hashes, and runtime knobs
  in machine-readable form. This is not installed or run;
  scripts can parse it later to check that a reproduction environment matches
  the validated one.
- `SOURCE_HASHES.txt`: source hashes for the generation/runtime files used by
  the H100 full-946 launcher.
- `unstep/clean_sigma.py`: the clean-input correction, using the same sigma
  override as the emitted prediction.
- `runtime/sitecustomize.py`: required H100-runtime knobs, including
  `torch.backends.cudnn.benchmark_limit = 32`.
- `imageio-ffmpeg` `0.6.0` for repo-local MP4 writing.
- `scripts/run_h100_speed_probe.sh`: one-H100 speed probe launcher.
- `scripts/run_h100_full946_generation.sh`: canonical H100 full-946
  generation launcher for paper-quality runs.
- `scripts/install_detectron2_for_vbench.sh`: optional helper for installing
  Detectron2 into the VBench scoring dependency directory.

## Required Stack

- Python `3.12.14`
- PyTorch `2.15.0a0+git65cd5f0`
- PyTorch source commit `65cd5f09ea45cf4c7e55821b840e2bbaf45a7ec5`
- Torch CUDA runtime `13.3`
- CUDA toolkit `13.3.1`
- cuDNN runtime `9.25.1`
- Triton `3.8.0+gitdf3f91dd`
- FA3 overlay providing `flash_attn_interface` and `flash_attn_3`
- runtime knobs from `runtime/sitecustomize.py`:
  `torch.backends.cudnn.benchmark = True`,
  `torch.backends.cudnn.benchmark_limit = 32`,
  `torch._dynamo.config.recompile_limit = 64`, and
  `torch._dynamo.config.accumulated_recompile_limit = 1024`

PyTorch `2.15.0a0+git65cd5f0` is not a normal released pip wheel. The
`2.15.0a0` part is the PyTorch development version, and `+git65cd5f0` means
the wheel was built from a git checkout whose short commit hash is `65cd5f0`.
For reproducibility, build PyTorch from the full source commit
`65cd5f09ea45cf4c7e55821b840e2bbaf45a7ec5`; after installation,
`python -c "import torch; print(torch.__version__)"` should report
`2.15.0a0+git65cd5f0`.

## Install Order

1. Create a fresh Python `3.12.14` environment.
2. Install the PyTorch build dependencies from `requirements-build.txt`.
3. Install/build CUDA `13.3` and cuDNN `9.25.1` using
   `requirements-cuda13.txt` or an equivalent system install.
4. Build PyTorch `2.15.0a0+git65cd5f0` from source commit
   `65cd5f09ea45cf4c7e55821b840e2bbaf45a7ec5`.
5. Install generation dependencies from `requirements-generation.txt`.
6. Install this package with `python -m pip install -e .`.
7. Install/build the FA3 overlay against the same Python/Torch/CUDA
   stack.
8. Run the import/runtime check in `SETUP_H100.md`.
9. Run `scripts/run_h100_speed_probe.sh`.
10. Run `scripts/run_h100_full946_generation.sh`.
11. For VBench scoring, clone VBench, install its dependencies without
    replacing the locked torch stack, download the VBench pretrained models,
    install Detectron2, and run `scripts/run_vbench_score.sh`.

The complete command details are in `SETUP_H100.md`.

## External Sources And Assets

The repository does not commit source checkouts, model weights, benchmark
images, prompt copies, or benchmark model caches. `assets/` is intentionally
gitignored and is local-only; a fresh checkout needs users to create this local
layout by following the commands below:

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

Create the folders and clone the upstream source repositories:

```bash
mkdir -p assets/source assets/data assets/weights .build
git clone https://github.com/guandeh17/Self-Forcing.git assets/source/Self-Forcing-main
git clone https://github.com/Vchitect/VBench.git assets/source/VBench
```

The launchers use these paths through `UNSTEP_ASSETS_ROOT`; the source trees can also live
elsewhere if `UNSTEP_SOURCE_URI` and `UNSTEP_VBENCH_SRC` are set explicitly.

Copy the prompt files from Self-Forcing:

```bash
cp assets/source/Self-Forcing-main/prompts/vbench/all_dimension.txt \
  assets/data/vbench_all_dimension.txt
cp assets/source/Self-Forcing-main/prompts/vbench/all_dimension_extended.txt \
  assets/data/vbench_all_dimension_extended.txt
```

Download the Wan/Self-Forcing generation weights from Hugging Face:

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir .build/Wan2.1-T2V-1.3B --local-dir-use-symlinks False
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .build/Self-Forcing --local-dir-use-symlinks False
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

Generation uses `assets/data/vbench_all_dimension_extended.txt`; scoring uses
VBench's standard `vbench/VBench_full_info.json` from the cloned VBench repo.
Keep the prompt indices and seed/shard settings from the launchers for
reproducible full-946 runs.

For standard VBench scoring, download the pretrained scoring models following
the upstream files under `assets/source/VBench/pretrained/` and store/cache them
under `assets/weights/vbench`. The scorer uses this cache via
`UNSTEP_VBENCH_CACHE`. Required sources include:

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

The standard cache contains ViCLIP, CLIP, DINO, AMT, RAFT, GRiT/DenseCap,
Tag2Text, pyiqa/aesthetic, CoTracker, and BERT files as required by VBench.
For BERT tokenizer/config files, use:

```bash
huggingface-cli download bert-base-uncased \
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
python -m pip install gdown
cd assets/source/VBench
sh vbench2_beta_i2v/download_data.sh
```

That downloads `i2v-bench-info.json`, `crop.zip`, and `origin.zip` and unpacks
the I2V image suite. These I2V files are external benchmark data and should not
be committed.

For VBench-long experiments, use the upstream instructions in
`assets/source/VBench/vbench2_beta_long/README.md` and keep any long-video
benchmark data or model caches under `assets/weights/vbench` or another local
path passed through the corresponding `UNSTEP_*` environment variable.

Install VBench scoring dependencies into an overlay instead of into the
generation environment:

```bash
python -m pip install --target .vbench_deps -r requirements-vbench.txt
```

Then set `UNSTEP_VBENCH_PYTHONPATH=.vbench_deps` when scoring.
The VBench source tree is not modified; `scripts/vbench_eval_entry.py` supplies
a small runtime compatibility shim for the current `transformers` import
locations.

## What To Run

The requirements files are ordinary pip requirement files:

```bash
python -m pip install -r requirements-build.txt
python -m pip install -r requirements-generation.txt
```

`requirements-cuda13.txt` is also installable with pip, but only use it when
CUDA/cuDNN are being provided by Python packages:

```bash
python -m pip install -r requirements-cuda13.txt
```

PyTorch is the exception: install/build version `2.15.0a0+git65cd5f0`
separately at the exact source commit before installing
`requirements-generation.txt`. The commit is the exact PyTorch source tree to
compile; the version string is what the installed wheel should report:

```bash
git clone --recursive https://github.com/pytorch/pytorch .build/pytorch
cd .build/pytorch
git checkout 65cd5f09ea45cf4c7e55821b840e2bbaf45a7ec5
git submodule sync
git submodule update --init --recursive
# Set CUDA_HOME to the CUDA 13.3 prefix, then build/install the wheel.
```

After PyTorch, CUDA/cuDNN, generation deps, and FA3 are installed, run a speed
probe:

```bash
./scripts/run_h100_speed_probe.sh H100_SPEED_PROBE
```

The speed probe also defaults to `--clean-sigma-override 0.02`, matching the
full-946 launcher.

For the canonical full-946 paper run, use the default generation launcher and
then score the generated videos with the two-GPU VBench scorer:

```bash
./scripts/run_h100_full946_generation.sh H100_SIGMA002_FULL946
./scripts/run_vbench_score.sh H100_SIGMA002_FULL946
```

The default full-946 launcher already sets `--clean-sigma-override 0.02`.
Override `UNSTEP_EXTRA_ARGS` only for an explicit ablation.

`--clean-sigma-override S` uses the same `S` in both clean-input re-noising,
`xt = (1 - S) * x + S * noise`, and the emitted prediction,
`x0 = xt - S * flow`. The DiT remains conditioned on `t=0`. Shift 5, scheduler
tables, and all nonzero denoising timesteps are unchanged. The package and both
generation launchers default to `S=0.02`.

Each process checks both coefficients before generation and records
`clean_sigma_validation`, including its `input_renoising` check. Generation
aborts if either coefficient is wrong. These checks consume no random draws
and run outside the speed timer.

Run the arithmetic and wrapper integration tests on CPU, without loading model
weights or generating videos:

```bash
UNSTEP_SOURCE_URI=assets/source/Self-Forcing-main \
  python -m unittest discover -s tests -v
```

The scorer deletes MP4s after a successful score by default. Set
`UNSTEP_DELETE_VIDEOS_AFTER_SCORE=0` if you need to preserve them.

The generation launcher fixes the generation protocol:

```text
prompt source: assets/data/vbench_all_dimension_extended.txt
prompt indices: all
num shards: 8
shard mode: contiguous
seed mode: sequential_global
rng skip mode: initial_only
decode mode: full
FA3 backend: h100
timed region: DiT latent generation + VAE decode
video writing/scoring: outside timed FPS
```

New runs are protected from older run outputs: the launcher refuses to reuse an
existing output directory unless `UNSTEP_OVERWRITE_OUTPUT=1` is set, and it
recreates each per-shard work directory before generation. This prevents stale
videos, JSONs, logs, or staged source files from being mixed into a new run.
The full-946 launcher uses isolated compiler caches by default. Source hashes
are recorded so a run can be audited against `SOURCE_HASHES.txt`.

## VBench Scoring Dependency

Full 16-dimension VBench scoring requires Detectron2 for the GRiT-based
object, color, and spatial-relation dimensions. Install it into the scoring
dependency overlay before running `scripts/run_vbench_score.sh`:

```bash
git clone --depth 1 https://github.com/facebookresearch/detectron2.git .build/detectron2
./scripts/install_detectron2_for_vbench.sh .build/detectron2
```

Detectron2 must be built after torch and CUDA are installed. If CUDA is provided
by `requirements-cuda13.txt`, `nvidia-cuda-nvcc`, `nvidia-cuda-crt`,
`nvidia-cuda-cccl`, and `nvidia-nvvm` are required because Detectron2 compiles
CUDA extensions during installation.

If using an external Python/CUDA setup, pass the same environment variables used
for generation, plus the scoring dependency target:

```bash
UNSTEP_PYENV=<python-3.12.14-env> \
UNSTEP_CUDA=<cuda-13.3-prefix> \
UNSTEP_VBENCH_DEPS=<vbench-dependency-dir> \
./scripts/install_detectron2_for_vbench.sh .build/detectron2
```

## Files Not Run Directly

Do not run or install these files:

- `ENVIRONMENT_LOCK.json`: machine-readable record of the environment,
  runtime knobs, and source hashes.
- `SOURCE_HASHES.txt`: quick hash checklist for source drift.

Use them to audit that a local setup matches the validated setup.

The JSON and Markdown files intentionally overlap. Markdown explains the setup
for a person; JSON stores the same critical facts in a form that can be checked
by code.

## Acknowledgments

UnStep builds on [Self Forcing](https://github.com/guandeh17/Self-Forcing),
including its inference pipeline, scheduler, and Wan2.1 model implementation.
Our runtime code adapts parts of that implementation. We use
[VBench](https://github.com/Vchitect/VBench) for evaluation, including its
prompt definitions, score normalization, and aggregation. We thank the authors
of these projects for releasing their code and models. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for code attribution and licenses.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and test instructions,
and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community guidelines.

## License

UnStep is licensed under [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](LICENSE).
Third-party dependencies and model weights retain their own licenses.
