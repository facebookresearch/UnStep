# Evaluation

Run commands from the repository root. Complete [installation](../installation/)
first, prepare scoring dependencies/models below, then run [generation](../generation/)
and scoring as separate stages. The scorer uses the original VBench prompts and
native dimension filtering; duplicate prompt names share the first indexed video.

## 1. Select the environment

Reuse the same exported Python, CUDA, asset, and optional output paths as
generation. Activate that Python environment for `pip` and download commands:

```bash
export UNSTEP_PYENV="${UNSTEP_PYENV:-$PWD/.venv}"
export UNSTEP_ASSETS_ROOT="${UNSTEP_ASSETS_ROOT:-$PWD/assets}"
source "$UNSTEP_PYENV/bin/activate"
```

For CUDA wheels installed inside this virtualenv, restore the library/build
environment now; skip this for a system CUDA installation:

```bash
source installation/cuda_wheels.sh
```

For a separate wheel overlay, retain the installation's exported
`UNSTEP_CUDA_OVERLAY` and `UNSTEP_CUDA="$UNSTEP_CUDA_OVERLAY/nvidia/cu13"` instead.
For system CUDA, set `CUDA_HOME` to that installation. Then select the scoring paths:

```bash
export UNSTEP_CUDA="${UNSTEP_CUDA:-${CUDA_HOME:?Set CUDA_HOME or UNSTEP_CUDA first}}"
export CUDA_HOME="$UNSTEP_CUDA"
export UNSTEP_VBENCH_SRC="${UNSTEP_VBENCH_SRC:-$UNSTEP_ASSETS_ROOT/source/VBench}"
export UNSTEP_VBENCH_CACHE="${UNSTEP_VBENCH_CACHE:-$UNSTEP_ASSETS_ROOT/weights/vbench}"
export UNSTEP_VBENCH_PYTHONPATH="${UNSTEP_VBENCH_PYTHONPATH:-$PWD/.vbench_deps}"
export UNSTEP_VBENCH_DEPS="$UNSTEP_VBENCH_PYTHONPATH"
```

The selected installation must already import its source-built PyTorch with the
CUDA library environment established during installation. Do not export scoring
dependencies into the generation `PYTHONPATH`; the scorer adds them only to its
own workers. Changing an inline assignment for one command does not change later
commands' paths.

## 2. Install the scoring overlay

Use a fresh overlay directory. Archive an existing overlay explicitly before
rebuilding it; do not mix it with an earlier dependency-resolved installation.

```bash
test ! -e "$UNSTEP_VBENCH_PYTHONPATH" && \
"$UNSTEP_PYENV/bin/python" -m pip install --no-deps --no-build-isolation \
  --target "$UNSTEP_VBENCH_PYTHONPATH" -r evaluation/requirements.txt
```

`--no-deps` is required. An ordinary `pip --target -r` recursively installs a
second Torch/Torchvision stack through scoring packages, which then shadows the
source-built generation stack. `requirements.txt` explicitly includes scoring
runtime additions and their numeric/image/plotting dependencies inventoried from
the legacy scoring overlay; shared imports come from the generation requirements.
It does not install PyIQA training tools or all optional PyIQA models.

This is a **legacy VBench inference environment**, not a dependency-resolver-clean
general PyIQA environment. Intentional metadata exceptions are:

| Package metadata | Legacy runtime choice |
|---|---|
| PyIQA 0.1.14.1 requests `timm>=0.8` | Native VBench uses `timm==0.6.13` |
| PyIQA requests Transformers 4.37.2 and Accelerate <=1.1.0 | Shared stack uses Transformers 4.57.1 and Accelerate 1.10.1; the scoring entrypoint applies the legacy Transformers shim |
| Detectron2 requests `iopath<0.1.10` | Scoring overlay retains 0.1.10 |
| LVIS requests `opencv-python` | Use the headless OpenCV distribution only; both provide `cv2` |
| OpenAI CLIP imports legacy `pkg_resources` | Scoring overlay pins `setuptools==80.9.0`, which still supplies it |

Keep this setuptools override in the scoring overlay only. Generation retains
setuptools 84, which does not provide `pkg_resources`; without the overlay pin,
CLIP and the VBench dimensions importing it fail before scoring. Do not downgrade
the generation environment to fix these imports.

Unused PyIQA/Detectron2 training/development and optional-metric dependencies can also be
reported missing by `pip check`; this file is scoped to VBench's MUSIQ/inference
path, not those features. Do not fix these reports by allowing pip to upgrade
Torch or timm. In particular, do not add `open_clip_torch==3.2.0`: it requests
`timm>=1.0.17` and is not the native VBench CLIP implementation used here.
Fresh-install import checks and actual scoring remain necessary to validate a
new platform; pinned metadata alone does not establish binary/kernel compatibility.

Build Detectron2 after the selected PyTorch, CUDA, and scoring overlay are ready:

```bash
git clone --depth 1 https://github.com/facebookresearch/detectron2.git .build/detectron2
./evaluation/install_detectron2.sh .build/detectron2
```

The installer derives architecture targets from visible GPUs and imports the
compiled `detectron2._C` extension afterward. It does not default to Hopper SM90.
For an explicit GB200 build, or a build host without visible GPUs:

```bash
TORCH_CUDA_ARCH_LIST=10.0 ./evaluation/install_detectron2.sh .build/detectron2
```

`MAX_JOBS` defaults to 4. `TORCH_CUDA_ARCH_LIST` overrides detection; include every
architecture on which the extension will run. This architecture selection has
CPU-stub coverage, not a claim that a full GB200 scoring run was performed.
The generation-side GB200 setting is `UNSTEP_ATTENTION_BACKEND=fa2`; scoring itself
does not require the generation FA3 overlay. Upstream VBench source is not patched.

## 3. Prepare offline scoring models

Download the standard models using the upstream files under
`$UNSTEP_VBENCH_SRC/pretrained/` and place/cache them under `$UNSTEP_VBENCH_CACHE`:

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

The cache must contain the CLIP/DINO, aesthetic/MUSIQ, AMT/RAFT, GRiT/DenseCap,
Tag2Text, UMT/ViCLIP and BERT files required by upstream VBench. Preserve upstream
checkpoint names and directory layouts. BERT tokenizer/config files use the
Hugging Face cache layout, not loose files:

```bash
huggingface-cli download bert-base-uncased \
  --cache-dir "$UNSTEP_VBENCH_CACHE" \
  --include config.json tokenizer.json tokenizer_config.json vocab.txt special_tokens_map.json
```

ViCLIP also needs its BPE tokenizer, separate from the model checkpoint. While
online, pre-cache the exact upstream filename and check the downloaded archive:

```bash
export VBENCH_CACHE_DIR="$UNSTEP_VBENCH_CACHE"
mkdir -p "$VBENCH_CACHE_DIR/ViCLIP"
wget -O "$VBENCH_CACHE_DIR/ViCLIP/bpe_simple_vocab_16e6.txt.gz" \
  https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz
gzip -t "$VBENCH_CACHE_DIR/ViCLIP/bpe_simple_vocab_16e6.txt.gz"
```

`evaluation/run.sh` exports `VBENCH_CACHE_DIR` from `UNSTEP_VBENCH_CACHE` for its
workers. Standalone VBench import probes must export it before importing VBench
as above; otherwise upstream defaults to `~/.cache/vbench`. Importing dimensions
such as temporal style can trigger the tokenizer's direct `wget` download if the
file is missing. Hugging Face offline flags do not disable that downloader.

The scorer sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and
`--load_ckpt_from_local True`; missing models must be obtained before scoring.
Native dimension filtering and duplicate prompts do not imply missing videos.

## 4. Generate, score, merge, and retain

After completing the preceding setup, keep these stages separate:

```bash
./generation/run.sh unstep_vbench
```

Only after generation completes successfully:

```bash
./evaluation/run.sh unstep_vbench
```

Scoring starts only after all indexed videos exist. It runs the two dimension
groups concurrently on the first two entries of `CUDA_VISIBLE_DEVICES` (physical
0 and 1 if unset), using separate logs and distributed ports. It requires two
visible GPUs even if generation used a different GPU count. It does not accept
generation subsets. Failed workers stop/reap their sibling; merging runs only
after both succeed. Nonfinite or missing dimension scores fail the merge.

Outputs are under `results/<RUN_NAME>/vbench_score`: `dim0/`, `dim1/`, `logs/`,
and `<RUN_NAME>_table1_score.json` / `.md`. Official Quality/Semantic/Total
aggregation is unchanged. Historical weighted Norm16 is an **auxiliary** metric,
not an unweighted mean: its dynamic-degree contribution is halved. The JSON
exposes `vbench_historical_weighted_norm16` and retains the misnamed legacy key
`vbench_unweighted_normalized_16dim_mean` with the same historical value.

By default, successful scoring **and** merge delete generated MP4s. To retain
them, use this command instead of the scoring command above:

```bash
UNSTEP_DELETE_VIDEOS_AFTER_SCORE=0 ./evaluation/run.sh unstep_vbench
```

Failures retain videos and diagnostic outputs. An existing `vbench_score`
directory is always refused, even after failure. For a retry, preserve/archive
that directory explicitly, ensure the complete videos still exist, then rerun.
A successful default run has deleted its videos; it cannot simply be rescored.

`UNSTEP_OUT_DIR` overrides the shared run root; `UNSTEP_VIDEO_DIR` overrides the
input video directory. Use dedicated directories outside protected source/assets
and environment paths. The deletion setting applies to the selected video
directory, including its MP4 descendants. `UNSTEP_VBENCH_SRC`,
`UNSTEP_VBENCH_CACHE`, and `UNSTEP_VBENCH_PYTHONPATH` override source, model cache,
and scoring dependencies. `UNSTEP_EXTRA_PYTHONPATH` adds other dependencies;
avoid unrelated sitecustomize modules or alternative Torch installations.

## Other upstream benchmarks

I2V and long-video benchmarks are not modes of `evaluation/run.sh`.
Follow `$UNSTEP_VBENCH_SRC/vbench2_beta_i2v/README.md` for I2V; its data helper is:

```bash
python -m pip install gdown
(
  cd "$UNSTEP_VBENCH_SRC"
  sh vbench2_beta_i2v/download_data.sh
)
```

It downloads `i2v-bench-info.json`, `crop.zip`, and `origin.zip` and unpacks the
image suite. For VBench-long, follow `vbench2_beta_long/README.md` upstream.
Keep external benchmark data/models in local asset/cache paths and do not commit
them. CoTracker and other experiment-specific dependencies/models belong to
those upstream workflows, not the default 16-dimension launcher.
