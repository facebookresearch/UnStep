# Evaluation

Run commands from the repository root. Complete [installation](../installation/)
first, prepare scoring dependencies/models below, then run [generation](../generation/)
and scoring as separate stages. The scorer uses the original VBench prompts
and dimension definitions.

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
own workers.

## 2. Install scoring dependencies

Install scoring packages in a separate directory, called an overlay. The scorer
loads this directory; generation does not. Use a new directory or archive an
existing one before rebuilding it.

```bash
test ! -e "$UNSTEP_VBENCH_PYTHONPATH" && \
"$UNSTEP_PYENV/bin/python" -m pip install --no-deps --no-build-isolation \
  --target "$UNSTEP_VBENCH_PYTHONPATH" -r evaluation/requirements.txt
```

`--no-deps` prevents pip from installing a second Torch/Torchvision stack over
the generation build. The requirements list the packages needed by VBench's
inference metrics, not PyIQA's training tools or optional models. Shared packages
come from the generation environment. These compatibility choices are intentional:

| Package requirement | VBench runtime choice |
|---|---|
| PyIQA 0.1.14.1 requests `timm>=0.8` | Native VBench uses `timm==0.6.13` |
| PyIQA requests Transformers 4.37.2 and Accelerate <=1.1.0 | Uses Transformers 4.57.1 and Accelerate 1.10.1; the scoring entrypoint supplies the older Transformers import locations |
| Detectron2 requests `iopath<0.1.10` | Scoring overlay retains 0.1.10 |
| LVIS requests `opencv-python` | Use the headless OpenCV distribution only; both provide `cv2` |
| OpenAI CLIP imports legacy `pkg_resources` | Scoring overlay pins `setuptools==80.9.0`, which still supplies it |

Keep this setuptools override in the scoring overlay only. Generation retains
setuptools 84, which does not provide `pkg_resources`; without the overlay pin,
CLIP and the VBench dimensions importing it fail before scoring. Do not downgrade
the generation environment to fix these imports.

`pip check` can also report missing packages for unused training or optional
metrics. Do not resolve these by upgrading Torch or timm. VBench uses OpenAI
CLIP here, not `open_clip_torch`, which requires a different timm version.
Validate imports and scoring on each new platform.

Build Detectron2 after the selected PyTorch, CUDA, and scoring overlay are ready:

```bash
git clone --depth 1 https://github.com/facebookresearch/detectron2.git .build/detectron2
./evaluation/install_detectron2.sh .build/detectron2
```

The installer detects the visible GPU architectures and checks that the compiled
`detectron2._C` extension imports.
For an explicit GB200 build, or a build host without visible GPUs:

```bash
TORCH_CUDA_ARCH_LIST=10.0 ./evaluation/install_detectron2.sh .build/detectron2
```

`MAX_JOBS` defaults to 4. `TORCH_CUDA_ARCH_LIST` overrides detection; include every
architecture on which the extension will run. Validate the build on the target
GPU; these commands have not been verified by a full GB200 scoring run.
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

The scorer uses `UNSTEP_VBENCH_CACHE` and loads models locally. Complete all
downloads before scoring.

## 4. Generate, score, merge, and retain

After completing the preceding setup, keep these stages separate:

```bash
./generation/run.sh unstep_vbench
```

Only after generation completes successfully:

```bash
./evaluation/run.sh unstep_vbench
```

Scoring requires the complete video set and two visible GPUs. It uses the first
two entries of `CUDA_VISIBLE_DEVICES` (0 and 1 if unset), then merges their results
only if both succeed and all dimension scores are valid. Subsets are not accepted.

Outputs are under `results/<RUN_NAME>/vbench_score`: `dim0/`, `dim1/`, `logs/`,
and `<RUN_NAME>_table1_score.json` / `.md`. Official Quality/Semantic/Total
aggregation is unchanged. Norm16 is an auxiliary normalized mean with half
weight for dynamic degree.

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
and scoring dependencies.

## I2V and long video

These tasks use their own upstream evaluators, not `evaluation/run.sh` or its
16-dimension T2V aggregation:

- [Image-to-video](../generation/i2v.md#evaluation): VBench-I2V, with 1,118
  image-prompt pairs, nine component metrics, and Quality/I2V/Total aggregates.
- [Long video](../generation/long_video.md#evaluation): the VBench++ 40-prompt
  competition track, with eight component metrics and Temporal Quality,
  Framewise Quality, and Text Alignment aggregates.

The guides include dataset/model requirements and upstream evaluation commands.
Their additional dependencies are not all supplied by this T2V scoring overlay;
prepare a separate evaluation environment following the linked upstream setup.
Do not let those installs replace the generation environment's PyTorch stack.
Keep datasets, model weights, videos, and generated scoring clips outside Git.

## Files

- [run.sh](run.sh) validates the video collection, launches both scoring workers,
  merges their results, and applies the requested video-retention policy.
- [vbench_eval_entry.py](vbench_eval_entry.py) supplies compatibility helpers
  before invoking upstream VBench; `run.sh` calls it for each worker.
- [merge_vbench_scores.py](merge_vbench_scores.py) combines the workers' dimension
  results and computes the reported aggregates; `run.sh` invokes it after scoring.
- [requirements.txt](requirements.txt) and [install_detectron2.sh](install_detectron2.sh)
  provide the scoring dependencies described above. They are installation steps,
  not additional scoring commands.
