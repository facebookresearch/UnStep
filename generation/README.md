# Generation

Generate at 832 x 480 with batch size one. The same launcher supports
five-second T2V (default), [five-second I2V](i2v.md), and [long T2V](long_video.md).
It records speed as part of generation. Installation and scoring are separate steps.

## Environment

Complete [installation](../installation/) and download the
[model files and prompts](../installation/assets.md) first. Use Bash on Linux
with `realpath` and `flock` available. Run commands from the repository root.
Export shared settings for generation and evaluation:

```bash
export UNSTEP_PYENV="${UNSTEP_PYENV:-$PWD/.venv}"
export UNSTEP_ASSETS_ROOT="${UNSTEP_ASSETS_ROOT:-$PWD/assets}"
source "$UNSTEP_PYENV/bin/activate"
```

For CUDA wheels installed inside this virtualenv, restore the build/library
environment now (skip this for a system CUDA installation):

```bash
source installation/cuda_wheels.sh
```

For a separate wheel overlay, retain the installation's exported
`UNSTEP_CUDA_OVERLAY` and `UNSTEP_CUDA="$UNSTEP_CUDA_OVERLAY/nvidia/cu13"` instead.
For system CUDA, set `CUDA_HOME` to that installation. Then finish shared setup:

```bash
export UNSTEP_CUDA="${UNSTEP_CUDA:-${CUDA_HOME:?Set CUDA_HOME or UNSTEP_CUDA first}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0,1}"
```

`UNSTEP_ATTENTION_BACKEND=auto` is the default: Hopper selects FA3;
GB200 uses FA2. For an explicit GB200 launch, install a Blackwell-compatible
FA2 build in the selected environment and set:

```bash
unset UNSTEP_FA3
export UNSTEP_ATTENTION_BACKEND=fa2
```

`UNSTEP_FA3` is only needed for FA3 installed in a separate directory on Hopper.

## Generate and then evaluate

```bash
./generation/run.sh unstep_vbench
```

This saves all 946 indexed MP4s under `results/unstep_vbench/indexed_videos`,
with timing JSONs and logs in `generation_jsons` and
`generation_logs`. Video writing is outside the generation timer. Generation
does not score or delete those videos. Follow [evaluation](../evaluation/)
only after generation succeeds; evaluation deletes MP4s after successful
scoring/merge unless `UNSTEP_DELETE_VIDEOS_AFTER_SCORE=0` is set.

Generation uses two GPUs by default. Select devices with `CUDA_VISIBLE_DEVICES`
and their count with `UNSTEP_NUM_GPUS`.

Use a new, single-component run name for every comparison. An existing output
directory is refused. `UNSTEP_OVERWRITE_OUTPUT=1` explicitly deletes its contents;
archive anything needed first. `UNSTEP_OUT_DIR` selects a dedicated output
directory; within the repository, use a directory below `results/`.
Export this override for evaluation too.

## Subsets and speed-only runs

For timing without retained videos, use the [speed probe](speed.md).
For a small saved-video run:

```bash
UNSTEP_NUM_SHARDS=1 UNSTEP_NUM_GPUS=1 UNSTEP_PROMPT_INDICES=0,1 \
./generation/run.sh subset_videos
```

Use sorted, comma-separated prompt indices with this subset command.
Subsets are for checking generation; evaluation requires the complete prompt set.

## Task and inputs

| Control | Default / purpose |
|---|---|
| `UNSTEP_TASK` | `t2v`; choose `i2v` or `long` for the other tasks |
| `UNSTEP_PROMPT_URI` | Extended T2V prompts, VBench-I2V JSON, or the 40 long-video prompts, selected by task |
| `UNSTEP_IMAGE_ROOT` | I2V reference directory; defaults to VBench's `data/crop/16-9` |
| `UNSTEP_PROMPT_INDICES` | `all`; see the subset command above for selected prompts |
| `CUDA_VISIBLE_DEVICES` | GPU IDs to use; full generation defaults to `0,1` |
| `UNSTEP_NUM_GPUS` | `2` workers for saved-video generation; the speed probe uses one |
| `UNSTEP_OUT_DIR` | `results/<RUN_NAME>`; use a dedicated directory |
| `UNSTEP_ATTENTION_BACKEND` | `auto`: FA3 on H100, FA2 on GB200 |

The [I2V](i2v.md) and [long-video](long_video.md) guides give exact generation
and evaluation commands. Their evaluators differ from standard T2V VBench.

## Method settings

Pass method options after the run name to the same generation command.
Omit options to use the default two-step method. No Python edits are needed.
Run `./generation/run.sh --help` for the command-line list.

| Setting | Default and meaning | Control |
|---|---|---|
| Model checkpoint | Self Forcing DMD checkpoint from the installed assets | `--checkpoint-uri /path/to/model.pt` |
| Long T2V duration | 123 latent frames produce 489 RGB frames (30.56 seconds at 16 FPS) | `--num-latent-frames INT`, a positive multiple of three; long task only |
| Later-block step count | Two denoising steps; first block always keeps four; every block also performs the clean pass | `--steps 1` or `--steps 2` |
| Clean sigma | `0.02` for both clean-input re-noising and emitted flow-to-x0 prediction; DiT label remains `t=0` | `--clean-sigma-override FLOAT`, finite and non-negative |
| Chunk schedules | First chunk `(0,1,2,3)`; later `(0,3)`; zero-based chunks 3, 4, 5 use `(0,2)`; each appends the clean step | `--disable-block-schedule-overrides` removes only the chunk 3/4/5 overrides |
| SVD projections | V and O only; layers 0-7 retain full rank; layers 8-29 retain rank fraction `0.92`, not 92% singular-value energy | `--svd-keep-ratio FLOAT`, in `(0, 1]`; `1` disables truncation |
| Attention budget | Default buffer is 9 latent frames, including the sink, with the per-layer mixture below | `--attention-window INT` selects a **uniform** budget for all layers; omit it to keep the default mixture |
| Sink | Three preserved latent frames within the attention budget | `--sink-size INT`; window must fit the sink plus a generated block |
| Default per-layer attention | Layers 8-29 read the last 8 frames, except layers 8, 15, 23, 29 read 9; layers 0-7 read 9 | Retained when `--attention-window` is omitted |

Schedule tuples are indices into the model's timestep list, not raw timestep
labels. The 8/9 **layer** mixture is separate from the **chunk** schedule mixture.
Once the cache is full, an eight-frame read uses two sink frames plus six recent
frames (2+6); a nine-frame read uses three plus six (3+6), not 9+3.

### Examples

```bash
# One-step generation with the installed checkpoint.
./generation/run.sh unstep_one_step --steps 1

# A different compatible checkpoint, keeping the default two-step method.
./generation/run.sh unstep_checkpoint --checkpoint-uri /path/to/model.pt

# Explicit method settings with uniform nine-frame attention.
./generation/run.sh unstep_uniform_window --steps 2 --clean-sigma-override 0.02 \
  --svd-keep-ratio 0.92 --attention-window 9 --sink-size 3
```

`--steps 1` removes the block-specific two-step schedules as well as the second
denoising step; it retains the clean pass. Sigma applies equally to re-noising
and emission. The effective settings are recorded in each generation JSON.

## Checkpoint selection

`--checkpoint-uri` replaces the DMD checkpoint with a local compatible
Self-Forcing/Wan2.1 state dict; the base Wan/text/VAE assets are still required.
It prefers `generator_ema`, then `generator`, then a raw state dict.

## Code layout

| File | Role |
|---|---|
| [`unstep/wrapper.py`](../unstep/wrapper.py) | Method configuration, schedules, SVD, generation, and decode orchestration |
| [`unstep/wrapper_runtime.py`](../unstep/wrapper_runtime.py) | Upstream loading and staging, attention/RoPE kernels, scheduler conversions, and VAE helpers |
| [`unstep/clean_sigma.py`](../unstep/clean_sigma.py) | Shared clean-input sigma override and startup checks against the upstream scheduler |
| [`unstep/video_tasks.py`](../unstep/video_tasks.py) | I2V reference conditioning and long-video decoder compilation; not loaded for five-second T2V |
| [`unstep/run_wrapper_generation.py`](../unstep/run_wrapper_generation.py) | Python entrypoint, inputs, timing, video export, and run records |
| [`runtime/sitecustomize.py`](../runtime/sitecustomize.py) | Interpreter-startup settings loaded by the generation and speed launchers, including cuDNN benchmark limit 32 |
| [`generation/run.sh`](run.sh) | Full or subset generation, GPU scheduling, warmup, and per-run cache setup |
| [`generation/run_speed_probe.sh`](run_speed_probe.sh) | Generation timing without retaining videos or running VBench |
| [`generation/path_safety.sh`](path_safety.sh) | Shared path checks used by generation, speed measurement, and evaluation; not a standalone command |

[`pyproject.toml`](../pyproject.toml) installs the `unstep` Python package.
Keep the checkout for the shell launchers and runtime startup module; installing
the Python package alone does not install the complete workflow.
