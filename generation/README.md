# Generation

## Environment

Complete [installation](../installation/) and download the
[model files and prompts](../installation/assets.md) first. Use Bash on Linux
with `realpath` and `flock` available. Run commands from the repository root.
Export shared settings so generation and the later evaluation use the same
environment and assets; command-local `NAME=value ./script` assignments do not
persist into the next command.

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

The launchers set their child processes' CUDA library paths themselves.

`UNSTEP_ATTENTION_BACKEND=auto` is the default: Hopper selects FA3;
GB200 uses FA2. For an explicit GB200 launch, install a Blackwell-compatible
FA2 build in the selected environment and set:

```bash
unset UNSTEP_FA3
export UNSTEP_ATTENTION_BACKEND=fa2
```

No H100 FA3 overlay is required on this path. `UNSTEP_FA3` is an optional
separate Hopper FA3 import directory. The other backend values, `h100` and
`flash_attn_3`, explicitly request the corresponding FA3 import layout; they
are not GB200 settings. Backend selection does not change the sampling method.
Check the generation JSON's runtime GPU capability and selected attention
backend when comparing machines. The thread defaults below are not a claim of
GB200-specific performance tuning.

## Generate and then evaluate

```bash
./generation/run.sh unstep_vbench
```

This saves all 946 indexed MP4s under `results/unstep_vbench/indexed_videos`,
with per-shard timing JSONs and logs in `generation_jsons` and
`generation_logs`. Video writing is outside the generation timer. Generation
does not score or delete those videos. Follow [evaluation](../evaluation/)
only after generation succeeds; evaluation deletes MP4s after successful
scoring/merge unless `UNSTEP_DELETE_VIDEOS_AFTER_SCORE=0` is set.

The full protocol uses seed 0, eight contiguous logical shards,
`sequential_global` seeding, and `initial_only` skipping of preceding prompts'
initial-noise draws. `UNSTEP_NUM_GPUS=2` schedules those shards in waves on the
first two entries of `CUDA_VISIBLE_DEVICES`, accepting numeric IDs or GPU UUIDs.
Changing GPU count changes scheduling, not the eight logical partitions.
Changing `UNSTEP_NUM_SHARDS` from 8 changes the sampling protocol and must be reported.

Use a new, single-component run name for every comparison. An existing output
directory is refused. `UNSTEP_OVERWRITE_OUTPUT=1` explicitly deletes its contents;
archive anything needed first. `UNSTEP_OUT_DIR` overrides the run directory and
must identify a dedicated output directory, not the repository, home, assets,
an input environment, or their ancestors. Source-tree output overrides are
limited to individual directories below `results/`; symlinks are resolved before
safety checks. Export this override for evaluation too.

## Subsets and speed-only runs

For timing without retained videos, use the [speed probe](../benchmarks/).
For a small saved-video smoke run:

```bash
UNSTEP_NUM_SHARDS=1 UNSTEP_NUM_GPUS=1 UNSTEP_PROMPT_INDICES=0,1 \
./generation/run.sh subset_videos
```

`UNSTEP_PROMPT_INDICES` defaults to `all`; `shard_first` saves one prompt per
logical shard. Explicit comma lists must be sorted and require one logical
shard. These are smoke/subset runs, not full-protocol score reproductions.
The evaluator requires the complete indexed prompt set and rejects subsets.

## Method settings

These are the current defaults, not additional command-line options:

| Setting | Default and meaning | Control |
|---|---|---|
| Clean sigma | `0.02` for both clean-input re-noising and emitted flow-to-x0 prediction | `--clean-sigma-override FLOAT` |
| Clean label / scheduler | DiT still receives `t=0`; scheduler shift is `5.0`; nonzero steps retain their scheduler sigmas | Fixed runtime contract |
| Chunk schedules | First chunk `(0,1,2,3)`; later `(0,3)`; zero-based chunks 3, 4, 5 use `(0,2)`; each appends the clean step | `--disable-block-schedule-overrides` removes only the chunk 3/4/5 overrides |
| SVD projections | V and O only; layers 0-7 retain full rank; layers 8-29 retain rank fraction `0.92`, not 92% singular-value energy | `WrapperConfig` only, no SVD CLI flag |
| Attention budgets | Maximum buffer is 9 latent frames, including sink history; sink buffer size is 3 | `WrapperConfig` only |
| Per-layer attention | Layers 8-29 read the last 8 frames, except layers 8, 15, 23, 29 read 9; layers 0-7 read 9 | `WrapperConfig` only |
| Video layout | 21 latent frames in chunks of 3; 81 decoded frames; saved at 16 FPS | Fixed production defaults |
| Precision / compilation | BF16 transformer; FP16 full VAE decode; `max-autotune-no-cudagraphs` | `WrapperConfig` only |
| VAE runtime | First latent decoded separately, then cached blocks of 5 latent frames; channels-last-3D layout and preallocated block output | `WrapperConfig` only |

Schedule tuples are indices into the producer's timestep list, not raw timestep
labels. The 8/9 **layer** mixture is separate from the **chunk** schedule mixture.
After the upstream nine-frame buffer has filled and rolled over, its final
eight-frame slice contains sink slots 1 and 2 plus the latest six frames (2+6).
The nine-frame read contains all three sink slots plus the latest six (3+6),
not 9+3. Initial filling has less history; it is not always a full 8/9-frame read.

## Overrides and warmup

```bash
UNSTEP_EXTRA_ARGS='--clean-sigma-override 0.01' ./generation/run.sh sigma001
```

Both launchers accept `--clean-sigma-override`, `--checkpoint-uri`, and
`--disable-block-schedule-overrides` in `UNSTEP_EXTRA_ARGS`. The probe also
accepts `--save-videos` for temporary debug videos, which it deletes. Other
options, including launcher-owned backend, shard, seed, timing, work/output,
warmup, prompt and speed-only flags, are rejected before starting workers.
Use the documented environment controls instead. No shell evaluation occurs:
the string is split on whitespace, so embedded quotes do not support spaced
paths. Repeated supported value options use their last value. A nonempty string
replaces the default extra string and is passed to both warmup and measurement;
an unset or empty string selects `--clean-sigma-override 0.02`.

`--checkpoint-uri` replaces the DMD checkpoint with a local compatible
Self-Forcing/Wan2.1 state dict; the base Wan/text/VAE assets are still required.
It prefers `generator_ema`, then `generator`, then a raw state dict. Supported
prefixes are normalized; the selected key is recorded in `staging_audit.checkpoint_audit`.

The default is **one separate warmup per GPU**, using its first assigned shard's
first prompt. The prepared source and per-GPU caches are reused for later shards
on that GPU. `UNSTEP_FULL_WARMUP_RUNS=0` disables warmup; separate mode accepts
only 0 or 1. `UNSTEP_FULL_WARMUP_PROCESS=inline` enables an explicit alternative
with non-negative warmup counts per shard. `UNSTEP_FULL_WARMUP_PROMPT_INDICES`
defaults to `shard_first`; explicit lists follow the same single-shard rule.
These alternatives must be reported, not mixed into default timing results.

`UNSTEP_CPU_THREADS_PER_SHARD` and `TORCHINDUCTOR_COMPILE_THREADS` both default
to 32. `UNSTEP_SOURCE_URI`, `UNSTEP_WEIGHTS_URI`, `UNSTEP_VAE_URI`, and
`UNSTEP_PROMPT_URI` override individual local assets; use absolute paths.
`UNSTEP_FFMPEG` selects an executable, otherwise ImageIO discovers one.
`UNSTEP_EXTRA_PYTHONPATH` adds dependency directories. Advanced source/runtime
overrides are `UNSTEP_SRC` and `UNSTEP_SITE` (defaults: repository and `runtime/`).

See [benchmarks](../benchmarks/) for cache controls and measurement boundaries.
