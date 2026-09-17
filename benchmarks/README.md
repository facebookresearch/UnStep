# Benchmarks and ablations

Complete [installation](../installation/), download the assets, and establish
the exported [generation environment](../generation/#environment) first.
Run commands from the repository root. Report the GPU model, attention backend,
software versions, method overrides, and warmup/cache settings with each result.

## Speed measurement

```bash
./benchmarks/run_speed_probe.sh unstep_speed
```

The probe uses the first GPU in `CUDA_VISIBLE_DEVICES` (GPU 0 if unset), prompt
indices 0-19, and one separate warmup process before measurement. It does not
retain MP4s or score videos. Detailed timings and logs are saved under
`results/<RUN_NAME>`. `warm_mean_excluding_first` and `min_after_first` exclude
the first measured prompt; `raw_mean` includes it and `steady_tail_mean` uses
the last three available measurements. Empty filtered statistics print `n/a`.

The default timed region is post-text-encoding DiT generation and full VAE
decoding, including final RGB clamp/layout conversion. Text encoding, video
writing, and VBench scoring are outside it. Defaults and the timer itself are
the same on the GB200 FA2 path:

```bash
UNSTEP_ATTENTION_BACKEND=fa2 ./benchmarks/run_speed_probe.sh gb200_speed
```

This requires the Blackwell-compatible FA2 environment described in generation;
it does not assume an H100 FA3 overlay. A single-prompt smoke probe is valid:

```bash
PROMPT_INDICES=0 WARMUP_PROMPT_INDICES=0 WARMUP_REPEATS=1 \
./benchmarks/run_speed_probe.sh one_prompt
```

| Probe control | Default |
|---|---|
| `PROMPT_INDICES` | Sorted comma list `0,1,...,19`; `all` also accepted |
| `WARMUP_PROMPT_INDICES` | Same list as `PROMPT_INDICES` |
| `WARMUP_REPEATS` | `1` separate process; non-negative counts supported |
| `DISABLE_GC_DURING_MEASUREMENT` | `1` |
| `EXCLUDE_FINAL_OUTPUT_LAYOUT_TIMING` | `0`; setting `1` changes the reported timing boundary |
| `UNSTEP_CPU_THREADS_PER_PROCESS` | `32` |
| `TORCHINDUCTOR_COMPILE_THREADS` | `32` |

The probe uses one logical shard, seed 0, `sequential_global`, and `initial_only`.
Arbitrary subsets, gaps, or changed logical partitioning are not bitwise sample
reproductions of the full eight-shard run. Use full generation for scoring.

For temporary encoding checks, `UNSTEP_EXTRA_ARGS='--save-videos'` enables
debug saves under `/tmp/unstep_wrapper_generation_outputs/<RUN_NAME>/indexed_videos`.
The runner deletes these videos after each probe process, including failed runs;
this is not a keep-videos switch. Use generation's subset example for retained MP4s.

## Configuration ablations

Pass explicit changes through `UNSTEP_EXTRA_ARGS`. For example, change only
the clean-cache noise level:

```bash
UNSTEP_EXTRA_ARGS='--clean-sigma-override 0.01' \
./generation/run.sh sigma001
```

The default is 0.02 for both clean-input re-noising and the emitted prediction.
The [method table](../generation/#method-settings) distinguishes supported flags
from config-only SVD/attention/schedule settings. See the adjacent override
section for whitespace splitting, precedence, and rejected launcher-owned flags.
Use a separate run name for each configuration, then follow evaluation after
full generation. The runner help is broader than launcher-supported extras:

```bash
PYTHONPATH="$PWD/runtime:$PWD${UNSTEP_EXTRA_PYTHONPATH:+:$UNSTEP_EXTRA_PYTHONPATH}" \
"$UNSTEP_PYENV/bin/python" -m unstep.run_wrapper_generation --help
```

## Fresh caches and staging

The full default protocol uses all 946 extended prompts, eight contiguous shards,
seed 0, `sequential_global`, and `initial_only`. Generation defaults to
`UNSTEP_CACHE_MODE=isolated` and `UNSTEP_CACHE_SCOPE=run_gpu`: fresh per-run GPU
cache directories, one separate warmup per GPU, then source/cache reuse across
that GPU's shards. The cache-key tag remains shard-specific. This is not one
separate warmup per shard. `UNSTEP_CACHE_SCOPE=shard` is a nondefault alternative.

`UNSTEP_CLEAR_GLOBAL_RUNTIME_CACHES=0` now leaves unrelated jobs' user-level
compiler caches intact. Isolated runs already select their own local cache
directories and disable remote compiler caches. Opt in to a global purge with
`UNSTEP_CLEAR_GLOBAL_RUNTIME_CACHES=1` only when no other jobs use those caches.
Fresh isolated caches are **not a guarantee of bitwise reproducibility** across
kernel choices, hardware, libraries, or nondeterministic operations.

`UNSTEP_CACHE_MODE=shared` is an explicit warm-cache alternative, with per-shard
state under `UNSTEP_SHARED_RUNTIME_STATE_DIR` (default
`.runtime_state/full946_generation`) and source staging under `UNSTEP_WORK_ROOT`
(default `/tmp/unstep_full946_generation`). Existing warmup markers skip separate
warmup. Shared runs must not overlap and must not be reported as fresh-cache runs.

The probe always isolates its compiler state under
`UNSTEP_RUN_STATE_ROOT` (default `results/<RUN_NAME>/runtime_state`). Existing state
is refused unless `UNSTEP_REUSE_RUNTIME_STATE=1`; use a new name for fresh results.
Its default `UNSTEP_WORK_ROOT` is `${TMPDIR:-/tmp}/unstep_speed_probe_stage`, held
under an exclusive `flock` for the entire probe. Concurrent probes must use
different dedicated work roots, for example:

```bash
CUDA_VISIBLE_DEVICES=1 UNSTEP_WORK_ROOT=/tmp/unstep_probe_gpu1 \
./benchmarks/run_speed_probe.sh speed_gpu1
```

Output/cache/work overrides must not overlap source, assets, Python/CUDA
installations, or broad roots; work directories cannot contain outputs or caches.
Successful generation with the default cache scope removes its temporary source
staging, but keeps run JSONs/logs/caches. Probe measurement removes its staged
source and releases the lock. A failed run can leave isolated state for diagnosis.

Direct Python launches also refuse existing work directories unless
`--reuse-workdir` is explicit and its staging audit matches. Failed validation
does not delete that directory. Source directory symlinks are rejected; point to
the real directory (without linked subdirectories) or an upstream source ZIP.

For correctness checks, see [tests](../tests/).
