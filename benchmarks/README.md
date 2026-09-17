# Benchmarks and ablations

Complete [installation](../installation/) first. Run commands from the
repository root. Report the GPU model alongside each speed result.

## Speed measurement

```bash
./benchmarks/run_speed_probe.sh unstep_speed
```

The probe uses prompt indices 0–19 by default and runs one separate warm-up
process before the measurement process. `PROMPT_INDICES`,
`WARMUP_PROMPT_INDICES`, and `WARMUP_REPEATS` control those choices.
The printed summary excludes the first measured prompt. Detailed timings
and logs are saved under `results/<RUN_NAME>`.

The timed region includes DiT generation and VAE decoding. Video writing and
VBench scoring are outside that region. Each new run uses its own compiler
cache directories, with remote compiler caches disabled.

The environment and asset path overrides are the same as for
[generation](../generation/).

## Configuration ablations

Pass explicit changes through `UNSTEP_EXTRA_ARGS`. For example, change only
the clean-cache noise level:

```bash
UNSTEP_EXTRA_ARGS='--clean-sigma-override 0.01' \
./generation/run.sh sigma001
```

The default is 0.02 for both clean-input re-noising and the emitted prediction.
Use a separate run name for every configuration, then follow the
[evaluation instructions](../evaluation/). The available wrapper options are
listed by:

```bash
python -m unstep.run_wrapper_generation --help
```

## Reproduction records

The default generation protocol uses all 946 extended prompts, eight contiguous
shards, seed 0, `sequential_global` seeding, and `initial_only` RNG skipping.
Per-shard work directories are recreated before generation, and compiler caches
are isolated by default.

- [ENVIRONMENT_LOCK.json](ENVIRONMENT_LOCK.json) records the reference software
  stack, attention build, runtime settings, and source hashes.
- [SOURCE_HASHES.txt](SOURCE_HASHES.txt) records the same source hashes in a
  simple text format. Paths in both files are relative to the repository root.

These files are records, not installation scripts. For correctness checks, see
[tests](../tests/).
