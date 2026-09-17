# Generation

Complete [installation](../installation/) and download the
[model files and prompts](../installation/assets.md) first.
Run the launcher from the repository root:

```bash
./generation/run.sh unstep_vbench
```

The default run generates videos for all 946 extended VBench prompt entries.
It uses the two-step wrapper with clean-cache noise 0.02 and V/O rank retention
0.92. Videos are saved in `results/unstep_vbench/indexed_videos`, with timing
records and logs in the same run directory.

Use a new run name for each run. The launcher refuses an existing output
directory unless `UNSTEP_OVERWRITE_OUTPUT=1` is set.

## Existing environments and asset locations

The defaults use `.venv`, `assets/`, and the CUDA installation selected by
`CUDA_HOME`. Override them when needed:

```bash
UNSTEP_PYENV=/path/to/python-env \
UNSTEP_CUDA=/path/to/cuda \
UNSTEP_ASSETS_ROOT=/path/to/assets \
./generation/run.sh unstep_vbench
```

`UNSTEP_FA3` can point to a separate FA3 installation. Leave it unset if FA3
is installed in the Python environment. `UNSTEP_EXTRA_PYTHONPATH` adds any
separate dependency directories.

The default launcher splits prompts into eight contiguous shards and schedules
those shards across two GPUs. Set `UNSTEP_NUM_GPUS` to change the number of GPUs
used to run those shards.

Continue to [evaluation](../evaluation/) to score the videos. Timing details
and configuration overrides are documented in [benchmarks](../benchmarks/).
