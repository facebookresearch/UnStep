# Speed measurement

Complete [installation](../installation/), download the assets, and establish
the exported [generation environment](README.md#environment) first.
Run commands from the repository root on an otherwise idle GPU.

## Run a probe

```bash
./generation/run_speed_probe.sh unstep_speed
```

The probe uses the first GPU in `CUDA_VISIBLE_DEVICES` (GPU 0 if unset), prompt
indices 0-19, and automatic warmup before measurement. It does not
retain MP4s or score videos. Detailed timings and logs are saved under
`results/<RUN_NAME>`. `warm_mean_excluding_first` excludes the first measured
prompt; `raw_mean` includes all measured prompts.

The synchronized timer includes DiT generation and its KV updates, VAE decoding
and VAE-cache resets, then RGB normalization, clamping, and channel ordering.
Initial-noise sampling, KV/cross-attention cache allocation, and text/reference
encoding happen before the timer. Conversion to uint8, the GPU-to-CPU copy,
video writing, and VBench scoring are outside it. GB200 FA2 uses the same timer:

```bash
UNSTEP_ATTENTION_BACKEND=fa2 ./generation/run_speed_probe.sh gb200_speed
```

This requires the [GB200 FA2 installation](../installation/flash_attention.md#gb200-fa2).
To check a single prompt:

```bash
PROMPT_INDICES=0 ./generation/run_speed_probe.sh one_prompt
```

`PROMPT_INDICES` accepts a sorted comma-separated list or `all`.

The probe accepts the same [checkpoint and method options](README.md#method-settings)
as saved-video generation. Use a new run name for each measurement and run probes
serially. Fresh compiler caches and warmup are handled automatically.
