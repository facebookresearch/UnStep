# Long video

Generate long T2V clips with the same checkpoint and method settings as five-second
T2V. The long-video adapter is in [video_tasks.py](../unstep/video_tasks.py).

## Generate

After [generation environment setup](README.md#environment), run:

```bash
UNSTEP_TASK=long ./generation/run.sh unstep_long
```

The default uses all 40 competition prompts and 123 latent frames, producing
489 RGB frames at 16 FPS (30.56 seconds). To select another duration:

```bash
UNSTEP_TASK=long ./generation/run.sh unstep_twenty_seconds --num-latent-frames 81
```

Latent length must be a positive multiple of three. Output length is
`4 * num_latent_frames - 3`; divide by 16 for seconds. This example produces
321 frames, or 20.06 seconds. The fixed-block compilation path is selected only
above 81 latent frames. Longer outputs still need more memory for latents and RGB.

Videos are saved under `results/<RUN_NAME>/indexed_videos` as `0001_video.mp4`,
`0002_video.mp4`, and so on, matching the upstream competition's one-based prompt
indices. Run JSONs retain the prompts, timings, and effective settings.
For timing without retaining videos:

```bash
UNSTEP_TASK=long ./generation/run_speed_probe.sh long_speed
```

The same duration and method flags apply; see [speed measurement](speed.md).

## Workload and method settings

Use the same model weights and environment as [T2V generation](README.md), with
[FA3 on H100 or FA2 on GB200](../installation/flash_attention.md). The paper's
long-video results use a single H100 per generated clip.

The [method settings](README.md#method-settings) are unchanged. Generation
continues with a bounded KV cache rather than stitching separate five-second
videos together.

## Long-video decoding

Above 81 latent frames, only the VAE decoder is compiled, rather than the
duration-dependent decode loop. This avoids excessive compilation memory.
Decoding uses blocks of up to five latent frames with a shared temporal cache
and preallocated RGB output. Output memory still grows with video duration.

The [timing boundaries](speed.md#run-a-probe) are the same as T2V. Exclude the
initial compilation/warmup clip from throughput, but evaluate all 40 videos.

## Evaluation

Use the [VBench++ competition evaluator](https://github.com/Vchitect/VBench/tree/master/competitions),
not the standard 946-prompt T2V evaluator. Forty prompts is the size of this
competition track, not a restriction on all VBench-Long evaluation.

Prepare a separate evaluation environment using upstream
`competitions/requirements.txt` and its setup instructions. Install PySceneDetect
and the FFmpeg executable, along with the upstream metric checkpoints. The
T2V overlay alone is not the full long-video environment. The VBench checkout
is obtained in [asset setup](../installation/assets.md).

Use one MP4 per prompt, with zero-padded one-based indices in filenames following
the competition convention, such as `0001_video.mp4`. Preserve prompt
order: upstream associates sorted video names with the prompt list. Keep each
run in its own directory; the evaluator creates `split_scene/` and `split_clip/`
beside the videos and can reuse existing splits.

With the evaluation environment activated, set absolute input/output paths:

```bash
: "${UNSTEP_VBENCH_SRC:?Set the absolute path to the VBench checkout}"
: "${UNSTEP_LONG_VIDEOS:?Set the absolute path to the 40 long videos}"
: "${UNSTEP_LONG_SCORES:?Set the absolute path for long-video score outputs}"
(
  cd "$UNSTEP_VBENCH_SRC/competitions"
  python run_eval.py \
    --video_path "$UNSTEP_LONG_VIDEOS" \
    --output_path "$UNSTEP_LONG_SCORES" \
    --prompt_file ./long_prompt_list.txt \
    --dimension temporal_quality frame_wise_quality text_alignment
)
```

Keep the `long_prompt_list.txt` filename: upstream uses it to select long-video
evaluation. The evaluator splits scenes into short clips and averages metrics
within each video, then across videos.

The output reports three aggregates and their eight component metrics:

- **Temporal Quality:** subject consistency, background consistency, motion
  smoothness, and dynamic degree, weighted 1, 1, 1, and 0.5.
- **Framewise Quality:** equal weights for aesthetic and imaging quality.
- **Text Alignment:** equal weights for overall video-text consistency and
  frame-text CLIP similarity.

Do not merge these into the standard T2V Total/Quality/Semantic score.

[Back to the main README](../README.md#long-video).
