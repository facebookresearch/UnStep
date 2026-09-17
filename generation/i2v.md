# Image-to-video

Generate five-second image-conditioned videos with the same checkpoint and
method controls as T2V. The adapter is in [video_tasks.py](../unstep/video_tasks.py).

## Data and setup

Use the environment and model assets described in [installation](../installation/)
and [assets](../installation/assets.md). The paper uses the same causal model
weights as T2V, not a separately trained I2V checkpoint. For attention, use
[FA3 on H100 or FA2 on GB200](../installation/flash_attention.md).

The [VBench-I2V image suite](https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v)
contains 355 reference images and 1,118 image-prompt entries: 246 subject entries,
109 background entries, and 763 camera-motion entries. Images are reused across
different prompts; these are not 1,118 distinct images.

Install the `unzip` executable. With `UNSTEP_VBENCH_SRC` set to the VBench
checkout from asset setup, run its downloader in the evaluation environment:

```bash
: "${UNSTEP_VBENCH_SRC:?Set the absolute path to the VBench checkout}"
python -m pip install gdown
(
  cd "$UNSTEP_VBENCH_SRC"
  sh vbench2_beta_i2v/download_data.sh
)
```

Use `vbench2_beta_i2v/data/crop/16-9/` for the reference crops and
`vbench2_beta_i2v/vbench2_i2v_full_info.json` for the prompt/dimension list.
The downloaded `i2v-bench-info.json` instead describes the source images and
crops. Each prompt entry identifies its reference through `image_name`.

## Generate

After [generation environment setup](README.md#environment), run:

```bash
UNSTEP_TASK=i2v ./generation/run.sh unstep_i2v
```

The launcher reads VBench-I2V's JSON and the 16:9 image directory from the
installed assets. `UNSTEP_PROMPT_URI` and `UNSTEP_IMAGE_ROOT` override these
paths. A custom JSON uses the same fields: `prompt_en` and `image_name`.
Prompts must be unique and usable as filenames; image names must refer to files
inside the reference directory.

Videos are saved in `results/unstep_i2v/indexed_videos` as `<prompt>-0.mp4`,
the upstream VBench naming convention. Run JSONs record the prompt,
reference path, video filename, timing, and effective method settings. This is
one video per image-prompt pair. For a speed-only run:

```bash
UNSTEP_TASK=i2v ./generation/run_speed_probe.sh i2v_speed
```

See [speed measurement](speed.md).

## Conditioning

The reference is resized to 832 x 480, encoded, and used to initialize the
attention cache. It remains the first latent frame; the model generates 20
continuation latent frames in five blocks of four. The result is 81 RGB frames
at 16 FPS, including the reference. Generated blocks use the same
[method settings](README.md#method-settings) as T2V.

Reference preprocessing and encoding run before the timer. The remaining
[timing boundaries](speed.md#run-a-probe) are the same as T2V. Evaluate the
complete 81-frame output.

## Evaluation

Follow the upstream [I2V evaluation setup](https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v)
in a separate evaluation environment. In addition to the shared video-quality
models, conditioning metrics need DINO ViT-B/16, DreamSim, and CoTracker2. Follow
the upstream checkpoint/cache layout; the T2V scoring requirements alone do
not supply every I2V dependency. Preserve upstream prompt-based video naming
and reference-image matching; indexed T2V filenames cannot be used unchanged.

Set absolute video and score directories, then run from the VBench root so its
relative reference-image paths resolve correctly:

```bash
: "${UNSTEP_VBENCH_SRC:?Set the absolute path to the VBench checkout}"
: "${UNSTEP_I2V_VIDEOS:?Set the absolute path to the prepared I2V videos}"
: "${UNSTEP_I2V_SCORES:?Set the absolute path for I2V score outputs}"
(
  cd "$UNSTEP_VBENCH_SRC"
  python evaluate_i2v.py \
    --videos_path "$UNSTEP_I2V_VIDEOS" \
    --output_path "$UNSTEP_I2V_SCORES" \
    --full_json_dir vbench2_beta_i2v/vbench2_i2v_full_info.json \
    --mode vbench_standard --ratio 16-9 \
    --dimension subject_consistency background_consistency motion_smoothness \
      dynamic_degree aesthetic_quality imaging_quality \
      i2v_subject i2v_background camera_motion
)
```

This produces the nine component results. The paper aggregates normalized
scores as follows: Quality weights dynamic degree by 0.5 and its other five
components by 1; I2V weights subject/background fidelity by 1 and camera motion
by 0.1. Total is the equal average of Quality and I2V. Do not apply the T2V
16-dimension score merger or its 4:1 Quality/Semantic weighting to these results.

[Back to the main README](../README.md#image-to-video).
