# Evaluation

Run these commands from the repository root after completing
[installation](../installation/) and [generation](../generation/).
The scorer evaluates the generated videos with the original VBench prompts.

## Dependencies and scoring models

For VBench scoring, install VBench-specific Python dependencies into a separate
overlay so they do not overwrite the generation stack:

```bash
python -m pip install --target .vbench_deps -r evaluation/requirements.txt
export UNSTEP_VBENCH_PYTHONPATH="$PWD/.vbench_deps"
```

The VBench source tree is not modified; `evaluation/vbench_eval_entry.py` supplies
a small runtime compatibility shim for the current `transformers` import
locations.

Download the standard VBench pretrained scoring models using the upstream
files under `assets/source/VBench/pretrained/` and put/cache them under
`assets/weights/vbench`. Required sources include:

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

This cache contains ViCLIP, CLIP, DINO, AMT, RAFT, GRiT/DenseCap, Tag2Text,
pyiqa/aesthetic, CoTracker, and BERT files as required by VBench. For BERT
tokenizer/config files, use:

```bash
huggingface-cli download bert-base-uncased \
  --local-dir assets/weights/vbench/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594 \
  --allow-pattern config.json \
  --allow-pattern tokenizer.json \
  --allow-pattern tokenizer_config.json \
  --allow-pattern vocab.txt \
  --allow-pattern special_tokens_map.json \
  --local-dir-use-symlinks False
```

For VBench-I2V experiments, use the upstream data instructions in
`assets/source/VBench/vbench2_beta_i2v/README.md`. The upstream helper is:

```bash
python -m pip install gdown
(
  cd assets/source/VBench
  sh vbench2_beta_i2v/download_data.sh
)
```

That downloads `i2v-bench-info.json`, `crop.zip`, and `origin.zip` and unpacks
the I2V image suite. These files are external benchmark data and should not be
committed.

For VBench-long experiments, use the upstream instructions in
`assets/source/VBench/vbench2_beta_long/README.md` and keep any long-video
benchmark data or model caches under `assets/weights/vbench` or another local
path passed through the corresponding `UNSTEP_*` environment variable.

Build Detectron2 after PyTorch and CUDA are installed:

```bash
git clone --depth 1 https://github.com/facebookresearch/detectron2.git .build/detectron2
UNSTEP_VBENCH_DEPS="$PWD/.vbench_deps" \
./evaluation/install_detectron2.sh .build/detectron2
```

If CUDA is installed from `installation/requirements-cuda13.txt`, set:

```bash
export UNSTEP_CUDA_OVERLAY=/path/to/cuda-wheel-overlay
export UNSTEP_CUDA="$UNSTEP_CUDA_OVERLAY/nvidia/cu13"
```


## Run VBench

The scorer uses two local GPUs and reads `results/<RUN_NAME>/indexed_videos`:

```bash
./evaluation/run.sh unstep_vbench
```

Scores and logs are written under `results/<RUN_NAME>/vbench_score`.
The scorer deletes generated MP4 files after a successful merge by default.
To preserve them:

```bash
UNSTEP_DELETE_VIDEOS_AFTER_SCORE=0 ./evaluation/run.sh unstep_vbench
```
