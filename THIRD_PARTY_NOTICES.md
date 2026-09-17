# Third-party code acknowledgments

## Self Forcing and Wan2.1

[Self Forcing](https://github.com/guandeh17/Self-Forcing) provides the base
inference pipeline, diffusion scheduler, attention implementation, and Wan2.1
model and VAE code used by UnStep. The default source location is
`assets/source/Self-Forcing-main`.

`unstep/wrapper_runtime.py` contains adaptations of Self Forcing's attention,
KV-cache, scheduler, and VAE decoding code. `unstep/clean_sigma.py` adapts the
upstream `FlowMatchScheduler.add_noise` method at runtime. These adaptations
are modifications by the UnStep authors.

Self Forcing is distributed under the Apache License 2.0. Its Wan attention
and VAE files retain this notice:

> Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

Self Forcing also credits [Wan2.1](https://github.com/Wan-Video/Wan2.1) and
[CausVid](https://github.com/tianweiy/CausVid). Their upstream notices remain
in the upstream source tree.

## VBench

[VBench](https://github.com/Vchitect/VBench) provides the benchmark prompts and
evaluation code. The default source location is `assets/source/VBench`.

`scripts/merge_vbench_scores.py` uses the dimension weights, normalization
constants, and score aggregation from VBench's `scripts/constant.py` and
`scripts/cal_final_score.py`, adapted to merge results from separate scoring
runs. The evaluation entry point in `scripts/vbench_eval_entry.py` adds
compatibility helpers around the upstream evaluator.

VBench's top-level code is distributed under the Apache License 2.0. Its
bundled third-party components retain their own licenses and notices.

## Licenses

A copy of the Apache License 2.0 for the adapted portions is provided in
[LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt). Upstream dependencies retain their
upstream license files. UnStep's own code and modifications are licensed as
specified in the root [LICENSE](LICENSE).
