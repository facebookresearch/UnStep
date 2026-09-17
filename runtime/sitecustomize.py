# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Runtime knobs for the H100 speed environment."""

from __future__ import annotations

try:
    import torch
    import torch._dynamo.config as _dynamo_config
    import torch._inductor.config as _inductor_config

    _dynamo_config.recompile_limit = 64
    _dynamo_config.accumulated_recompile_limit = 1024
    _inductor_config._use_fp64_for_unbacked_floats = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark_limit = 32
except Exception as exc:
    print(f"unstep runtime setup failed: {exc!r}")

try:
    import torchvision.io as _torchvision_io

    if not hasattr(_torchvision_io, "write_video"):

        def _write_video_unavailable(*args, **kwargs):
            raise RuntimeError(
                "torchvision.io.write_video is unavailable in this build; "
                "use --speed-only or install a torchvision build with video IO."
            )

        _torchvision_io.write_video = _write_video_unavailable
except Exception:
    pass
