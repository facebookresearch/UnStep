#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import runpy
import sys
import types


def patch_torch_distributed() -> None:
    try:
        import torch
    except Exception:
        return
    if torch.distributed.is_nccl_available():
        return
    original_init_process_group = torch.distributed.init_process_group

    def init_process_group(backend=None, *args, **kwargs):
        if backend == "nccl":
            backend = "gloo"
        return original_init_process_group(backend=backend, *args, **kwargs)

    torch.distributed.init_process_group = init_process_group


def patch_transformers() -> None:
    try:
        import transformers.modeling_utils as modeling_utils
        from transformers.generation.configuration_utils import GenerationConfig
        from transformers.generation.utils import GenerationMixin
        from transformers.pytorch_utils import (
            apply_chunking_to_forward,
            find_pruneable_heads_and_indices,
            prune_linear_layer,
        )
    except Exception:
        return
    if not hasattr(modeling_utils, "apply_chunking_to_forward"):
        modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward
    if not hasattr(modeling_utils, "find_pruneable_heads_and_indices"):
        modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    if not hasattr(modeling_utils, "prune_linear_layer"):
        modeling_utils.prune_linear_layer = prune_linear_layer
    if not hasattr(modeling_utils.PreTrainedModel, "generate"):
        for name in dir(GenerationMixin):
            if name.startswith("__"):
                continue
            if hasattr(modeling_utils.PreTrainedModel, name):
                continue
            descriptor = GenerationMixin.__dict__.get(name)
            if isinstance(descriptor, (staticmethod, classmethod)):
                setattr(modeling_utils.PreTrainedModel, name, descriptor)
            else:
                setattr(modeling_utils.PreTrainedModel, name, getattr(GenerationMixin, name))
    descriptor = GenerationMixin.__dict__.get("_expand_inputs_for_generation")
    if isinstance(descriptor, staticmethod):
        modeling_utils.PreTrainedModel._expand_inputs_for_generation = descriptor
    if not getattr(modeling_utils.PreTrainedModel, "_unstep_generation_config_patch", False):
        original_init = modeling_utils.PreTrainedModel.__init__

        def init_with_generation_config(self, config, *args, **kwargs):
            original_init(self, config, *args, **kwargs)
            if getattr(self, "generation_config", None) is None:
                self.generation_config = GenerationConfig.from_model_config(config)

        modeling_utils.PreTrainedModel.__init__ = init_with_generation_config
        modeling_utils.PreTrainedModel._unstep_generation_config_patch = True


def patch_numpy() -> None:
    try:
        import numpy  # noqa: F401
    except Exception:
        return
    if "numpy.lib.function_base" in sys.modules:
        return

    module = types.ModuleType("numpy.lib.function_base")

    def disp(message, device=None, linefeed=True):
        text = str(message)
        if linefeed:
            text += "\n"
        if device is None:
            sys.stdout.write(text)
        else:
            device.write(text)

    module.disp = disp
    sys.modules["numpy.lib.function_base"] = module


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: vbench_eval_entry.py /path/to/evaluate.py [args...]")
    target = sys.argv[1]
    sys.argv = [target] + sys.argv[2:]
    patch_torch_distributed()
    patch_numpy()
    patch_transformers()
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
