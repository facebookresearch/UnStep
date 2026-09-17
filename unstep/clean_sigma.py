# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapts Self Forcing's scheduler method at runtime (Apache-2.0).
# See THIRD_PARTY_NOTICES.md for attribution and upstream license details.

"""Apply the clean-sigma override to input re-noising at conditioning t=0.

The same configuration value is used by the flow-to-x0 converter. No scheduler
table or timestep is edited.
"""

import ast
import inspect
import math
from textwrap import dedent
from types import MethodType

import torch


# Expected Self Forcing scheduler arithmetic, checked before inserting the override.
_NATIVE_ADD_NOISE = ast.parse("""
def add_noise(self, original_samples, noise, timestep):
    if timestep.ndim == 2:
        timestep = timestep.flatten(0, 1)
    self.sigmas = self.sigmas.to(noise.device)
    self.timesteps = self.timesteps.to(noise.device)
    timestep_id = torch.argmin(
        (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
    sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
    sample = (1 - sigma) * original_samples + sigma * noise
    return sample.type_as(noise)
""").body[0]


def _matches_native_add_noise(node):
    if not isinstance(node, ast.FunctionDef):
        return False
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    return (node.name == _NATIVE_ADD_NOISE.name
            and ast.dump(node.args) == ast.dump(_NATIVE_ADD_NOISE.args)
            and not node.decorator_list and node.returns is None
            and [ast.dump(statement) for statement in body]
            == [ast.dump(statement) for statement in _NATIVE_ADD_NOISE.body])


def _validate_input(scheduler, requested, device):
    device = torch.device(device)
    table = scheduler.timesteps.to(device)
    times = torch.cat((table, table.new_tensor([1, 4, 0])))
    zeros = torch.zeros((len(times), 1, 1, 1), dtype=torch.float32, device=device)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    actual = scheduler.add_noise(zeros, torch.ones_like(zeros), times).flatten()
    indices = (table.unsqueeze(0) - times.unsqueeze(1)).abs().argmin(dim=1)
    expected = scheduler.sigmas.to(device)[indices].clone()
    expected[times == 0] = requested
    if not torch.equal(actual, expected):
        raise RuntimeError("clean input sigma validation failed: clean/nonzero coefficient mismatch")
    if not torch.equal(cpu_rng, torch.get_rng_state()) or (
        cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    ):
        raise RuntimeError("clean input sigma validation changed RNG state")
    return {"requested": requested, "effective_at_t0": actual[-1].item(),
            "nonzero_timesteps_unchanged": True, "verified": True,
            "scope": "input_renoising_at_input_timestep_zero"}


def install_shared_clean_sigma(pipeline, transformer, sigma: float, device):
    """Bind one native-method copy to the shared scheduler, then validate input."""
    if sigma is None or isinstance(sigma, bool) or not math.isfinite(sigma) or sigma < 0:
        raise ValueError("cfg.clean_sigma_override must be finite, non-negative and not None")
    scheduler = pipeline.scheduler
    if pipeline.generator is not transformer or scheduler is not transformer.scheduler:
        raise RuntimeError("clean sigma requires one transformer and the same scheduler identity")
    if scheduler.shift != 5 or scheduler.sigmas.dtype != torch.float32:
        raise RuntimeError("clean sigma requires shift 5 and native FP32 scheduler sigmas")
    original = scheduler.add_noise
    try:
        node = ast.parse(dedent(inspect.getsource(original))).body[0]
    except (OSError, TypeError, SyntaxError, IndexError) as exc:
        raise RuntimeError("cannot inspect native add_noise; refusing to install") from exc
    if (getattr(original, "__self__", None) is not scheduler
            or getattr(original, "__func__", None) is not type(scheduler).add_noise
            or not _matches_native_add_noise(node)):
        raise RuntimeError("native add_noise source/binding changed; refusing to install")
    # Insert after sigma lookup/reshape, before mixing. Retain every native statement.
    node.body.insert(-2, ast.parse(
        "sigma = torch.where((timestep == 0).reshape(-1, 1, 1, 1), "
        "torch.full_like(sigma, clean_sigma_override), sigma)"
    ).body[0])
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {"torch": torch, "clean_sigma_override": float(sigma)}
    exec(compile(module, "<unstep clean-input add_noise>", "exec"), namespace)
    scheduler.add_noise = MethodType(namespace["add_noise"], scheduler)
    try:
        return _validate_input(scheduler, float(sigma), device)
    except Exception:
        scheduler.add_noise = original
        raise
