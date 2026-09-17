#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Wan runtime adapter used by wrapper.py.

This file intentionally contains runtime plumbing only: loading the upstream Wan
pipeline, creating caches, mapping timesteps, and doing the generic VAE decode
call. Method choices such as reduced schedules, SVD ratios, attention windows,
and runtime stack selection live in wrapper.py.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import logging
import math
import os
import shutil
import sys
import tarfile
import types
import zipfile
from pathlib import Path
from typing import Any

import torch
from torch import nn


DEFAULT_KV_CACHE_SIZE = 32760
WAN_TOKENS_PER_LATENT_FRAME = DEFAULT_KV_CACHE_SIZE // 21
CHECKPOINT_AUDIT_FILENAME = "checkpoint_audit.json"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaseSFConfig:
    """Self-Forcing/Wan benchmark setup, not wrapper method logic."""

    num_latent_frames: int = 21
    num_frame_per_block: int = 3
    seed: int = 0
    seed_mode: str = "sequential_global"
    precision: str = "bf16"


class LocalPathManager:
    def get_local_path(self, uri: str | Path) -> str:
        if "://" in str(uri):
            raise RuntimeError(
                "Only local filesystem paths are supported; download assets first."
            )
        return str(Path(uri))

    def exists(self, uri: str | Path) -> bool:
        if "://" in str(uri):
            return False
        return Path(uri).exists()


def path_manager() -> LocalPathManager:
    return LocalPathManager()


def chunk_frame_counts(num_latent_frames: int, num_frame_per_block: int) -> list[int]:
    if num_frame_per_block <= 0:
        raise RuntimeError("num_frame_per_block must be positive")
    counts = []
    current_start_frame = 0
    while current_start_frame < num_latent_frames:
        current_num_frames = min(
            num_frame_per_block,
            num_latent_frames - current_start_frame,
        )
        counts.append(current_num_frames)
        current_start_frame += current_num_frames
    if not counts:
        raise RuntimeError("num_latent_frames must be positive")
    return counts


def read_prompts(manager: Any, prompt_uri: str, indices: list[int]) -> dict[int, str]:
    prompt_path = Path(manager.get_local_path(prompt_uri))
    prompts = [
        line.rstrip() for line in prompt_path.read_text(encoding="utf-8").splitlines()
    ]
    selected = {}
    for index in indices:
        if index < 0 or index >= len(prompts):
            raise RuntimeError(f"prompt index {index} outside {len(prompts)} prompts")
        selected[index] = prompts[index]
    return selected


def torch_dtype(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    raise RuntimeError(f"unsupported precision: {precision}")


def _copy_source_tree(src: Path, dst: Path) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    if src.is_dir():
        shutil.copytree(src, dst)
        return dst
    if src.suffix == ".zip":
        extract_dir = dst
        extract_dir.mkdir(parents=True)
        with zipfile.ZipFile(src) as zf:
            zf.extractall(extract_dir)
        children = [child for child in extract_dir.iterdir() if child.is_dir()]
        if len(children) != 1:
            raise RuntimeError(f"expected one source root in {src}, found {children}")
        return children[0]
    raise RuntimeError(f"--source-uri must be an SF directory or zip, got {src}")


def _unpack_weights(manager: Any, weights_uri: str | Path, source_root: Path) -> None:
    weights_path = Path(manager.get_local_path(str(weights_uri)))
    if not weights_path.exists():
        raise RuntimeError(f"weights path does not exist: {weights_path}")
    if weights_path.suffix == ".tar":
        with tarfile.open(weights_path) as tar:
            tar.extractall(source_root, filter="data")
        return
    checkpoint_dst = source_root / "checkpoints" / "self_forcing_dmd.pt"
    checkpoint_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights_path, checkpoint_dst)


def _copy_vae(manager: Any, vae_uri: str | Path, source_root: Path) -> None:
    vae_path = Path(manager.get_local_path(str(vae_uri)))
    dst = source_root / "wan_models" / "Wan2.1-T2V-1.3B" / "Wan2.1_VAE.pth"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(vae_path, dst)


def _strip_state_dict_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = (
        "module.",
        "_orig_mod.",
        "generator.",
        "generator_ema.",
        "_fsdp_wrapped_module.",
    )
    out = {}
    for key, value in state_dict.items():
        clean_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix) :]
                    changed = True
        out[clean_key] = value
    return out


def write_normalized_checkpoint(src: Path, dst: Path) -> dict[str, object]:
    raw = torch.load(src, map_location="cpu")
    if isinstance(raw, dict) and "generator_ema" in raw:
        generator_state = raw["generator_ema"]
        source_key = "generator_ema"
    elif isinstance(raw, dict) and "generator" in raw:
        generator_state = raw["generator"]
        source_key = "generator"
    elif isinstance(raw, dict):
        generator_state = raw
        source_key = "raw_state_dict"
    else:
        raise RuntimeError(f"unsupported checkpoint format in {src}")
    if not isinstance(generator_state, dict):
        raise RuntimeError(f"checkpoint key {source_key} is not a state dict")
    normalized = _strip_state_dict_prefixes(generator_state)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"generator_ema": normalized}, dst)
    return {
        "source": str(src),
        "destination": str(dst),
        "source_key": source_key,
        "tensor_count": len(normalized),
        "generator_ema_present": isinstance(raw, dict) and "generator_ema" in raw,
    }


def patch_attention_fallback(
    source_root: Path,
    *,
    enable_flash_attn3: bool,
    fa3_import_backend: str = "auto",
) -> None:
    attention_py = source_root / "wan" / "modules" / "attention.py"
    text = attention_py.read_text(encoding="utf-8")
    old = """try:
    import flash_attn_interface

    def is_h100_gpu():
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return "h100" in device_name
    FLASH_ATTN_3_AVAILABLE = is_h100_gpu()
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

"""
    if enable_flash_attn3 and fa3_import_backend in {"auto", "h100"}:
        new = """try:
    import flash_attn_interface
except ModuleNotFoundError:
    flash_attn_interface = None

def is_h100_gpu():
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 9
FLASH_ATTN_3_AVAILABLE = flash_attn_interface is not None and is_h100_gpu()

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

"""
    elif enable_flash_attn3 and fa3_import_backend == "flash_attn_3":
        new = """try:
    import flash_attn_3.flash_attn_interface as flash_attn_interface
except ModuleNotFoundError:
    flash_attn_interface = None

def is_h100_gpu():
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 9
FLASH_ATTN_3_AVAILABLE = flash_attn_interface is not None and is_h100_gpu()

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

"""
    else:
        new = """FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

"""
    if old in text:
        text = text.replace(old, new)

    fa3_old = """            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
"""
    fa3_new = """            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)
        if isinstance(x, tuple):
            x = x[0]
        x = x.unflatten(0, (b, lq))
"""
    if enable_flash_attn3 and fa3_old in text:
        text = text.replace(fa3_old, fa3_new)

    fallback_old = """    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))
"""
    fallback_new = """    elif FLASH_ATTN_2_AVAILABLE:
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))
    else:
        if b != 1:
            raise RuntimeError(
                'scaled_dot_product_attention fallback only supports batch size 1; install flash-attn for larger batches.'
            )
        q = q.unflatten(0, (b, int(q_lens[0].item()))).transpose(1, 2)
        k = k.unflatten(0, (b, int(k_lens[0].item()))).transpose(1, 2)
        v = v.unflatten(0, (b, int(k_lens[0].item()))).transpose(1, 2)
        x = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=causal,
            dropout_p=dropout_p,
            scale=softmax_scale,
        ).transpose(1, 2).contiguous()
"""
    if fallback_old in text:
        text = text.replace(fallback_old, fallback_new)
    elif "scaled_dot_product_attention fallback only supports batch size 1" not in text:
        raise RuntimeError(f"attention fallback patch did not match {attention_py}")

    attention_py.write_text(text, encoding="utf-8")


def patch_causal_forward_debug_dict_abi(source_root: Path) -> None:
    causal_model_py = source_root / "wan" / "modules" / "causal_model.py"
    text = causal_model_py.read_text(encoding="utf-8")
    if "debug_dict=None" in text:
        return
    old = "        current_start: int = 0,\n        cache_start: int = 0\n    ):\n"
    new = (
        "        current_start: int = 0,\n"
        "        cache_start: int = 0,\n"
        "        debug_dict=None,\n"
        "        **unused_inference_kwargs,\n"
        "    ):\n"
    )
    if old in text:
        causal_model_py.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_causal_cache_cursor_reads(source_root: Path) -> None:
    causal_model_py = source_root / "wan" / "modules" / "causal_model.py"
    text = causal_model_py.read_text(encoding="utf-8")
    old = """            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                # Clone the source slice to avoid overlapping memory error
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \\
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \\
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                # Insert the new keys/values at the end
                local_end_index = kv_cache["local_end_index"].item() + current_end - \\
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
"""
    new = """            global_end_index = kv_cache["global_end_index"].item()
            previous_local_end_index = kv_cache["local_end_index"].item()
            if self.local_attn_size != -1 and (current_end > global_end_index) and (
                    num_new_tokens + previous_local_end_index > kv_cache_size):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                # Clone the source slice to avoid overlapping memory error
                num_evicted_tokens = num_new_tokens + previous_local_end_index - kv_cache_size
                num_rolled_tokens = previous_local_end_index - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \\
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \\
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                # Insert the new keys/values at the end
                local_end_index = previous_local_end_index + current_end - \\
                    global_end_index - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = previous_local_end_index + current_end - global_end_index
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
"""
    if old in text:
        causal_model_py.write_text(text.replace(old, new), encoding="utf-8")


def prepare_source(args: Any, manager: Any, workdir: Path) -> Path:
    source_path = Path(manager.get_local_path(str(args.source_uri)))
    source_root = _copy_source_tree(source_path, workdir / "src")
    _unpack_weights(manager, args.weights_uri, source_root)
    _copy_vae(manager, args.vae_uri, source_root)

    checkpoint_uri = getattr(args, "checkpoint_uri", None)
    if checkpoint_uri is not None:
        src = Path(manager.get_local_path(checkpoint_uri))
        dst = source_root / "checkpoints" / "self_forcing_dmd.pt"
        audit = write_normalized_checkpoint(src, dst)
        (source_root / CHECKPOINT_AUDIT_FILENAME).write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    patch_attention_fallback(
        source_root,
        enable_flash_attn3=getattr(args, "enable_flash_attn3", False),
        fa3_import_backend=getattr(args, "fa3_import_backend", "auto"),
    )
    patch_causal_forward_debug_dict_abi(source_root)
    if not getattr(args, "skip_cache_cursor_patch", False):
        patch_causal_cache_cursor_reads(source_root)

    from omegaconf import OmegaConf

    config = OmegaConf.load(source_root / "configs/self_forcing_dmd.yaml")
    config.num_frame_per_block = args.num_frame_per_block
    timestep_shift = getattr(args, "timestep_shift", None)
    if timestep_shift is not None:
        config.timestep_shift = float(timestep_shift)
        if "model_kwargs" not in config or config.model_kwargs is None:
            config.model_kwargs = {}
        config.model_kwargs.timestep_shift = float(timestep_shift)
    if getattr(args, "local_attn_size", None) is not None or getattr(args, "sink_size", None) is not None:
        if "model_kwargs" not in config or config.model_kwargs is None:
            config.model_kwargs = {}
        if getattr(args, "local_attn_size", None) is not None:
            config.model_kwargs.local_attn_size = args.local_attn_size
        if getattr(args, "sink_size", None) is not None:
            config.model_kwargs.sink_size = args.sink_size
    OmegaConf.save(config, source_root / "configs/self_forcing_dmd_speed.yaml")
    return source_root


def runtime_info(source_root: Path, precision: str) -> dict[str, object]:
    def has_module(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except ModuleNotFoundError:
            return False

    info: dict[str, object] = {
        "precision": precision,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "flash_attn_spec": has_module("flash_attn"),
        "flash_attn_interface_spec": has_module("flash_attn_interface"),
        "flash_attn_3_spec": has_module("flash_attn_3.flash_attn_interface"),
    }
    if torch.cuda.is_available():
        info.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_capability": torch.cuda.get_device_capability(0),
            }
        )
    try:
        import wan.modules.attention as attention

        info.update(
            {
                "wan_flash_attn_2_available": attention.FLASH_ATTN_2_AVAILABLE,
                "wan_flash_attn_3_available": attention.FLASH_ATTN_3_AVAILABLE,
            }
        )
    except Exception as exc:
        info["wan_attention_import_error"] = repr(exc)
    return info


def load_pipeline(
    source_root: Path,
    precision: str,
    load_full_vae: bool = False,
    move_generator_to_device: bool = True,
    generator: Any | None = None,
):
    del load_full_vae
    compute_dtype = torch_dtype(precision)
    old_cwd = Path.cwd()
    os.chdir(source_root)
    sys.path.insert(0, str(source_root))
    try:
        from demo_utils.constant import ZERO_VAE_CACHE
        from pipeline import CausalInferencePipeline

        from omegaconf import OmegaConf

        config = OmegaConf.load("configs/self_forcing_dmd_speed.yaml")
        default_config = OmegaConf.load("configs/default_config.yaml")
        config = OmegaConf.merge(default_config, config)

        device = torch.device("cuda")
        pipeline = CausalInferencePipeline(config, device=device, generator=generator)
        if generator is None:
            state_dict = torch.load("checkpoints/self_forcing_dmd.pt", map_location="cpu")
            pipeline.generator.load_state_dict(state_dict["generator_ema"])
        pipeline.eval().requires_grad_(False)
        if move_generator_to_device:
            pipeline = pipeline.to(dtype=compute_dtype)
            pipeline.text_encoder.to(device=device)
            pipeline.generator.to(device=device)
            pipeline.vae.to(device=device)
        else:
            pipeline.text_encoder.to(device=device, dtype=compute_dtype)
            pipeline.generator.to(device=torch.device("cpu"), dtype=torch.float32)
            pipeline.vae.to(device=device, dtype=compute_dtype)
        return (
            pipeline,
            pipeline.generator,
            pipeline.text_encoder,
            pipeline.vae,
            ZERO_VAE_CACHE,
            device,
        )
    finally:
        os.chdir(old_cwd)


def kv_cache_size_for_latents(num_latent_frames: int) -> int:
    return max(DEFAULT_KV_CACHE_SIZE, num_latent_frames * WAN_TOKENS_PER_LATENT_FRAME)


def kv_cache_size_for_pipeline(pipeline: Any, num_latent_frames: int) -> int:
    override = getattr(pipeline, "_spec_kv_cache_latent_frames", None)
    if override is not None:
        return int(override) * WAN_TOKENS_PER_LATENT_FRAME
    local_attn_size = getattr(pipeline, "local_attn_size", -1)
    if local_attn_size != -1:
        return int(local_attn_size) * WAN_TOKENS_PER_LATENT_FRAME
    return kv_cache_size_for_latents(num_latent_frames)


def parse_index_range_list(
    value: str | None,
    *,
    name: str,
    upper_bound: int | None = None,
) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    out: set[int] = set()
    for item in value.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
        else:
            start = end = int(item)
        if start < 0 or end < start:
            raise RuntimeError(f"{name} contains invalid range {item!r}")
        if upper_bound is not None and end >= upper_bound:
            raise RuntimeError(
                f"{name} range {item!r} is outside available depth {upper_bound}"
            )
        out.update(range(start, end + 1))
    if not out:
        raise RuntimeError(f"{name} did not contain any integer values")
    return sorted(out)


def parse_layer_int_ranges(spec: str | None, num_layers: int) -> dict[int, int] | None:
    if spec is None or spec.strip() == "":
        return None
    parsed: dict[int, int] = {}
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise RuntimeError("layer-size spec entries must look like start-end:size")
        span_text, size_text = item.split(":", 1)
        size = int(size_text)
        if size <= 0:
            raise RuntimeError("local attention layer sizes must be positive")
        layers = parse_index_range_list(
            span_text,
            name="local_attn_layer_sizes",
            upper_bound=num_layers,
        )
        assert layers is not None
        for layer in layers:
            parsed[layer] = size
    if not parsed:
        raise RuntimeError("local attention layer-size spec was empty")
    return parsed


def selected_timesteps(pipeline: Any, indices: list[int] | None = None) -> list[Any]:
    steps = list(pipeline.denoising_step_list)
    if indices is None:
        return steps
    return [steps[index] for index in indices]


def _float_attr(obj: Any, name: str) -> float | None:
    if not hasattr(obj, name):
        return None
    value = getattr(obj, name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_attr(obj: Any, name: str) -> bool | None:
    if not hasattr(obj, name):
        return None
    value = getattr(obj, name)
    if value is None:
        return None
    return bool(value)


def _as_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        flat = value.detach().float().cpu().flatten()
        return [float(item) for item in flat.tolist()]
    try:
        return [float(item) for item in value]
    except TypeError:
        try:
            return [float(value)]
        except (TypeError, ValueError):
            return None


def _scheduler_summary(scheduler: Any) -> dict[str, object]:
    timesteps = _as_float_list(getattr(scheduler, "timesteps", None))
    sigmas = _as_float_list(getattr(scheduler, "sigmas", None))
    summary: dict[str, object] = {
        "class": type(scheduler).__name__,
        "shift": _float_attr(scheduler, "shift"),
        "timestep_shift": _float_attr(scheduler, "timestep_shift"),
        "sigma_min": _float_attr(scheduler, "sigma_min"),
        "sigma_max": _float_attr(scheduler, "sigma_max"),
        "extra_one_step": _bool_attr(scheduler, "extra_one_step"),
    }
    if timesteps:
        summary["first_timestep"] = float(timesteps[0])
        summary["last_timestep"] = float(timesteps[-1])
        zero_index = min(range(len(timesteps)), key=lambda i: abs(timesteps[i]))
        summary["zero_lookup_index"] = int(zero_index)
        summary["zero_lookup_timestep"] = float(timesteps[zero_index])
        if sigmas is not None and zero_index < len(sigmas):
            summary["zero_lookup_sigma"] = float(sigmas[zero_index])
    return summary


def scheduler_shift_summary(*, pipeline: Any, method_transformer: Any) -> dict[str, object]:
    return {
        "pipeline_scheduler": _scheduler_summary(pipeline.scheduler),
        "target_transformer_scheduler": _scheduler_summary(pipeline.generator.scheduler),
        "method_transformer_scheduler": _scheduler_summary(method_transformer.scheduler),
        "selected_timesteps": [
            float(timestep.item()) if torch.is_tensor(timestep) else float(timestep)
            for timestep in selected_timesteps(pipeline)
        ],
    }


def is_zero_timestep(timestep: object) -> bool:
    if torch.is_tensor(timestep):
        return float(timestep.item()) == 0.0
    return float(timestep) == 0.0


def timestep_value(timestep: object) -> int:
    if torch.is_tensor(timestep):
        return int(timestep.item())
    return int(timestep)


def timestep_tensor(
    cache: dict[tuple[tuple[int, ...], str, int, int | float], torch.Tensor] | None,
    shape: tuple[int, ...],
    timestep: object,
    device: torch.device,
) -> torch.Tensor:
    value = timestep_value(timestep)
    if cache is None:
        return torch.full(shape, value, device=device, dtype=torch.long)
    key = (shape, device.type, -1 if device.index is None else device.index, value)
    cached = cache.get(key)
    if cached is None:
        cached = torch.full(shape, value, device=device, dtype=torch.long)
        cache[key] = cached
    return cached


def randn_like_with_generator(
    tensor: torch.Tensor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if generator is None:
        return torch.randn_like(tensor)
    return torch.randn(
        tensor.shape,
        device=tensor.device,
        dtype=tensor.dtype,
        generator=generator,
    )


def add_noise_transition(
    *,
    pipeline: Any,
    denoised_pred: torch.Tensor,
    next_timestep: object,
    generator: torch.Generator | None,
        timestep_cache: dict[tuple[tuple[int, ...], str, int, int | float], torch.Tensor]
    | None = None,
) -> torch.Tensor:
    batch_size, frame_count = [int(item) for item in denoised_pred.shape[:2]]
    flat_pred = denoised_pred.flatten(0, 1)
    return pipeline.scheduler.add_noise(
        flat_pred,
        randn_like_with_generator(flat_pred, generator),
        timestep_tensor(
            timestep_cache,
            (batch_size * frame_count,),
            next_timestep,
            denoised_pred.device,
        ),
    ).unflatten(0, denoised_pred.shape[:2])


def make_kv_cache(
    pipeline: Any,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    init: str = "empty",
    cursor: str = "cpu_tensor",
    kv_cache_size: int | None = None,
    allow_local_attn_eviction: bool = False,
) -> list[dict[str, object]]:
    if pipeline.local_attn_size != -1 and not allow_local_attn_eviction:
        raise RuntimeError(
            "local-attention KV eviction must be explicitly enabled"
        )
    if init not in {"zero", "empty"}:
        raise RuntimeError("kv cache init must be zero or empty")
    if cursor not in {"tensor", "cpu_tensor"}:
        raise RuntimeError("kv cache cursor must be tensor or cpu_tensor")
    allocate = torch.zeros if init == "zero" else torch.empty
    cursor_device = torch.device("cpu") if cursor == "cpu_tensor" else device
    size = kv_cache_size or kv_cache_size_for_pipeline(
        pipeline,
        getattr(pipeline, "_spec_num_latent_frames", 21),
    )

    def new_cursor() -> torch.Tensor:
        return torch.tensor([0], dtype=torch.long, device=cursor_device)

    return [
        {
            "k": allocate([batch_size, size, 12, 128], dtype=dtype, device=device),
            "v": allocate([batch_size, size, 12, 128], dtype=dtype, device=device),
            "global_end_index": new_cursor(),
            "local_end_index": new_cursor(),
        }
        for _ in range(pipeline.num_transformer_blocks)
    ]


def make_crossattn_cache(
    pipeline: Any,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    init: str = "empty",
) -> list[dict[str, object]]:
    if init not in {"zero", "empty"}:
        raise RuntimeError("cross-attention cache init must be zero or empty")
    allocate = torch.zeros if init == "zero" else torch.empty
    return [
        {
            "k": allocate([batch_size, 512, 12, 128], dtype=dtype, device=device),
            "v": allocate([batch_size, 512, 12, 128], dtype=dtype, device=device),
            "is_init": False,
        }
        for _ in range(pipeline.num_transformer_blocks)
    ]


def wan_model(transformer: Any) -> Any:
    return getattr(transformer, "model", transformer)


_wan_model = wan_model


def _optional_model_state(transformer: Any, name: str) -> dict[str, object] | None:
    return getattr(wan_model(transformer), name, None)


def set_step_layer_skip_route_active(transformer: Any, active: bool | None) -> None:
    state = _optional_model_state(transformer, "_step_layer_skip_state")
    if state is not None:
        state["temporal_delta_fast_route_active"] = active


def set_attention_svd_route_active(transformer: Any, active: bool) -> int:
    count = 0
    blocks = getattr(wan_model(transformer), "blocks", None)
    if blocks is None:
        return 0
    for block in blocks:
        self_attn = getattr(block, "self_attn", None)
        if self_attn is None:
            continue
        for module_name in ("q", "k", "v", "o"):
            module = getattr(self_attn, module_name, None)
            if hasattr(module, "route_active"):
                module.route_active = bool(active)
                count += 1
    return count


def set_cross_step_kv_context(
    transformer: Any,
    *,
    current_start_frame: int,
    step_index: int | None,
    timestep: int | float | None,
    active: bool,
    block_idx: int | None = None,
    num_blocks: int | None = None,
) -> None:
    """Match the validated per-step context bookkeeping."""

    context = {
        "active": active,
        "current_start_frame": int(current_start_frame),
        "step_index": None if step_index is None else int(step_index),
        "timestep": None if timestep is None else int(timestep),
        "block_idx": None if block_idx is None else int(block_idx),
        "num_blocks": None if num_blocks is None else int(num_blocks),
    }
    for state in (
        _optional_model_state(transformer, "_cross_step_kv_profile"),
        _optional_model_state(transformer, "_cross_step_mlp_residual_profile"),
    ):
        if state is None:
            continue
        state["context"] = context
        if step_index == 0 or not active:
            state["prev"] = {}
    for name in (
        "_step_layer_skip_state",
        "_frame_interleaved_sparsity_state",
    ):
        state = _optional_model_state(transformer, name)
        if state is not None:
            state["context"] = context
    temporal_kv_state = _optional_model_state(transformer, "_temporal_kv_delta_profile")
    if temporal_kv_state is not None:
        temporal_kv_state["context"] = context
        if not active or (current_start_frame == 0 and step_index == 0):
            temporal_kv_state["prev"] = {}


def apply_layerwise_local_attention(
    transformer: Any,
    pipeline: Any,
    spec: str | None,
) -> None:
    if spec is None:
        return
    model = _wan_model(transformer)
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise RuntimeError("local-attention layer sizing requires Wan blocks")
    layer_sizes = parse_layer_int_ranges(spec, len(blocks))
    assert layer_sizes is not None
    base_size = getattr(pipeline, "local_attn_size", -1)
    if base_size == -1:
        raise RuntimeError("local-attention layer sizing requires local_attn_size")
    max_size = max(max(layer_sizes.values()), int(base_size))
    for layer_index, block in enumerate(blocks):
        self_attn = getattr(block, "self_attn", None)
        if self_attn is None:
            raise RuntimeError("expected Wan block with self_attn")
        layer_size = layer_sizes.get(layer_index, int(base_size))
        self_attn.local_attn_size = layer_size
        self_attn.max_attention_size = layer_size * WAN_TOKENS_PER_LATENT_FRAME
    pipeline.local_attn_size = max_size
    model.local_attn_size = max_size
    pipeline._spec_kv_cache_latent_frames = max_size
    model._spec_local_attn_layer_sizes = {
        str(layer): size for layer, size in sorted(layer_sizes.items())
    }


def copy_local_attention_settings(source_transformer: Any, target_transformer: Any) -> None:
    if source_transformer is target_transformer:
        return
    source_model = _wan_model(source_transformer)
    target_model = _wan_model(target_transformer)
    source_blocks = getattr(source_model, "blocks", None)
    target_blocks = getattr(target_model, "blocks", None)
    if source_blocks is None or target_blocks is None:
        raise RuntimeError("copy_local_attention_settings requires Wan blocks")
    target_model.local_attn_size = getattr(source_model, "local_attn_size", -1)
    if hasattr(source_model, "_spec_local_attn_layer_sizes"):
        target_model._spec_local_attn_layer_sizes = dict(
            source_model._spec_local_attn_layer_sizes
        )
    if hasattr(source_model, "_spec_sink_stamp_noroll"):
        target_model._spec_sink_stamp_noroll = bool(source_model._spec_sink_stamp_noroll)
    for source_block, target_block in zip(source_blocks, target_blocks, strict=True):
        source_attn = source_block.self_attn
        target_attn = target_block.self_attn
        target_attn.local_attn_size = getattr(source_attn, "local_attn_size", -1)
        target_attn.sink_size = getattr(source_attn, "sink_size", 0)
        target_attn.max_attention_size = getattr(source_attn, "max_attention_size", -1)
        if hasattr(source_attn, "sink_stamp_noroll"):
            target_attn.sink_stamp_noroll = bool(source_attn.sink_stamp_noroll)


def load_self_forcing_transformer(
    *,
    source_root: Path,
    precision: str,
    device: torch.device,
    move_to_device: bool = True,
    timestep_shift: float | None = None,
) -> Any:
    """Load the upstream Self-Forcing transformer without wrapper changes."""

    old_cwd = Path.cwd()
    os.chdir(source_root)
    sys.path.insert(0, str(source_root))
    try:
        from utils.wan_wrapper import WanDiffusionWrapper

        kwargs = {"is_causal": True}
        if timestep_shift is not None:
            kwargs["timestep_shift"] = float(timestep_shift)
        transformer = WanDiffusionWrapper(**kwargs)
        state_dict = torch.load("checkpoints/self_forcing_dmd.pt", map_location="cpu")
        transformer.load_state_dict(state_dict["generator_ema"])
        if move_to_device:
            transformer.eval().requires_grad_(False)
            transformer.to(device=device, dtype=torch_dtype(precision))
        return transformer
    finally:
        os.chdir(old_cwd)


def reset_transformer_scheduler(transformer: Any, timestep_shift: float) -> dict[str, Any]:
    from utils.scheduler import FlowMatchScheduler

    scheduler = FlowMatchScheduler(
        shift=float(timestep_shift),
        sigma_min=0.0,
        extra_one_step=True,
    )
    scheduler.set_timesteps(1000, training=True)
    transformer.scheduler = scheduler
    return {
        "transformer_scheduler_shift": float(timestep_shift),
        "transformer_scheduler_reset": True,
    }


def bind_pipeline_scheduler_to_transformer(pipeline: Any, transformer: Any) -> dict[str, Any]:
    """Keep the one-DiT pipeline scheduler object tied to the reused generator."""

    previous_steps = [
        float(timestep.item()) if torch.is_tensor(timestep) else float(timestep)
        for timestep in selected_timesteps(pipeline)
    ]
    pipeline.scheduler = transformer.scheduler
    if getattr(pipeline.args, "warp_denoising_step", False):
        raw_steps = torch.tensor(pipeline.args.denoising_step_list, dtype=torch.long)
        timesteps = torch.cat(
            (
                pipeline.scheduler.timesteps.cpu(),
                torch.tensor([0], dtype=torch.float32),
            )
        )
        pipeline.denoising_step_list = timesteps[1000 - raw_steps]
    next_steps = [
        float(timestep.item()) if torch.is_tensor(timestep) else float(timestep)
        for timestep in selected_timesteps(pipeline)
    ]
    if previous_steps != next_steps:
        raise RuntimeError(
            "scheduler rebinding changed selected timesteps: "
            f"{previous_steps} -> {next_steps}"
        )
    return {
        "pipeline_scheduler_rebound_to_transformer": True,
        "pipeline_generator_scheduler_same_object": (
            pipeline.scheduler is pipeline.generator.scheduler
        ),
    }


def enable_cached_scheduler_sigmas(
    transformer: Any,
    *,
    clean_sigma_override: float | None = None,
) -> None:
    if clean_sigma_override is not None and (
        not math.isfinite(clean_sigma_override) or clean_sigma_override < 0
    ):
        raise ValueError("clean_sigma_override must be finite and non-negative")
    cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    def cached_convert_flow_pred_to_x0(
        self,
        flow_pred: torch.Tensor,
        xt: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        original_dtype = flow_pred.dtype
        device = flow_pred.device
        cached = cache.get(device)
        if cached is None:
            cached = (
                self.scheduler.sigmas.double().to(device),
                self.scheduler.timesteps.double().to(device),
            )
            cache[device] = cached
        sigmas, timesteps = cached
        flow_pred = flow_pred.double().to(device)
        xt = xt.double().to(device)
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(),
            dim=1,
        )
        sigma_values = sigmas[timestep_id]
        if clean_sigma_override is not None:
            # The input is t=0; the nearest scheduler entry is generally nonzero.
            clean_sigma = torch.full_like(sigma_values, float(clean_sigma_override))
            sigma_values = torch.where(timestep == 0.0, clean_sigma, sigma_values)
        sigma_t = sigma_values.reshape(-1, 1, 1, 1)
        return (xt - sigma_t * flow_pred).to(original_dtype)

    transformer._convert_flow_pred_to_x0 = cached_convert_flow_pred_to_x0.__get__(
        transformer,
        type(transformer),
    )


def validate_clean_sigma_override(
    transformer: Any,
    *,
    clean_sigma_override: float | None,
    device: torch.device,
) -> dict[str, object]:
    """Exercise the installed converter before generation, outside its timer."""
    scheduler = transformer.scheduler
    table = scheduler.timesteps.double().to(device)
    sigmas = scheduler.sigmas.double().to(device)
    native_sigma = sigmas[table.abs().argmin()].item()
    expected_clean = native_sigma if clean_sigma_override is None else clean_sigma_override
    times = torch.cat((table, table.new_zeros(1)))
    flow = torch.ones((len(times), 1, 1, 1), dtype=torch.float64, device=device)
    # With x_t=0 and flow=1, the negated output is the effective sigma.
    actual = -transformer._convert_flow_pred_to_x0(flow, torch.zeros_like(flow), times).flatten()
    expected = torch.cat((sigmas, sigmas.new_tensor([expected_clean])))
    if clean_sigma_override is not None:
        expected[times == 0] = clean_sigma_override
    if not torch.equal(actual, expected):
        raise RuntimeError(
            "clean sigma validation failed: "
            f"requested={clean_sigma_override}, effective_at_t0={actual[-1].item()}, "
            f"expected_at_t0={expected_clean}; check clean and non-clean conversions"
        )
    return {
        "requested": clean_sigma_override,
        "effective_at_t0": actual[-1].item(),
        "scheduler_native_at_t0": native_sigma,
        "scope": "flow_to_x0_at_input_timestep_zero",
        "nonzero_timesteps_unchanged": True,
        "verified": True,
    }


def install_flash_attention_seqlen_cache(
    *,
    use_fixed_fa3_batch1: bool,
    use_fixed_fa3_batch1_slice_lens: bool,
) -> None:
    import wan.modules.attention as attention_module
    import wan.modules.model as model_module

    if hasattr(attention_module, "_sf_original_flash_attention"):
        return

    original_flash_attention = attention_module.flash_attention
    attention_module._sf_original_flash_attention = original_flash_attention
    lens_cache: dict[tuple[str, int, int, int], torch.Tensor] = {}
    cu_cache: dict[tuple[str, int, int, int], torch.Tensor] = {}

    def cached_lens(
        device: torch.device,
        batch_size: int,
        sequence_length: int,
    ) -> torch.Tensor:
        key = (
            device.type,
            -1 if device.index is None else device.index,
            batch_size,
            sequence_length,
        )
        cached = lens_cache.get(key)
        if cached is None:
            cached = torch.full(
                [batch_size],
                sequence_length,
                dtype=torch.int32,
                device=device,
            )
            lens_cache[key] = cached
        return cached

    def cached_cu_seqlens(
        device: torch.device,
        batch_size: int,
        sequence_length: int,
    ) -> torch.Tensor:
        key = (
            device.type,
            -1 if device.index is None else device.index,
            batch_size,
            sequence_length,
        )
        cached = cu_cache.get(key)
        if cached is None:
            cached = torch.arange(
                0,
                (batch_size + 1) * sequence_length,
                sequence_length,
                dtype=torch.int32,
                device=device,
            )
            cu_cache[key] = cached
        return cached

    def prewarm_flash_attention_seqlens(
        device: torch.device,
        batch_size: int,
        sequence_lengths: list[int],
    ) -> None:
        for sequence_length in sorted(set(sequence_lengths)):
            if sequence_length <= 0:
                continue
            cached_lens(device, batch_size, sequence_length)
            cached_cu_seqlens(device, batch_size, sequence_length)

    def cached_flash_attention(
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.0,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=torch.bfloat16,
        version=None,
    ):
        half_dtypes = (torch.float16, torch.bfloat16)

        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)

        def first_cpu_length(lengths) -> int | None:
            if torch.is_tensor(lengths):
                if lengths.numel() != 1 or lengths.device.type != "cpu":
                    return None
                return int(lengths.reshape(-1)[0].item())
            if isinstance(lengths, (list, tuple)) and len(lengths) == 1:
                return int(lengths[0])
            return None

        def run_dense_fa3(q_in, k_in, v_in, out_dtype):
            q_dense = half(q_in).to(v_in.dtype)
            k_dense = half(k_in).to(v_in.dtype)
            v_dense = half(v_in)
            if q_scale is not None:
                q_dense = q_dense * q_scale
            dense_out = attention_module.flash_attn_interface.flash_attn_func(
                q_dense,
                k_dense,
                v_dense,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic,
            )
            if isinstance(dense_out, tuple):
                dense_out = dense_out[0]
            return dense_out.type(out_dtype)

        if q_lens is not None or k_lens is not None:
            batch_size, k_length = q.size(0), k.size(1)
            k_len = first_cpu_length(k_lens) if k_lens is not None else None
            if (
                use_fixed_fa3_batch1_slice_lens
                and use_fixed_fa3_batch1
                and q_lens is None
                and k_len is not None
                and batch_size == 1
                and k.size(0) == 1
                and (version is None or version == 3)
                and attention_module.FLASH_ATTN_3_AVAILABLE
                and dropout_p == 0.0
                and window_size == (-1, -1)
                and not causal
                and dtype in half_dtypes
                and q.device.type == "cuda"
                and q.size(-1) <= 256
                and 0 < k_len <= k_length
            ):
                return run_dense_fa3(q, k[:, :k_len], v[:, :k_len], q.dtype)
            return original_flash_attention(
                q=q,
                k=k,
                v=v,
                q_lens=q_lens,
                k_lens=k_lens,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                q_scale=q_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
                dtype=dtype,
                version=version,
            )

        if dtype not in half_dtypes:
            raise RuntimeError("cache-attention-seqlens requires half attention dtype")
        if q.device.type != "cuda" or q.size(-1) > 256:
            return original_flash_attention(
                q=q,
                k=k,
                v=v,
                q_lens=q_lens,
                k_lens=k_lens,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                q_scale=q_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
                dtype=dtype,
                version=version,
            )

        batch_size, q_length, k_length = q.size(0), q.size(1), k.size(1)
        out_dtype = q.dtype

        if (
            use_fixed_fa3_batch1
            and (version is None or version == 3)
            and attention_module.FLASH_ATTN_3_AVAILABLE
            and dropout_p == 0.0
            and window_size == (-1, -1)
        ):
            return run_dense_fa3(q, k, v, out_dtype)

        q = half(q.flatten(0, 1))
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        q = q.to(v.dtype)
        k = k.to(v.dtype)
        if q_scale is not None:
            q = q * q_scale

        q_lens_cached = cached_lens(q.device, batch_size, q_length)
        k_lens_cached = cached_lens(k.device, batch_size, k_length)
        cu_q = cached_cu_seqlens(q.device, batch_size, q_length)
        cu_k = cached_cu_seqlens(k.device, batch_size, k_length)

        if (
            version is None or version == 3
        ) and attention_module.FLASH_ATTN_3_AVAILABLE:
            x = attention_module.flash_attn_interface.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                seqused_q=None,
                seqused_k=None,
                max_seqlen_q=q_length,
                max_seqlen_k=k_length,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic,
            )[0].unflatten(0, (batch_size, q_length))
        else:
            x = attention_module.flash_attn.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=q_length,
                max_seqlen_k=k_length,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
            ).unflatten(0, (batch_size, q_length))

        _ = (q_lens_cached, k_lens_cached)
        return x.type(out_dtype)

    attention_module.flash_attention = cached_flash_attention
    model_module.flash_attention = cached_flash_attention
    attention_module._sf_prewarm_flash_attention_seqlens = prewarm_flash_attention_seqlens
    model_module._sf_prewarm_flash_attention_seqlens = prewarm_flash_attention_seqlens


def expected_attention_seqlen_cache_lengths(
    num_latent_frames: int,
    num_frame_per_block: int,
    local_attn_size: int,
) -> list[int]:
    lengths = {512}
    cumulative_tokens = 0
    max_local_tokens = (
        local_attn_size * WAN_TOKENS_PER_LATENT_FRAME if local_attn_size != -1 else None
    )
    for count in chunk_frame_counts(num_latent_frames, num_frame_per_block):
        chunk_tokens = count * WAN_TOKENS_PER_LATENT_FRAME
        cumulative_tokens += chunk_tokens
        lengths.add(chunk_tokens)
        lengths.add(
            min(cumulative_tokens, max_local_tokens)
            if max_local_tokens is not None
            else cumulative_tokens
        )
    return sorted(set(lengths))


def prewarm_flash_attention_seqlen_cache(
    *,
    device: torch.device,
    batch_size: int,
    sequence_lengths: list[int],
) -> None:
    import wan.modules.attention as attention_module

    prewarm = getattr(attention_module, "_sf_prewarm_flash_attention_seqlens", None)
    if prewarm is None:
        raise RuntimeError("prewarm-attention-seqlens requires cache-attention-seqlens")
    prewarm(device, batch_size, sequence_lengths)


def install_causal_rope_triton_fp32_cache(
    transformers,
    *,
    triton_layers,
    query_only,
):
    import wan.modules.causal_model as causal_model

    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise RuntimeError("causal-rope-triton-fp32 requires Triton") from exc

    if hasattr(causal_model, "_sf_original_causal_rope_apply"):
        return

    original_causal_rope_apply = causal_model.causal_rope_apply
    causal_model._sf_original_causal_rope_apply = original_causal_rope_apply
    trig_cache = {}
    exact_freq_cache = {}

    @triton.jit
    def _causal_rope_fp32_kernel(
        x_ptr,
        out_ptr,
        cos_ptr,
        sin_ptr,
        x_stride_s: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_d: tl.constexpr,
        out_stride_s: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_d: tl.constexpr,
        cos_stride_s: tl.constexpr,
        seq_len: tl.constexpr,
        num_heads: tl.constexpr,
        half_dim: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (seq_len * num_heads * half_dim)
        half_idx = offsets % half_dim
        head_idx = (offsets // half_dim) % num_heads
        token_idx = offsets // (num_heads * half_dim)

        x_base = token_idx * x_stride_s + head_idx * x_stride_h + (2 * half_idx) * x_stride_d
        out_base = (
            token_idx * out_stride_s
            + head_idx * out_stride_h
            + (2 * half_idx) * out_stride_d
        )
        freq_base = token_idx * cos_stride_s + half_idx

        x_even = tl.load(x_ptr + x_base, mask=mask, other=0.0).to(tl.float32)
        x_odd = tl.load(x_ptr + x_base + x_stride_d, mask=mask, other=0.0).to(tl.float32)
        cos = tl.load(cos_ptr + freq_base, mask=mask, other=1.0)
        sin = tl.load(sin_ptr + freq_base, mask=mask, other=0.0)

        y_even = x_even * cos - x_odd * sin
        y_odd = x_even * sin + x_odd * cos
        tl.store(out_ptr + out_base, y_even, mask=mask)
        tl.store(out_ptr + out_base + out_stride_d, y_odd, mask=mask)

    def triton_fp32_causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
        num_heads, half_dim = x.size(2), x.size(3) // 2
        split_freqs = freqs.split(
            [half_dim - 2 * (half_dim // 3), half_dim // 3, half_dim // 3],
            dim=1,
        )
        output = torch.empty_like(x)
        device = freqs.device
        device_index = -1 if device.index is None else device.index
        for index, (frames, height, width) in enumerate(grid_sizes.tolist()):
            seq_len = frames * height * width
            key = (
                device.type,
                device_index,
                int(freqs.data_ptr()),
                int(start_frame),
                int(frames),
                int(height),
                int(width),
                int(half_dim),
            )
            cached = trig_cache.get(key)
            if cached is None:
                freqs_i = torch.cat(
                    [
                        split_freqs[0][start_frame : start_frame + frames]
                        .view(frames, 1, 1, -1)
                        .expand(frames, height, width, -1),
                        split_freqs[1][:height]
                        .view(1, height, 1, -1)
                        .expand(frames, height, width, -1),
                        split_freqs[2][:width]
                        .view(1, 1, width, -1)
                        .expand(frames, height, width, -1),
                    ],
                    dim=-1,
                ).reshape(seq_len, -1)
                cached = (
                    freqs_i.real.float().contiguous(),
                    freqs_i.imag.float().contiguous(),
                )
                trig_cache[key] = cached
            cos_i, sin_i = cached
            x_i = x[index]
            output_i = output[index]
            total = seq_len * num_heads * half_dim
            block_size = 256
            _causal_rope_fp32_kernel[(triton.cdiv(total, block_size),)](
                x_i,
                output_i,
                cos_i,
                sin_i,
                x_i.stride(0),
                x_i.stride(1),
                x_i.stride(2),
                output_i.stride(0),
                output_i.stride(1),
                output_i.stride(2),
                cos_i.stride(0),
                seq_len,
                num_heads,
                half_dim,
                BLOCK_SIZE=block_size,
            )
            if seq_len < x.size(1):
                output_i[seq_len:].copy_(x_i[seq_len:])
        return output

    def exact_cached_causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
        num_heads, half_dim = x.size(2), x.size(3) // 2
        split_freqs = freqs.split(
            [half_dim - 2 * (half_dim // 3), half_dim // 3, half_dim // 3],
            dim=1,
        )
        output = []
        device = freqs.device
        device_index = -1 if device.index is None else device.index
        for index, (frames, height, width) in enumerate(grid_sizes.tolist()):
            seq_len = frames * height * width
            x_i = torch.view_as_complex(
                x[index, :seq_len].to(torch.float64).reshape(seq_len, num_heads, -1, 2)
            )
            key = (
                device.type,
                device_index,
                int(freqs.data_ptr()),
                int(start_frame),
                int(frames),
                int(height),
                int(width),
                int(half_dim),
            )
            freqs_i = exact_freq_cache.get(key)
            if freqs_i is None:
                freqs_i = torch.cat(
                    [
                        split_freqs[0][start_frame : start_frame + frames]
                        .view(frames, 1, 1, -1)
                        .expand(frames, height, width, -1),
                        split_freqs[1][:height]
                        .view(1, height, 1, -1)
                        .expand(frames, height, width, -1),
                        split_freqs[2][:width]
                        .view(1, 1, width, -1)
                        .expand(frames, height, width, -1),
                    ],
                    dim=-1,
                ).reshape(seq_len, 1, -1)
                exact_freq_cache[key] = freqs_i
            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
            x_i = torch.cat([x_i, x[index, seq_len:]])
            output.append(x_i)
        return torch.stack(output).type_as(x)

    if triton_layers is not None or query_only:

        def selective_triton_fp32_causal_rope_apply(
            x,
            grid_sizes,
            freqs,
            start_frame=0,
        ):
            layer_index = getattr(causal_model, "_sf_current_causal_rope_layer", None)
            call_index = getattr(causal_model, "_sf_current_causal_rope_call", 0)
            causal_model._sf_current_causal_rope_call = call_index + 1
            if triton_layers is not None and layer_index not in triton_layers:
                return exact_cached_causal_rope_apply(
                    x,
                    grid_sizes,
                    freqs,
                    start_frame=start_frame,
                )
            if query_only and call_index != 0:
                return exact_cached_causal_rope_apply(
                    x,
                    grid_sizes,
                    freqs,
                    start_frame=start_frame,
                )
            return triton_fp32_causal_rope_apply(
                x,
                grid_sizes,
                freqs,
                start_frame=start_frame,
            )

        causal_model.causal_rope_apply = selective_triton_fp32_causal_rope_apply
        seen_attention_ids: set[int] = set()
        for transformer in transformers:
            model = wan_model(transformer)
            blocks = getattr(model, "blocks", None)
            if blocks is None:
                raise RuntimeError("selective/query-only Triton RoPE requires Wan blocks")
            for layer_index, block in enumerate(blocks):
                attention = getattr(block, "self_attn", None)
                if attention is None or id(attention) in seen_attention_ids:
                    continue
                seen_attention_ids.add(id(attention))
                if hasattr(attention, "_sf_rope_context_wrapped"):
                    continue
                original_forward = attention.forward

                def make_forward(forward_fn, current_layer_index: int):
                    def wrapped_forward(*args, **kwargs):
                        missing = object()
                        previous_layer = getattr(
                            causal_model,
                            "_sf_current_causal_rope_layer",
                            missing,
                        )
                        previous_call = getattr(
                            causal_model,
                            "_sf_current_causal_rope_call",
                            missing,
                        )
                        causal_model._sf_current_causal_rope_layer = current_layer_index
                        causal_model._sf_current_causal_rope_call = 0
                        try:
                            return forward_fn(*args, **kwargs)
                        finally:
                            if previous_layer is missing:
                                if hasattr(causal_model, "_sf_current_causal_rope_layer"):
                                    delattr(causal_model, "_sf_current_causal_rope_layer")
                            else:
                                causal_model._sf_current_causal_rope_layer = previous_layer
                            if previous_call is missing:
                                if hasattr(causal_model, "_sf_current_causal_rope_call"):
                                    delattr(causal_model, "_sf_current_causal_rope_call")
                            else:
                                causal_model._sf_current_causal_rope_call = previous_call

                    return wrapped_forward

                attention.forward = make_forward(original_forward, layer_index)
                attention._sf_rope_context_wrapped = True
    else:
        causal_model.causal_rope_apply = triton_fp32_causal_rope_apply


def enable_fast_vae_cache_clear(vae_model: Any) -> None:
    if getattr(vae_model, "_sf_fast_cache_clear_enabled", False):
        return
    original_clear_cache = vae_model.clear_cache
    original_clear_cache()
    conv_num = int(vae_model._conv_num)
    enc_conv_num = int(vae_model._enc_conv_num)

    def fast_clear_cache(module_self) -> None:
        module_self._conv_num = conv_num
        module_self._conv_idx = [0]
        module_self._feat_map = [None] * conv_num
        module_self._enc_conv_num = enc_conv_num
        module_self._enc_conv_idx = [0]
        module_self._enc_feat_map = [None] * enc_conv_num

    vae_model.clear_cache = types.MethodType(fast_clear_cache, vae_model)
    vae_model._sf_fast_cache_clear_enabled = True


def enable_vae_decode_runtime_fast_path(
    vae: Any,
    *,
    cache_scale_tensors: bool,
    batch1_fast_path: bool,
) -> None:
    if getattr(vae, "_sf_decode_runtime_fast_path_enabled", False):
        return
    scale_cache: dict[tuple[str, int, torch.dtype], list[torch.Tensor]] = {}

    def decode_to_pixel_fast(
        module_self,
        latent: torch.Tensor,
        use_cache: bool = False,
    ) -> torch.Tensor:
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        if cache_scale_tensors:
            device_index = -1 if device.index is None else int(device.index)
            key = (device.type, device_index, dtype)
            scale = scale_cache.get(key)
            if scale is None:
                scale = [
                    module_self.mean.to(device=device, dtype=dtype),
                    1.0 / module_self.std.to(device=device, dtype=dtype),
                ]
                scale_cache[key] = scale
        else:
            scale = [
                module_self.mean.to(device=device, dtype=dtype),
                1.0 / module_self.std.to(device=device, dtype=dtype),
            ]

        decode_function = (
            module_self.model.cached_decode if use_cache else module_self.model.decode
        )
        if batch1_fast_path and zs.shape[0] == 1:
            output = decode_function(zs, scale).float().clamp_(-1, 1)
            return output.permute(0, 2, 1, 3, 4)

        output = []
        for item in zs:
            output.append(
                decode_function(item.unsqueeze(0), scale)
                .float()
                .clamp_(-1, 1)
                .squeeze(0)
            )
        return torch.stack(output, dim=0).permute(0, 2, 1, 3, 4)

    vae.decode_to_pixel = types.MethodType(decode_to_pixel_fast, vae)
    vae._sf_decode_runtime_fast_path_enabled = True


def enable_cached_vae_list_cat_output(vae_model: Any) -> None:
    if getattr(vae_model, "_sf_cached_decode_list_cat_enabled", False):
        return

    def cached_decode_list_cat(module_self, z, scale):
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, module_self.z_dim, 1, 1, 1) + scale[0].view(
                1,
                module_self.z_dim,
                1,
                1,
                1,
            )
        else:
            z = z / scale[1] + scale[0]
        x = module_self.conv2(z)
        output_chunks = []
        for i in range(z.shape[2]):
            module_self._conv_idx = [0]
            output_chunks.append(
                module_self.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=module_self._feat_map,
                    feat_idx=module_self._conv_idx,
                )
            )
        return torch.cat(output_chunks, dim=2)

    vae_model.cached_decode = types.MethodType(cached_decode_list_cat, vae_model)
    vae_model._sf_cached_decode_list_cat_enabled = True


def enable_vae_channels_last_3d(vae_model: Any) -> None:
    if getattr(vae_model, "_sf_channels_last_3d_enabled", False):
        return
    for module in vae_model.modules():
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            module.weight.data = module.weight.detach().contiguous(
                memory_format=torch.channels_last_3d
            )
    vae_model._sf_channels_last_3d_enabled = True


def maybe_channels_last_3d(tensor: torch.Tensor, enabled: bool) -> torch.Tensor:
    if enabled:
        return tensor.contiguous(memory_format=torch.channels_last_3d)
    return tensor


def enable_cached_vae_block_decode(
    vae_model,
    *,
    max_block_size: int | None,
    prealloc_output: bool = False,
) -> None:
    if getattr(vae_model, "_sf_cached_decode_block_enabled", False):
        return

    def cached_decode_block(module_self, z, scale):
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, module_self.z_dim, 1, 1, 1) + scale[0].view(
                1,
                module_self.z_dim,
                1,
                1,
                1,
            )
        else:
            z = z / scale[1] + scale[0]
        channels_last_3d = bool(
            getattr(module_self, "_sf_channels_last_3d_enabled", False)
        )
        z = maybe_channels_last_3d(z, channels_last_3d)
        x = module_self.conv2(z)
        x = maybe_channels_last_3d(x, channels_last_3d)
        module_self._conv_idx = [0]
        first = module_self.decoder(
            maybe_channels_last_3d(x[:, :, :1, :, :], channels_last_3d),
            feat_cache=module_self._feat_map,
            feat_idx=module_self._conv_idx,
        )
        if x.shape[2] == 1:
            return first

        step = x.shape[2] - 1 if max_block_size is None else max_block_size
        if not prealloc_output:
            output_chunks = [first]
            for start_idx in range(1, x.shape[2], step):
                module_self._conv_idx = [0]
                output_chunks.append(
                    module_self.decoder(
                        maybe_channels_last_3d(
                            x[:, :, start_idx : start_idx + step, :, :],
                            channels_last_3d,
                        ),
                        feat_cache=module_self._feat_map,
                        feat_idx=module_self._conv_idx,
                    )
                )
            return torch.cat(output_chunks, dim=2)

        first_t = first.shape[2]
        first_warm_end = min(1 + step, x.shape[2])
        first_warm_latents = first_warm_end - 1
        module_self._conv_idx = [0]
        first_warm = module_self.decoder(
            maybe_channels_last_3d(x[:, :, 1:first_warm_end, :, :], channels_last_3d),
            feat_cache=module_self._feat_map,
            feat_idx=module_self._conv_idx,
        )
        if first_warm.shape[2] % first_warm_latents != 0:
            raise RuntimeError(
                "cached VAE block output temporal size is not divisible by "
                f"latent count: {first_warm.shape[2]} vs {first_warm_latents}"
            )
        frames_per_warm_latent = first_warm.shape[2] // first_warm_latents
        total_t = first_t + frames_per_warm_latent * (x.shape[2] - 1)
        output_shape = (
            first.shape[0],
            first.shape[1],
            total_t,
            first.shape[3],
            first.shape[4],
        )
        output = torch.empty(
            output_shape,
            dtype=first.dtype,
            device=first.device,
        )
        output[:, :, :first_t, :, :].copy_(first)
        write_t = first_t
        output[:, :, write_t : write_t + first_warm.shape[2], :, :].copy_(first_warm)
        write_t += first_warm.shape[2]
        for start_idx in range(first_warm_end, x.shape[2], step):
            end_idx = min(start_idx + step, x.shape[2])
            module_self._conv_idx = [0]
            current = module_self.decoder(
                maybe_channels_last_3d(
                    x[:, :, start_idx:end_idx, :, :], channels_last_3d
                ),
                feat_cache=module_self._feat_map,
                feat_idx=module_self._conv_idx,
            )
            expected_t = frames_per_warm_latent * (end_idx - start_idx)
            if current.shape[2] != expected_t:
                raise RuntimeError(
                    "cached VAE block output temporal size changed: "
                    f"{current.shape[2]} vs expected {expected_t}"
                )
            output[:, :, write_t : write_t + current.shape[2], :, :].copy_(current)
            write_t += current.shape[2]
        if write_t != total_t:
            raise RuntimeError(
                f"cached VAE block output wrote {write_t} frames, expected {total_t}"
            )
        return output

    vae_model.cached_decode = types.MethodType(cached_decode_block, vae_model)
    vae_model._sf_cached_decode_block_enabled = True


def vae_decode_z_to_pixel(
    vae: Any,
    latents_z: torch.Tensor,
    *,
    use_cache: bool,
    output_dtype: str,
    cache_scale: bool,
    batch1_fast_path: bool,
    scale_cache: dict[tuple[str, int, torch.dtype], list[torch.Tensor]],
) -> torch.Tensor:
    if use_cache:
        assert latents_z.shape[0] == 1, "Batch size must be 1 when using cache"

    device, dtype = latents_z.device, latents_z.dtype
    if cache_scale:
        device_index = -1 if device.index is None else int(device.index)
        key = (device.type, device_index, dtype)
        scale = scale_cache.get(key)
        if scale is None:
            scale = [
                vae.mean.to(device=device, dtype=dtype),
                1.0 / vae.std.to(device=device, dtype=dtype),
            ]
            scale_cache[key] = scale
    else:
        scale = [
            vae.mean.to(device=device, dtype=dtype),
            1.0 / vae.std.to(device=device, dtype=dtype),
        ]

    decode_function = vae.model.cached_decode if use_cache else vae.model.decode
    latents_z = maybe_channels_last_3d(
        latents_z,
        bool(getattr(vae.model, "_sf_channels_last_3d_enabled", False)),
    )
    keep_decoder_dtype = output_dtype == "decoder"
    if batch1_fast_path and latents_z.shape[0] == 1:
        output = decode_function(latents_z, scale).clamp_(-1, 1)
        if not keep_decoder_dtype:
            output = output.float()
        return output.permute(0, 2, 1, 3, 4)

    output = []
    for item in latents_z:
        decoded = decode_function(item.unsqueeze(0), scale).clamp_(-1, 1)
        if not keep_decoder_dtype:
            decoded = decoded.float()
        output.append(decoded.squeeze(0))
    return torch.stack(output, dim=0).permute(0, 2, 1, 3, 4)
