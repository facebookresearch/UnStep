#!/usr/bin/env python3
"""Minimal source definition of the current offline UnStep wrapper.

This file owns the method: the reduced denoising schedule, V/O SVD projection
update, local/sink attention policy, RoPE/VAE runtime stack selection, and
full-video decode call. Self-Forcing/Wan loading and low-level backend patches
live in wrapper_runtime.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from unstep import wrapper_runtime as runtime
from unstep.clean_sigma import install_shared_clean_sigma


@dataclass(frozen=True)
class WrapperConfig:
    """Current confirmed offline wrapper settings.

    The default config is the validated H100 two-step T2V setting.
    """

    base: runtime.BaseSFConfig = field(default_factory=runtime.BaseSFConfig)
    compile_mode: str = "max-autotune-no-cudagraphs"

    first_schedule: tuple[int, ...] = (0, 1, 2, 3)
    later_schedule: tuple[int, ...] = (0, 3)
    block_schedule_overrides: tuple[tuple[int, tuple[int, ...]], ...] = (
        (3, (0, 2)),
        (4, (0, 2)),
        (5, (0, 2)),
    )
    append_clean_final_step: bool = True

    local_attn_size: int = 9
    sink_size: int = 3
    local_layer_sizes: str = "8-29:8;8:9;15:9;23:9;29:9"

    svd_targets: str = "v,o"
    svd_keep_ratio: float = 0.92
    svd_layer_keep_ratios: str = "0-7:1.00;8-23:0.92;24-29:0.92"

    # None means direct all-layer Q/K Triton RoPE. Passing "0-29" gives the
    # earlier explicit-layer variant.
    rope_layers: str | None = None
    rope_query_only: bool = False

    vae_block_size: int = 5
    strict_attention_prewarm: bool = True
    use_fa3_batch1: bool = True
    use_fa3_slice_lens: bool = False
    full_decode_channel_first_latents: bool = True
    cache_timestep_tensors: bool = False

    full_vae_decode_dtype: str = "fp16"
    full_vae_output_dtype: str = "float"
    full_vae_use_cache: bool = True
    full_vae_fast_cache_clear: bool = True
    full_vae_skip_post_clear: bool = True
    cache_vae_scale_tensors: bool = True
    full_vae_batch1_fast_path: bool = True
    full_vae_list_cat_output: bool = True
    full_vae_block_cached_decode: bool = True
    full_vae_prealloc_block_output: bool = True
    full_vae_channels_last_3d: bool = True
    save_video_gpu_uint8: bool = True
    defer_video_copy_timing: bool = True
    preserve_cuda_cache_between_runs: bool = True
    clean_sigma_override: float | None = 0.02


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


def parse_projection_targets(raw: str) -> list[str]:
    supported = {"q", "k", "v", "o"}
    targets = [item.strip() for item in raw.split(",") if item.strip()]
    if not targets:
        raise RuntimeError("SVD target list is empty")
    invalid = [target for target in targets if target not in supported]
    if invalid:
        raise RuntimeError(f"unsupported SVD projection targets: {invalid}")
    return targets


def parse_layer_keep_ratio_spans(raw: str | None) -> list[tuple[int, int, float]] | None:
    if raw is None or raw.strip() == "":
        return None
    spans: list[tuple[int, int, float]] = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise RuntimeError("SVD layer spans must look like start-end:ratio")
        span_text, ratio_text = item.split(":", 1)
        if "-" in span_text:
            start_text, end_text = span_text.split("-", 1)
            start = int(start_text)
            end = int(end_text)
        else:
            start = end = int(span_text)
        ratio = float(ratio_text)
        if start < 0 or end < start:
            raise RuntimeError(f"invalid SVD layer span: {span_text}")
        if ratio <= 0.0 or ratio > 1.0:
            raise RuntimeError("SVD keep ratios must be in (0, 1]")
        spans.append((start, end, ratio))
    if not spans:
        raise RuntimeError("SVD layer keep-ratio spans were empty")
    return spans


def local_attention_sizes_from_spec(raw: str | None) -> set[int]:
    if raw is None or raw.strip() == "":
        return set()
    sizes: set[int] = set()
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise RuntimeError("local attention spans must look like start-end:size")
        _span_text, size_text = item.split(":", 1)
        size = int(size_text)
        if size <= 0:
            raise RuntimeError("local attention sizes must be positive")
        sizes.add(size)
    return sizes


def _linear_svd_rank(source: nn.Linear, keep_ratio: float) -> int:
    if keep_ratio <= 0.0 or keep_ratio > 1.0:
        raise RuntimeError("SVD keep ratio must be in (0, 1]")
    full_rank = min(source.in_features, source.out_features)
    return min(max(1, int(round(full_rank * keep_ratio))), full_rank)


def _dense_truncated_svd_linear(source: nn.Linear, rank: int) -> nn.Linear:
    dense = nn.Linear(
        source.in_features,
        source.out_features,
        bias=source.bias is not None,
        device=source.weight.device,
        dtype=source.weight.dtype,
    )
    with torch.no_grad():
        weight = source.weight.float()
        u, singular_values, vh = torch.linalg.svd(weight, full_matrices=False)
        u = u[:, :rank]
        singular_values = singular_values[:rank]
        vh = vh[:rank, :]
        dense.weight.copy_(
            ((u * singular_values.unsqueeze(0)) @ vh).to(
                device=source.weight.device,
                dtype=source.weight.dtype,
            )
        )
        if source.bias is not None:
            dense.bias.copy_(source.bias)
    return dense


def _ratio_for_layer(
    layer_index: int,
    default: float,
    spans: list[tuple[int, int, float]] | None,
) -> float:
    if spans is None:
        return default
    for start, end, ratio in spans:
        if start <= layer_index <= end:
            return ratio
    return default


def apply_attention_svd_dense(
    transformer: Any,
    *,
    keep_ratio: float,
    targets: list[str],
    layer_keep_ratio_spans: list[tuple[int, int, float]] | None,
) -> dict[str, Any]:
    """Materialize V/O SVD back into dense Linear layers.

    This is method logic, so it stays in wrapper.py rather than the glue file.
    """

    ranks: list[dict[str, Any]] = []
    model = runtime.wan_model(transformer)
    for layer_index, block in enumerate(model.blocks):
        self_attn = block.self_attn
        layer_ratio = _ratio_for_layer(layer_index, keep_ratio, layer_keep_ratio_spans)
        row: dict[str, Any] = {"block": layer_index, "keep_ratio": layer_ratio}
        for target in targets:
            module = getattr(self_attn, target)
            if not isinstance(module, nn.Linear):
                raise RuntimeError(f"expected self_attn.{target} to be nn.Linear")
            rank = _linear_svd_rank(module, layer_ratio)
            row[f"{target}_rank"] = rank
            if rank < min(module.in_features, module.out_features):
                setattr(self_attn, target, _dense_truncated_svd_linear(module, rank))
        ranks.append(row)
    return {"method_kind": "attn_vo_svd_dense", "attn_vo_ranks": ranks}


class OfflineSelfForcingWrapper:
    """Source-level implementation of the current offline wrapper method."""

    def __init__(self, config: WrapperConfig | None = None) -> None:
        self.config = config or WrapperConfig()

    def schedule_indices_for_block(self, block_idx: int) -> list[int]:
        if block_idx == 0:
            return list(self.config.first_schedule)
        overrides = dict(self.config.block_schedule_overrides)
        return list(overrides.get(block_idx, self.config.later_schedule))

    def resolve_timesteps(self, pipeline: Any, block_idx: int) -> list[Any]:
        full_timesteps = runtime.selected_timesteps(pipeline)
        timesteps = [full_timesteps[index] for index in self.schedule_indices_for_block(block_idx)]
        if self.config.append_clean_final_step and not runtime.is_zero_timestep(timesteps[-1]):
            timesteps.append(0.0)
        return timesteps

    def prepare_pipeline_transformer(
        self,
        *,
        transformer: Any,
        precision: str,
        device: torch.device,
        timestep_shift: float,
    ) -> tuple[Any, dict[str, Any]]:
        """Apply the method mutations to the pipeline DiT itself."""

        cfg = self.config
        model = runtime.wan_model(transformer)
        previous_num_frame_per_block = getattr(model, "num_frame_per_block", None)
        # Preserve CausalWanModel's causal-mask block size of 1 while the
        # pipeline chunks latents in groups of 3.
        model.num_frame_per_block = 1
        info = apply_attention_svd_dense(
            transformer,
            keep_ratio=cfg.svd_keep_ratio,
            targets=parse_projection_targets(cfg.svd_targets),
            layer_keep_ratio_spans=parse_layer_keep_ratio_spans(
                cfg.svd_layer_keep_ratios
            ),
        )
        info["single_transformer"] = True
        info["transformer_kind"] = "pipeline_generator_attn_vo_svd_dense"
        info["model_num_frame_per_block_before_reuse"] = (
            previous_num_frame_per_block
        )
        info["model_num_frame_per_block"] = model.num_frame_per_block
        info.update(runtime.reset_transformer_scheduler(transformer, timestep_shift))
        transformer.eval().requires_grad_(False)
        transformer.to(device=device, dtype=runtime.torch_dtype(precision))
        return transformer, info

    def install_runtime(
        self,
        *,
        pipeline: Any,
        target_transformer: Any,
        method_transformer: Any,
        device: torch.device,
    ) -> None:
        """Install the quality-preserving runtime stack used by the 47 FPS keeper."""

        cfg = self.config
        torch.backends.cudnn.benchmark = True

        pipeline.local_attn_size = cfg.local_attn_size
        pipeline.sink_size = cfg.sink_size
        for transformer in self._unique_modules(target_transformer, method_transformer):
            model = runtime.wan_model(transformer)
            model.local_attn_size = cfg.local_attn_size
            model.sink_size = cfg.sink_size
            for block in model.blocks:
                block.self_attn.local_attn_size = cfg.local_attn_size
                block.self_attn.max_attention_size = (
                    cfg.local_attn_size * runtime.WAN_TOKENS_PER_LATENT_FRAME
                )
                block.self_attn.sink_size = cfg.sink_size

        if cfg.full_vae_decode_dtype != "fp16":
            raise RuntimeError("UnStep wrapper expects full_vae_decode_dtype='fp16'")
        pipeline.vae.to(device=device, dtype=torch.float16)
        runtime.apply_layerwise_local_attention(
            target_transformer,
            pipeline,
            cfg.local_layer_sizes,
        )
        if method_transformer is not target_transformer:
            runtime.copy_local_attention_settings(target_transformer, method_transformer)

        for transformer in self._unique_modules(target_transformer, method_transformer):
            runtime.enable_cached_scheduler_sigmas(
                transformer,
                clean_sigma_override=cfg.clean_sigma_override,
            )

        self.clean_sigma_input_validation = None
        if cfg.clean_sigma_override is not None:
            self.clean_sigma_input_validation = install_shared_clean_sigma(
                pipeline, method_transformer, cfg.clean_sigma_override, device
            )

        runtime.install_flash_attention_seqlen_cache(
            use_fixed_fa3_batch1=cfg.use_fa3_batch1,
            use_fixed_fa3_batch1_slice_lens=cfg.use_fa3_slice_lens,
        )
        if cfg.strict_attention_prewarm:
            attention_lengths = set(
                runtime.expected_attention_seqlen_cache_lengths(
                    cfg.base.num_latent_frames,
                    cfg.base.num_frame_per_block,
                    cfg.local_attn_size,
                )
            )
        else:
            attention_lengths: set[int] = set()
            for local_size in {
                cfg.local_attn_size,
                *local_attention_sizes_from_spec(cfg.local_layer_sizes),
            }:
                attention_lengths.update(
                    runtime.expected_attention_seqlen_cache_lengths(
                        cfg.base.num_latent_frames,
                        cfg.base.num_frame_per_block,
                        local_size,
                    )
                )
            attention_lengths.update(
                frame_count * runtime.WAN_TOKENS_PER_LATENT_FRAME
                for frame_count in range(1, cfg.base.num_latent_frames + 1)
            )
        runtime.prewarm_flash_attention_seqlen_cache(
            device=device,
            batch_size=1,
            sequence_lengths=sorted(attention_lengths),
        )

        triton_layers = (
            None
            if cfg.rope_layers is None
            else set(
                parse_index_range_list(
                    cfg.rope_layers,
                    name="WrapperConfig.rope_layers",
                )
                or []
            )
        )
        runtime.install_causal_rope_triton_fp32_cache(
            list(self._unique_modules(target_transformer, method_transformer)),
            triton_layers=triton_layers,
            query_only=cfg.rope_query_only,
        )

        vae_model = pipeline.vae.model
        if cfg.full_vae_fast_cache_clear:
            runtime.enable_fast_vae_cache_clear(vae_model)
        runtime.enable_vae_decode_runtime_fast_path(
            pipeline.vae,
            cache_scale_tensors=cfg.cache_vae_scale_tensors,
            batch1_fast_path=cfg.full_vae_batch1_fast_path,
        )
        if cfg.full_vae_list_cat_output:
            runtime.enable_cached_vae_list_cat_output(vae_model)
        if cfg.full_vae_channels_last_3d:
            runtime.enable_vae_channels_last_3d(vae_model)
        if cfg.full_vae_block_cached_decode:
            runtime.enable_cached_vae_block_decode(
                vae_model,
                max_block_size=cfg.vae_block_size,
                prealloc_output=cfg.full_vae_prealloc_block_output,
            )

        for transformer in self._unique_modules(target_transformer, method_transformer):
            transformer.compile(mode=cfg.compile_mode)

        if not getattr(vae_model, "_sf_compiled_decode", False):
            vae_model.decode = torch.compile(vae_model.decode, mode=cfg.compile_mode)
            vae_model.cached_decode = torch.compile(
                vae_model.cached_decode,
                mode=cfg.compile_mode,
            )
            vae_model._sf_compiled_decode = True

    def denoise_chunk(
        self,
        *,
        transformer: Any,
        pipeline: Any,
        noisy_input: torch.Tensor,
        conditional_dict: dict[str, torch.Tensor],
        kv_cache: list[dict[str, Any]],
        crossattn_cache: list[dict[str, Any]],
        current_start_frame: int,
        block_idx: int,
        generator: torch.Generator | None,
        timestep_cache: dict[tuple[tuple[int, ...], str, int, int | float], torch.Tensor]
        | None = None,
        timesteps_override: list[Any] | None = None,
        num_blocks: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        denoised_pred = noisy_input
        forward_count = 0
        batch_size = int(noisy_input.shape[0])
        current_num_frames = int(noisy_input.shape[1])
        timesteps = (
            timesteps_override
            if timesteps_override is not None
            else self.resolve_timesteps(pipeline, block_idx)
        )
        if num_blocks is None:
            num_blocks = len(
                runtime.chunk_frame_counts(
                    self.config.base.num_latent_frames,
                    self.config.base.num_frame_per_block,
                )
            )

        runtime.set_step_layer_skip_route_active(transformer, False)
        try:
            for step_idx, current_timestep in enumerate(timesteps):
                runtime.set_cross_step_kv_context(
                    transformer,
                    current_start_frame=current_start_frame,
                    step_index=step_idx,
                    timestep=current_timestep,
                    active=True,
                    block_idx=block_idx,
                    num_blocks=num_blocks,
                )
                timestep = runtime.timestep_tensor(
                    timestep_cache,
                    (batch_size, current_num_frames),
                    current_timestep,
                    noisy_input.device,
                )
                _, denoised_pred = transformer(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_frame * pipeline.frame_seq_length,
                )
                forward_count += 1
                if step_idx != len(timesteps) - 1:
                    noisy_input = runtime.add_noise_transition(
                        pipeline=pipeline,
                        denoised_pred=denoised_pred,
                        next_timestep=timesteps[step_idx + 1],
                        generator=generator,
                        timestep_cache=timestep_cache,
                    )
        finally:
            runtime.set_step_layer_skip_route_active(transformer, False)
        return denoised_pred, forward_count

    def generate_latents(
        self,
        *,
        pipeline: Any,
        method_transformer: Any,
        text_encoder: Any | None = None,
        prompt: str,
        noise: torch.Tensor,
        seed: int = 0,
        conditional_dict: dict[str, torch.Tensor] | None = None,
        kv_cache: list[dict[str, Any]] | None = None,
        crossattn_cache: list[dict[str, Any]] | None = None,
        use_global_rng: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        cfg = self.config
        chunk_counts = runtime.chunk_frame_counts(
            cfg.base.num_latent_frames,
            cfg.base.num_frame_per_block,
        )
        active_kv_cache_size = runtime.kv_cache_size_for_pipeline(
            pipeline,
            cfg.base.num_latent_frames,
        )
        allow_local_attn_eviction = pipeline.local_attn_size != -1
        if kv_cache is None:
            kv_cache = runtime.make_kv_cache(
                pipeline,
                batch_size=1,
                dtype=noise.dtype,
                device=noise.device,
                init="empty",
                cursor="cpu_tensor",
                kv_cache_size=active_kv_cache_size,
                allow_local_attn_eviction=allow_local_attn_eviction,
            )
        if crossattn_cache is None:
            crossattn_cache = runtime.make_crossattn_cache(
                pipeline,
                batch_size=1,
                dtype=noise.dtype,
                device=noise.device,
                init="empty",
            )
        if conditional_dict is None:
            if text_encoder is None:
                raise RuntimeError("generate_latents requires text_encoder or conditional_dict")
            conditional_dict = text_encoder(text_prompts=[prompt])
        generator = (
            None
            if use_global_rng
            else torch.Generator(device=noise.device).manual_seed(seed + 1_000_003)
        )
        chunks = []
        channel_first_latents: torch.Tensor | None = None
        current_start = 0
        total_forwards = 0
        timestep_cache: dict[tuple[tuple[int, ...], str, int, int | float], torch.Tensor] | None
        timestep_cache = {} if cfg.cache_timestep_tensors else None
        for block_idx, current_num_frames in enumerate(chunk_counts):
            noisy_input = noise[:, current_start : current_start + current_num_frames]
            denoised, forwards = self.denoise_chunk(
                transformer=method_transformer,
                pipeline=pipeline,
                noisy_input=noisy_input,
                conditional_dict=conditional_dict,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start_frame=current_start,
                block_idx=block_idx,
                generator=generator,
                timestep_cache=timestep_cache,
            )
            if cfg.full_decode_channel_first_latents:
                if channel_first_latents is None:
                    latent_shape = (
                        denoised.shape[0],
                        denoised.shape[2],
                        cfg.base.num_latent_frames,
                        *denoised.shape[3:],
                    )
                    channel_first_latents = torch.empty(
                        latent_shape,
                        dtype=denoised.dtype,
                        device=denoised.device,
                    )
                channel_first_latents[
                    :, :, current_start : current_start + current_num_frames
                ].copy_(denoised.permute(0, 2, 1, 3, 4))
            chunks.append(denoised)
            current_start += current_num_frames
            total_forwards += forwards
        if cfg.full_decode_channel_first_latents:
            if channel_first_latents is None:
                raise RuntimeError("channel-first latent buffer was not allocated")
            latents = channel_first_latents
            layout = "bcthw"
        else:
            latents = torch.cat(chunks, dim=1)
            layout = "btchw"
        return latents, {
            "total_transformer_forward_count": total_forwards,
            "latent_layout": layout,
            "_latent_chunk_refs": chunks,
        }

    @staticmethod
    def finalize_decoded_pixels(pixels: torch.Tensor) -> torch.Tensor:
        return (pixels * 0.5 + 0.5).clamp(0, 1).permute(0, 1, 3, 4, 2)

    def decode_full_video(
        self,
        pipeline: Any,
        latents: torch.Tensor,
        *,
        finalize_output: bool = True,
    ) -> torch.Tensor:
        z_latents = (
            latents
            if self.config.full_decode_channel_first_latents
            else latents.permute(0, 2, 1, 3, 4)
        )
        if self.config.full_vae_decode_dtype == "bf16":
            z_latents = z_latents.bfloat16()
        elif self.config.full_vae_decode_dtype == "fp16":
            z_latents = z_latents.half()
        if (
            self.config.full_vae_use_cache
            and getattr(pipeline.vae.model, "clear_cache", None) is not None
        ):
            pipeline.vae.model.clear_cache()
        pixels = runtime.vae_decode_z_to_pixel(
            pipeline.vae,
            z_latents,
            use_cache=self.config.full_vae_use_cache,
            output_dtype=self.config.full_vae_output_dtype,
            cache_scale=self.config.cache_vae_scale_tensors,
            batch1_fast_path=self.config.full_vae_batch1_fast_path,
            scale_cache={},
        )
        if (
            self.config.full_vae_use_cache
            and not self.config.full_vae_skip_post_clear
            and getattr(pipeline.vae.model, "clear_cache", None) is not None
        ):
            pipeline.vae.model.clear_cache()
        if finalize_output:
            return self.finalize_decoded_pixels(pixels)
        return pixels

    @staticmethod
    def _unique_modules(*modules: Any) -> tuple[Any, ...]:
        seen: set[int] = set()
        unique = []
        for module in modules:
            if id(module) not in seen:
                unique.append(module)
                seen.add(id(module))
        return tuple(unique)
