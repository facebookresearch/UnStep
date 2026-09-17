#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
from torchvision.io import write_video

from unstep import wrapper_runtime as runtime
from unstep.wrapper import OfflineSelfForcingWrapper, WrapperConfig


ROOT = Path(__file__).resolve().parent.parent
UNSTEP_TIMESTEP_SHIFT = 5.0
_ORIGINAL_RMTREE = shutil.rmtree
_PROTECTED_RMTREE_ROOTS = (
    ROOT / "results",
)


def guarded_rmtree(path, *args, **kwargs):
    path_resolved = Path(path).resolve(strict=False)
    for protected_root in _PROTECTED_RMTREE_ROOTS:
        protected = protected_root.resolve(strict=False)
        if (path_resolved == protected or protected in path_resolved.parents
                or path_resolved in protected.parents):
            raise RuntimeError(
                f"refusing to recursively delete protected output path: {path_resolved}"
            )
    return _ORIGINAL_RMTREE(path, *args, **kwargs)


shutil.rmtree = guarded_rmtree


def assert_safe_cleanup_dir(
    workdir: Path, *, video_dir: Path, out_json: Path, input_paths: tuple[Path, ...] = (),
) -> None:
    workdir_resolved = workdir.resolve(strict=False)
    for protected_path in (ROOT, *input_paths):
        protected = protected_path.resolve(strict=False)
        if workdir_resolved.is_relative_to(protected) or protected.is_relative_to(workdir_resolved):
            raise RuntimeError(
                f"unsafe --workdir cleanup target {workdir_resolved} overlaps protected path {protected}"
            )
    protected_dirs = [
        ("video_dir", video_dir.resolve(strict=False)),
        ("out_json_dir", out_json.parent.resolve(strict=False)),
    ]
    for label, protected in protected_dirs:
        if workdir_resolved == protected or workdir_resolved in protected.parents:
            raise RuntimeError(
                f"unsafe --workdir cleanup target {workdir_resolved} overlaps {label} "
                f"{protected}; use a per-shard workdir that is not an ancestor of outputs"
            )


def assert_safe_speed_video_cleanup_dir(
    video_dir: Path, *, run_name: str, out_json: Path, input_paths: tuple[Path, ...] = (),
) -> None:
    video_dir_resolved = video_dir.resolve(strict=False)
    if video_dir_resolved == Path("/") or str(video_dir_resolved) in {"", "/tmp"}:
        raise RuntimeError(f"refusing unsafe video cleanup path: {video_dir_resolved}")
    if run_name not in video_dir_resolved.parts:
        raise RuntimeError(
            f"refusing speed-only video cleanup path {video_dir_resolved}; "
            f"path must contain run name {run_name!r}"
        )
    if out_json.resolve(strict=False).is_relative_to(video_dir_resolved):
        raise RuntimeError("speed-only video cleanup would delete --out-json; place it outside --video-dir")
    for protected_root in (ROOT, *_PROTECTED_RMTREE_ROOTS):
        if protected_root.resolve(strict=False).is_relative_to(video_dir_resolved):
            raise RuntimeError(f"speed-only video cleanup would delete protected path: {protected_root}")
    protected_paths = input_paths
    if not video_dir_resolved.is_relative_to(ROOT.resolve(strict=False) / "results"):
        protected_paths = (ROOT, *protected_paths)
    for protected_path in protected_paths:
        protected = protected_path.resolve(strict=False)
        if video_dir_resolved.is_relative_to(protected) or protected.is_relative_to(video_dir_resolved):
            raise RuntimeError(f"speed-only video cleanup overlaps protected path: {protected}")


def parse_indices(raw: str, total_count: int) -> list[int]:
    raw = raw.strip()
    if raw == "all":
        return list(range(total_count))
    indices = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not indices:
        raise RuntimeError("--prompt-indices resolved to an empty list")
    if any(index < 0 or index >= total_count for index in indices):
        raise RuntimeError(f"prompt indices must be in [0, {total_count})")
    if len(set(indices)) != len(indices):
        raise RuntimeError("prompt indices must be unique")
    return indices


def load_prompt_lines(manager, prompt_uri: str) -> list[str]:
    prompt_path = Path(manager.get_local_path(prompt_uri))
    return [
        line.rstrip()
        for line in prompt_path.read_text(encoding="utf-8").splitlines()
    ]


def shard_indices(
    indices: list[int],
    *,
    shard_index: int,
    num_shards: int,
    shard_mode: str,
) -> list[int]:
    if num_shards <= 0:
        raise RuntimeError("--num-shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise RuntimeError("--shard-index must be in [0, num_shards)")
    if shard_mode == "modulo":
        return [index for index in indices if index % num_shards == shard_index]
    if shard_mode == "contiguous":
        start = (len(indices) * shard_index) // num_shards
        end = (len(indices) * (shard_index + 1)) // num_shards
        return indices[start:end]
    raise RuntimeError(f"unsupported shard mode: {shard_mode}")


def should_shard_prompt_indices(raw_prompt_indices: str, num_shards: int, allow_subset_sharding: bool) -> bool:
    if raw_prompt_indices.strip() == "all":
        return True
    if num_shards == 1:
        return False
    if not allow_subset_sharding:
        raise RuntimeError(
            "refusing to shard an explicit --prompt-indices list. Pass each "
            "shard-local comma list with --num-shards 1, or use --prompt-indices all "
            "with --num-shards/--shard-index. Use --allow-subset-sharding only for "
            "intentional nested subset sharding."
        )
    return True


def make_wrapper_config(args: argparse.Namespace, backend: str) -> WrapperConfig:
    cfg = WrapperConfig()
    cfg = replace(cfg, base=replace(cfg.base, seed_mode=args.seed_mode))
    if args.disable_block_schedule_overrides:
        cfg = replace(cfg, block_schedule_overrides=())
    if args.clean_sigma_override is not None:
        cfg = replace(cfg, clean_sigma_override=float(args.clean_sigma_override))
    if backend == "fa2":
        cfg = replace(cfg, use_fa3_batch1=False, use_fa3_slice_lens=False)
    return cfg


def make_prepare_args(args: argparse.Namespace, cfg: WrapperConfig, backend: str) -> argparse.Namespace:
    return argparse.Namespace(
        source_uri=str(args.source_uri),
        weights_uri=str(args.weights_uri),
        vae_uri=str(args.vae_uri),
        checkpoint_uri=args.checkpoint_uri,
        skip_hf_download=True,
        num_frame_per_block=cfg.base.num_frame_per_block,
        local_attn_size=cfg.local_attn_size,
        sink_size=cfg.sink_size,
        enable_flash_attn3=backend != "fa2",
        fa3_import_backend=backend,
        timestep_shift=UNSTEP_TIMESTEP_SHIFT,
        skip_cache_cursor_patch=False,
        cache_text_projection=False,
    )


def save_video(path: Path, video: torch.Tensor, fps: int) -> None:
    if video.shape[0] != 1:
        raise RuntimeError(f"expected batch size 1 video, got {tuple(video.shape)}")
    frames = video[0]
    if frames.dtype != torch.uint8:
        frames = (frames * 255.0).clamp(0, 255).to(torch.uint8)
    if frames.device.type != "cpu":
        frames = frames.cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_video(str(path), frames, fps=fps)
        return
    except RuntimeError as exc:
        if "write_video is unavailable" not in str(exc):
            raise

    frames_np = frames.contiguous().numpy()
    height, width = int(frames_np.shape[1]), int(frames_np.shape[2])
    ffmpeg = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if not ffmpeg:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-loglevel",
        "error",
        str(path),
    ]
    subprocess.run(cmd, input=frames_np.tobytes(), check=True)


def consume_skipped_prompt_rng(
    *,
    cfg: WrapperConfig,
    prompt_index: int,
    device: torch.device,
    dtype: torch.dtype,
    mode: str,
) -> dict[str, object]:
    if mode == "none":
        return {
            "prompt_index": prompt_index,
            "initial_noise_draws": 0,
            "transition_noise_draws": 0,
            "transition_noise_shape_counts": {},
            "timestep_counts_by_block": [],
            "rng_skip_contract": "producer_no_skipped_prompt_rng",
        }
    if mode not in {"initial_only", "producer_transition_noise"}:
        raise RuntimeError(f"unsupported rng skip mode: {mode}")
    torch.randn(
        [1, cfg.base.num_latent_frames, 16, 60, 104],
        device=device,
        dtype=dtype,
    )
    if mode == "producer_transition_noise":
        transition_draws = 0
        transition_shape_counts: dict[str, int] = {}
        timestep_counts_by_block: list[int] = []
        schedule_overrides = dict(cfg.block_schedule_overrides)
        chunk_counts = runtime.chunk_frame_counts(
            cfg.base.num_latent_frames,
            cfg.base.num_frame_per_block,
        )
        for block_idx, current_num_frames in enumerate(chunk_counts):
            if block_idx in schedule_overrides:
                active_schedule = schedule_overrides[block_idx]
            elif block_idx == 0:
                active_schedule = cfg.first_schedule
            else:
                active_schedule = cfg.later_schedule
            # The old working driver skipped RNG from the actual active
            # timestep list after append-clean had been applied. That means the
            # final clean timestep contributes to len(active_timesteps) and
            # therefore to the number of transition noise draws burned here.
            active_timestep_count = len(active_schedule)
            if cfg.append_clean_final_step:
                active_timestep_count += 1
            timestep_counts_by_block.append(active_timestep_count)
            for _ in range(max(active_timestep_count - 1, 0)):
                shape = [current_num_frames, 16, 60, 104]
                torch.randn(shape, device=device, dtype=dtype)
                transition_draws += 1
                key = "x".join(str(item) for item in shape)
                transition_shape_counts[key] = transition_shape_counts.get(key, 0) + 1
        return {
            "prompt_index": prompt_index,
            "initial_noise_draws": 1,
            "transition_noise_draws": transition_draws,
            "transition_noise_shape_counts": transition_shape_counts,
            "timestep_counts_by_block": timestep_counts_by_block,
            "rng_skip_contract": "producer_transition_noise_with_clean_append",
        }
    return {
        "prompt_index": prompt_index,
        "initial_noise_draws": 1,
        "transition_noise_draws": 0,
        "transition_noise_shape_counts": {},
        "timestep_counts_by_block": [],
        "rng_skip_contract": "producer_initial_noise_only",
    }


def draw_prompt_noise(
    *,
    cfg: WrapperConfig,
    prompt_index: int,
    device: torch.device,
    dtype: torch.dtype,
    seed_mode: str,
) -> tuple[torch.Tensor, int, bool]:
    if seed_mode == "sequential_global":
        sample_seed = cfg.base.seed
        noise = torch.randn(
            [1, cfg.base.num_latent_frames, 16, 60, 104],
            device=device,
            dtype=dtype,
        )
        return noise, sample_seed, True

    sample_seed = cfg.base.seed + prompt_index * 1000
    generator = torch.Generator(device=device).manual_seed(sample_seed)
    noise = torch.randn(
        [1, cfg.base.num_latent_frames, 16, 60, 104],
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return noise, sample_seed, False


def make_conditioning(text_encoder, prompt: str) -> dict[str, torch.Tensor]:
    with torch.inference_mode():
        return text_encoder(text_prompts=[prompt])


def run_prompt(
    *,
    wrapper: OfflineSelfForcingWrapper,
    pipeline,
    method_transformer,
    text_encoder,
    prompt: str,
    prompt_index: int,
    cfg: WrapperConfig,
    device: torch.device,
    noise: torch.Tensor,
    sample_seed: int,
    use_global_rng: bool,
    exclude_final_output_layout_timing: bool,
) -> tuple[torch.Tensor, dict[str, object]]:
    active_kv_cache_size = runtime.kv_cache_size_for_pipeline(
        pipeline,
        cfg.base.num_latent_frames,
    )
    allow_local_attn_eviction = pipeline.local_attn_size != -1
    kv_cache = runtime.make_kv_cache(
        pipeline,
        batch_size=1,
        dtype=noise.dtype,
        device=device,
        init="empty",
        cursor="cpu_tensor",
        kv_cache_size=active_kv_cache_size,
        allow_local_attn_eviction=allow_local_attn_eviction,
    )
    crossattn_cache = runtime.make_crossattn_cache(
        pipeline,
        batch_size=1,
        dtype=noise.dtype,
        device=device,
        init="empty",
    )
    conditional_dict = make_conditioning(text_encoder, prompt)

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        latents, info = wrapper.generate_latents(
            pipeline=pipeline,
            method_transformer=method_transformer,
            prompt=prompt,
            conditional_dict=conditional_dict,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            noise=noise,
            seed=sample_seed,
            use_global_rng=use_global_rng,
        )
        pixels = wrapper.decode_full_video(
            pipeline,
            latents,
            finalize_output=not exclude_final_output_layout_timing,
        )
    torch.cuda.synchronize()
    total_sec = time.perf_counter() - start
    if exclude_final_output_layout_timing:
        with torch.inference_mode():
            pixels = wrapper.finalize_decoded_pixels(pixels)

    decoded_frames = int(pixels.shape[0] * pixels.shape[1])
    row: dict[str, object] = {
        "prompt_index": prompt_index,
        "sample_seed": sample_seed,
        "decode_mode": "full",
        "decoded_frames": decoded_frames,
        "total_sec": total_sec,
        "finite_video_fps": decoded_frames / total_sec,
        "throughput_fps": decoded_frames / total_sec,
        "fps": decoded_frames / total_sec,
        "fps_primary": "finite_video_fps",
        "total_transformer_forward_count": int(
            info["total_transformer_forward_count"]
        ),
    }
    del noise, latents
    return pixels, row


def main() -> None:
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="WRAPPER_GENERATION")
    parser.add_argument("--source-uri", type=Path, default=ROOT / "assets/source/Self-Forcing-main")
    parser.add_argument("--weights-uri", type=Path, default=ROOT / "assets/weights/self_forcing_weights.tar")
    parser.add_argument("--vae-uri", type=Path, default=ROOT / "assets/weights/Wan2.1_VAE.pth")
    parser.add_argument("--prompt-uri", type=Path, default=ROOT / "assets/data/vbench_all_dimension_extended.txt")
    parser.add_argument("--checkpoint-uri", default=None)
    parser.add_argument(
        "--prompt-indices",
        default="all",
        help=(
            "Comma-separated prompt indices, all, or shard_first to run only the "
            "first prompt assigned to this shard."
        ),
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-mode", choices=("modulo", "contiguous"), default="contiguous")
    parser.add_argument("--allow-subset-sharding", action="store_true")
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--video-fps", type=int, default=16)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/unstep_wrapper_generation"))
    parser.add_argument(
        "--workdir-name",
        default=None,
        help=(
            "Directory name under --workdir used for staging source. Defaults to "
            "run-name. Set to a stable value for repeated runs so compiled source "
            "paths do not depend on result names."
        ),
    )
    parser.add_argument(
        "--reuse-workdir",
        action="store_true",
        help="Reuse an existing prepared workdir/src instead of restaging source.",
    )
    parser.add_argument(
        "--fa3-import-backend",
        default="auto",
        choices=("auto", "h100", "flash_attn_3", "fa2"),
        help=("auto selects FA3 on Hopper and FA2 on other GPUs, including GB200; "
              "fa2 forces FA2, while h100 and flash_attn_3 select Hopper FA3 import paths."),
    )
    parser.add_argument("--seed-mode", choices=("sequential_global", "per_prompt"), default="sequential_global")
    parser.add_argument(
        "--rng-skip-mode",
        choices=("none", "initial_only", "producer_transition_noise"),
        default="initial_only",
        help=(
            "How sequential_global shards advance RNG for prompts before the shard. "
            "initial_only burns the initial latent noise only; "
            "producer_transition_noise additionally burns transition noise for "
            "the working driver schedule, including appended clean output."
        ),
    )
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument(
        "--speed-only",
        action="store_true",
        help="Record speed only. If videos are saved for debugging, delete them after JSON is written.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Run unrecorded speed warmup passes in the same process before measurement.",
    )
    parser.add_argument(
        "--warmup-prompt-indices",
        default="",
        help=(
            "Prompt indices for in-process warmup. Defaults to --prompt-indices. "
            "Use shard_first to warm only the first prompt assigned to this shard."
        ),
    )
    parser.add_argument(
        "--disable-block-schedule-overrides",
        action="store_true",
        help="Ablation: remove block-specific later schedules so every non-first block uses later_schedule.",
    )
    parser.add_argument(
        "--disable-gc-during-measurement",
        action="store_true",
        help="Disable Python garbage collection around measured prompt generation.",
    )
    parser.add_argument(
        "--exclude-final-output-layout-timing",
        action="store_true",
        help=(
            "Stop the speed timer after VAE decode and before final RGB clamp/layout "
            "permutation. The conversion still runs before saving videos."
        ),
    )
    parser.add_argument(
        "--clean-sigma-override",
        type=float,
        default=WrapperConfig().clean_sigma_override,
        help=(
            "Use this sigma for both input re-noising and flow-to-x0 conversion "
            "at the appended clean step (default: 0.02). Other denoising timesteps "
            "keep their scheduler values; the DiT remains conditioned on t=0."
        ),
    )
    args = parser.parse_args()
    if args.warmup_runs < 0:
        raise RuntimeError("--warmup-runs must be non-negative")
    if args.clean_sigma_override is not None and (
        not math.isfinite(args.clean_sigma_override) or args.clean_sigma_override < 0
    ):
        raise RuntimeError("--clean-sigma-override must be finite and non-negative")

    backend = runtime.resolve_attention_backend(args.fa3_import_backend)
    cfg = make_wrapper_config(args, backend)
    manager = runtime.path_manager()
    default_run_dir = Path("/tmp/unstep_wrapper_generation_outputs") / args.run_name
    video_dir = args.video_dir or (default_run_dir / "indexed_videos")
    out_json = args.out_json or (default_run_dir / f"{args.run_name}_generation_scores.json")

    input_paths = tuple(
        Path(manager.get_local_path(str(uri)))
        for uri in (args.source_uri, args.weights_uri, args.vae_uri, args.prompt_uri, args.checkpoint_uri)
        if uri is not None
    )
    workdir = args.workdir / (args.workdir_name or args.run_name)
    assert_safe_cleanup_dir(workdir, video_dir=video_dir, out_json=out_json, input_paths=input_paths)
    if args.speed_only and args.save_videos:
        assert_safe_speed_video_cleanup_dir(
            video_dir, run_name=args.run_name, out_json=out_json, input_paths=input_paths)
    owns_workdir = False
    owns_video_dir = False
    if workdir.is_symlink():
        raise RuntimeError(f"refusing symlinked workdir: {workdir}")
    if workdir.exists():
        if not args.reuse_workdir:
            raise RuntimeError(f"workdir already exists; choose a new path or use --reuse-workdir: {workdir}")
    else:
        if args.reuse_workdir:
            raise RuntimeError(f"--reuse-workdir requested but workdir does not exist: {workdir}")
        workdir.mkdir(parents=True)
        owns_workdir = True

    try:
        all_prompts = load_prompt_lines(manager, str(args.prompt_uri))
        raw_prompt_indices = args.prompt_indices.strip()
        if raw_prompt_indices == "shard_first":
            prompt_indices = shard_indices(
                list(range(len(all_prompts))),
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                shard_mode=args.shard_mode,
            )[:1]
        else:
            prompt_indices = parse_indices(raw_prompt_indices, len(all_prompts))
        if raw_prompt_indices != "shard_first" and should_shard_prompt_indices(
            raw_prompt_indices,
            args.num_shards,
            args.allow_subset_sharding,
        ):
            prompt_indices = shard_indices(
                prompt_indices,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                shard_mode=args.shard_mode,
            )
        if not prompt_indices:
            raise RuntimeError("this shard has no prompt indices")
        if args.seed_mode == "sequential_global" and prompt_indices != sorted(prompt_indices):
            raise RuntimeError("--seed-mode=sequential_global requires sorted prompt indices")

        prepare_args = make_prepare_args(args, cfg, backend)
        if args.reuse_workdir:
            source_root, staging_audit = runtime.reuse_prepared_source(prepare_args, manager, workdir)
            owns_workdir = True
        else:
            source_root = runtime.prepare_source(prepare_args, manager, workdir)
            staging_audit = runtime.read_staging_audit(workdir)
        if args.speed_only and args.save_videos:
            if video_dir.exists() or video_dir.is_symlink():
                raise RuntimeError(f"speed-only video directory already exists; choose a new path: {video_dir}")
            video_dir.mkdir(parents=True)
            owns_video_dir = True
        if args.seed_mode == "sequential_global":
            torch.manual_seed(cfg.base.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.base.seed)

        device = torch.device("cuda")
        single_generator = runtime.load_self_forcing_transformer(
            source_root=source_root,
            precision=cfg.base.precision,
            device=device,
            move_to_device=False,
            timestep_shift=UNSTEP_TIMESTEP_SHIFT,
        )
        pipeline, target_transformer, text_encoder, _vae, _zero_cache, device = runtime.load_pipeline(
            source_root,
            cfg.base.precision,
            load_full_vae=True,
            move_generator_to_device=False,
            generator=single_generator,
        )
        wrapper = OfflineSelfForcingWrapper(cfg)
        method_transformer, method_info = wrapper.prepare_pipeline_transformer(
            transformer=target_transformer,
            precision=cfg.base.precision,
            device=device,
            timestep_shift=UNSTEP_TIMESTEP_SHIFT,
        )
        method_info.update(
            runtime.bind_pipeline_scheduler_to_transformer(pipeline, target_transformer)
        )
        method_info["single_transformer_loader"] = "load_self_forcing_transformer"
        wrapper.install_runtime(
            pipeline=pipeline,
            target_transformer=target_transformer,
            method_transformer=method_transformer,
            device=device,
        )
        clean_sigma_validation = runtime.validate_clean_sigma_override(
            method_transformer, clean_sigma_override=cfg.clean_sigma_override, device=device
        )
        clean_sigma_validation["input_renoising"] = wrapper.clean_sigma_input_validation
        print(json.dumps({"clean_sigma_validation": clean_sigma_validation}), flush=True)
        dtype = runtime.torch_dtype(cfg.base.precision)

        if args.warmup_runs:
            warmup_raw = args.warmup_prompt_indices.strip() or args.prompt_indices
            if warmup_raw == "shard_first":
                warmup_indices = prompt_indices[:1]
            else:
                warmup_indices = parse_indices(warmup_raw, len(all_prompts))
                if should_shard_prompt_indices(
                    warmup_raw,
                    args.num_shards,
                    args.allow_subset_sharding,
                ):
                    warmup_indices = shard_indices(
                        warmup_indices,
                        shard_index=args.shard_index,
                        num_shards=args.num_shards,
                        shard_mode=args.shard_mode,
                    )
            if not warmup_indices:
                raise RuntimeError("--warmup-prompt-indices resolved to an empty list")
            for _ in range(args.warmup_runs):
                if args.seed_mode == "sequential_global":
                    torch.manual_seed(cfg.base.seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(cfg.base.seed)
                warmup_last_sequential_index = -1
                for prompt_index in warmup_indices:
                    if args.seed_mode == "sequential_global":
                        for skipped_index in range(
                            warmup_last_sequential_index + 1,
                            prompt_index,
                        ):
                            consume_skipped_prompt_rng(
                                cfg=cfg,
                                prompt_index=skipped_index,
                                device=device,
                                dtype=dtype,
                                mode=args.rng_skip_mode,
                            )
                        warmup_last_sequential_index = prompt_index
                    noise, sample_seed, use_global_rng = draw_prompt_noise(
                        cfg=cfg,
                        prompt_index=prompt_index,
                        device=device,
                        dtype=dtype,
                        seed_mode=args.seed_mode,
                    )
                    pixels, _row = run_prompt(
                        wrapper=wrapper,
                        pipeline=pipeline,
                        method_transformer=method_transformer,
                        text_encoder=text_encoder,
                        prompt=all_prompts[prompt_index],
                        prompt_index=prompt_index,
                        cfg=cfg,
                        device=device,
                        noise=noise,
                        sample_seed=sample_seed,
                        use_global_rng=use_global_rng,
                        exclude_final_output_layout_timing=args.exclude_final_output_layout_timing,
                    )
                    del pixels
            if args.seed_mode == "sequential_global":
                torch.manual_seed(cfg.base.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(cfg.base.seed)

        rows: list[dict[str, object]] = []
        manifest: list[dict[str, object]] = []
        last_sequential_index = -1
        gc_was_enabled = gc.isenabled()
        if args.disable_gc_during_measurement:
            gc.disable()
        try:
            for prompt_index in prompt_indices:
                prompt = all_prompts[prompt_index]
                skipped_prompt_rng_records = []
                if args.seed_mode == "sequential_global":
                    for skipped_index in range(last_sequential_index + 1, prompt_index):
                        skipped_prompt_rng_records.append(
                            consume_skipped_prompt_rng(
                                cfg=cfg,
                                prompt_index=skipped_index,
                                device=device,
                                dtype=dtype,
                                mode=args.rng_skip_mode,
                            )
                        )
                    last_sequential_index = prompt_index

                noise, sample_seed, use_global_rng = draw_prompt_noise(
                    cfg=cfg,
                    prompt_index=prompt_index,
                    device=device,
                    dtype=dtype,
                    seed_mode=args.seed_mode,
                )
                pixels, row = run_prompt(
                    wrapper=wrapper,
                    pipeline=pipeline,
                    method_transformer=method_transformer,
                    text_encoder=text_encoder,
                    prompt=prompt,
                    prompt_index=prompt_index,
                    cfg=cfg,
                    device=device,
                    noise=noise,
                    sample_seed=sample_seed,
                    use_global_rng=use_global_rng,
                    exclude_final_output_layout_timing=args.exclude_final_output_layout_timing,
                )
                row["seed_mode"] = args.seed_mode
                if args.seed_mode == "sequential_global":
                    row["noise_draw_ordinal"] = prompt_index
                    row["skipped_noise_draw_ordinals"] = [
                        item["prompt_index"] for item in skipped_prompt_rng_records
                    ]
                    row["skipped_prompt_rng_records"] = skipped_prompt_rng_records
                    row["noise_shape"] = [1, cfg.base.num_latent_frames, 16, 60, 104]
                    row["noise_dtype"] = str(dtype)
                    row["rng_device"] = str(device)
                if args.save_videos:
                    video_path = video_dir / f"{prompt_index}-0_ema.mp4"
                    save_video(video_path, pixels, fps=args.video_fps)
                    manifest.append(
                        {
                            "prompt_index": prompt_index,
                            "filename": video_path.name,
                            "path": str(video_path),
                        }
                    )
                del pixels
                rows.append(row)
        finally:
            if args.disable_gc_during_measurement and gc_was_enabled:
                gc.enable()

        mean_fps = sum(float(row["finite_video_fps"]) for row in rows) / len(rows)
        result = {
            "run_name": args.run_name,
            "argv": sys.argv,
            "speed_protocol_version": "standard_speed_v2",
            "fps_primary": "finite_video_fps",
            "timed_region": "post_text_encode_generation_and_vae",
            "exclude_final_output_layout_timing": args.exclude_final_output_layout_timing,
            "decode_mode": "full",
            "num_frame_per_block": cfg.base.num_frame_per_block,
            "num_latent_frames": cfg.base.num_latent_frames,
            "decoded_frames": 81,
            "seed": cfg.base.seed,
            "seed_mode": args.seed_mode,
            "rng_skip_mode": args.rng_skip_mode,
            "speed_only": args.speed_only,
            "source_uri": str(args.source_uri),
            "weights_uri": str(args.weights_uri),
            "vae_uri": str(args.vae_uri),
            "checkpoint_uri": args.checkpoint_uri,
            "prompt_uri": str(args.prompt_uri),
            "prompt_indices": prompt_indices,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "shard_mode": args.shard_mode,
            "allow_subset_sharding": args.allow_subset_sharding,
            "reuse_workdir": args.reuse_workdir,
            "workdir_name": args.workdir_name or args.run_name,
            "runtime_state_env": {
                "TMPDIR": os.environ.get("TMPDIR"),
                "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME"),
                "TORCH_HOME": os.environ.get("TORCH_HOME"),
                "TORCHINDUCTOR_CACHE_DIR": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
                "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR"),
                "TRITON_HOME": os.environ.get("TRITON_HOME"),
                "CUDA_CACHE_PATH": os.environ.get("CUDA_CACHE_PATH"),
                "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE": os.environ.get(
                    "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE"
                ),
                "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE": os.environ.get(
                    "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE"
                ),
                "TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE": os.environ.get(
                    "TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE"
                ),
                "TRITON_REMOTE_CACHE_ENABLE": os.environ.get("TRITON_REMOTE_CACHE_ENABLE"),
                "TORCH_COMPILE_CACHE_KEY_TAG": os.environ.get("TORCH_COMPILE_CACHE_KEY_TAG"),
            },
            "save_videos": args.save_videos,
            "warmup_runs": args.warmup_runs,
            "warmup_prompt_indices": args.warmup_prompt_indices,
            "disable_gc_during_measurement": args.disable_gc_during_measurement,
            "clean_sigma_override": args.clean_sigma_override,
            "clean_sigma_validation": clean_sigma_validation,
            "video_dir": str(video_dir) if args.save_videos else None,
            "video_fps": args.video_fps,
            "fa3_import_backend": args.fa3_import_backend,
            "effective_attention_backend": backend,
            "staging_audit": staging_audit,
            "runtime_info": runtime.runtime_info(source_root, cfg.base.precision, attention_backend=backend),
            "torch_backend_info": {
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_benchmark_limit": getattr(
                    torch.backends.cudnn, "benchmark_limit", None
                ),
            },
            "config": {
                "base": cfg.base.__dict__,
                "compile_mode": cfg.compile_mode,
                "method_lock_flags": {
                    "defer_block_timing_sync": True,
                    "causal_rope_triton_fp32": True,
                    "cache_scheduler_sigmas": True,
                    "prewarm_attention_seqlens": True,
                    "full_vae_decode_dtype": cfg.full_vae_decode_dtype,
                },
                "first_schedule": cfg.first_schedule,
                "later_schedule": cfg.later_schedule,
                "block_schedule_overrides": cfg.block_schedule_overrides,
                "append_clean_final_step": cfg.append_clean_final_step,
                "local_attn_size": cfg.local_attn_size,
                "sink_size": cfg.sink_size,
                "local_layer_sizes": cfg.local_layer_sizes,
                "svd_targets": cfg.svd_targets,
                "svd_keep_ratio": cfg.svd_keep_ratio,
                "svd_layer_keep_ratios": cfg.svd_layer_keep_ratios,
                "rope_layers": cfg.rope_layers,
                "rope_query_only": cfg.rope_query_only,
                "vae_block_size": cfg.vae_block_size,
                "strict_attention_prewarm": cfg.strict_attention_prewarm,
                "use_fa3_batch1": cfg.use_fa3_batch1,
                "use_fa3_slice_lens": cfg.use_fa3_slice_lens,
                "full_decode_channel_first_latents": cfg.full_decode_channel_first_latents,
                "cache_timestep_tensors": cfg.cache_timestep_tensors,
                "full_vae_decode_dtype": cfg.full_vae_decode_dtype,
                "full_vae_output_dtype": cfg.full_vae_output_dtype,
                "full_vae_use_cache": cfg.full_vae_use_cache,
                "full_vae_fast_cache_clear": cfg.full_vae_fast_cache_clear,
                "full_vae_skip_post_clear": cfg.full_vae_skip_post_clear,
                "cache_vae_scale_tensors": cfg.cache_vae_scale_tensors,
                "full_vae_batch1_fast_path": cfg.full_vae_batch1_fast_path,
                "full_vae_list_cat_output": cfg.full_vae_list_cat_output,
                "full_vae_block_cached_decode": cfg.full_vae_block_cached_decode,
                "full_vae_prealloc_block_output": cfg.full_vae_prealloc_block_output,
                "full_vae_channels_last_3d": cfg.full_vae_channels_last_3d,
                "save_video_gpu_uint8": cfg.save_video_gpu_uint8,
                "defer_video_copy_timing": cfg.defer_video_copy_timing,
                "preserve_cuda_cache_between_runs": cfg.preserve_cuda_cache_between_runs,
                "clean_sigma_override": cfg.clean_sigma_override,
                "single_transformer": method_transformer is target_transformer,
                "separate_method_transformer": False,
                "timestep_shift": UNSTEP_TIMESTEP_SHIFT,
                "scheduler_shift_summary": runtime.scheduler_shift_summary(
                    pipeline=pipeline,
                    method_transformer=method_transformer,
                ),
            },
            "method_info": method_info,
            "mean_finite_video_fps": mean_fps,
            "measurements": rows,
            "saved_video_manifest": manifest,
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        if owns_video_dir:
            assert_safe_speed_video_cleanup_dir(
                video_dir, run_name=args.run_name, out_json=out_json, input_paths=input_paths)
            _ORIGINAL_RMTREE(video_dir, ignore_errors=True)
        if owns_workdir and not args.keep_workdir:
            assert_safe_cleanup_dir(workdir, video_dir=video_dir, out_json=out_json, input_paths=input_paths)
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
