# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Image conditioning and long-video decoding; imported only for those tasks."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from unstep import wrapper_runtime as runtime
from unstep.wrapper import OfflineSelfForcingWrapper


def load_image_prompts(path: Path, image_root: Path):
    entries = json.loads(path.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("I2V prompt file must be a nonempty JSON list")
    prompts, images = [], []
    for entry in entries:
        prompt, name = entry.get("prompt_en"), entry.get("image_name")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Each I2V entry needs a nonempty prompt_en")
        if any(c in prompt for c in ("/", "\\", "\0")) or len((prompt + "-0.mp4").encode()) > 255:
            raise ValueError("I2V prompt cannot be used as a VBench video filename")
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ValueError("Each I2V entry needs an image_name without directories")
        image = (image_root / name).resolve()
        if not image.is_relative_to(image_root.resolve()) or not image.is_file():
            raise ValueError(f"Missing or invalid reference image: {name}")
        prompts.append(prompt)
        images.append(image)
    if len(set(prompts)) != len(prompts):
        raise ValueError("I2V prompts must be unique for VBench output naming")
    return prompts, images


class ImageToVideoWrapper(OfflineSelfForcingWrapper):
    """One reference latent followed by five blocks of four generated latents."""

    @torch.inference_mode()
    def encode_references(self, pipeline, prompts, images, indices, device):
        preprocess = transforms.Compose([
            transforms.Resize((480, 832)), transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        encoded = {}
        self.references = {}
        for index in indices:
            image_path = images[index]
            if image_path not in encoded:
                with Image.open(image_path) as image:
                    pixels = preprocess(image.convert("RGB")).unsqueeze(0).unsqueeze(2)
                latent = pipeline.vae.encode_to_latent(pixels.to(device=device, dtype=torch.bfloat16))
                if tuple(latent.shape) != (1, 1, 16, 60, 104):
                    raise ValueError(f"Unexpected reference latent shape: {tuple(latent.shape)}")
                encoded[image_path] = latent
                pipeline.vae.model.clear_cache()
            self.references[prompts[index]] = encoded[image_path]

    def install_runtime(self, **kwargs):
        super().install_runtime(**kwargs)
        runtime.prewarm_flash_attention_seqlen_cache(
            device=kwargs["device"], batch_size=1,
            sequence_lengths=[i * runtime.WAN_TOKENS_PER_LATENT_FRAME for i in range(1, 22)],
        )

    def generate_latents(self, *, pipeline, method_transformer, prompt, noise,
                         conditional_dict, kv_cache, crossattn_cache, seed=0,
                         use_global_rng=False):
        if tuple(noise.shape) != (1, 20, 16, 60, 104):
            raise ValueError("I2V requires 20 generated latent frames")
        reference = self.references[prompt].to(device=noise.device, dtype=noise.dtype)
        latents = torch.empty((1, 16, 21, 60, 104), device=noise.device, dtype=noise.dtype)
        latents[:, :, :1].copy_(reference.permute(0, 2, 1, 3, 4))
        timestep_cache = {} if self.config.cache_timestep_tensors else None
        generator = None if use_global_rng else torch.Generator(device=noise.device).manual_seed(seed + 1_000_003)

        # Condition the KV cache on the unchanged reference, discarding its prediction.
        runtime.set_step_layer_skip_route_active(method_transformer, False)
        runtime.set_cross_step_kv_context(method_transformer, current_start_frame=0,
                                         step_index=0, timestep=0, active=False,
                                         block_idx=None, num_blocks=5)
        method_transformer(
            noisy_image_or_video=reference, conditional_dict=conditional_dict,
            timestep=runtime.timestep_tensor(timestep_cache, (1, 1), 0, noise.device),
            kv_cache=kv_cache, crossattn_cache=crossattn_cache, current_start=0,
        )
        total_forwards = 1
        for block_idx in range(5):
            noise_start = 4 * block_idx
            current_start = 1 + noise_start
            denoised, forwards = self.denoise_chunk(
                transformer=method_transformer, pipeline=pipeline,
                noisy_input=noise[:, noise_start:noise_start + 4],
                conditional_dict=conditional_dict, kv_cache=kv_cache,
                crossattn_cache=crossattn_cache, current_start_frame=current_start,
                block_idx=block_idx, generator=generator,
                timestep_cache=timestep_cache, num_blocks=5,
            )
            latents[:, :, current_start:current_start + 4].copy_(denoised.permute(0, 2, 1, 3, 4))
            total_forwards += forwards
        return latents, {"total_transformer_forward_count": total_forwards,
                         "latent_layout": "bcthw"}


class LongVideoWrapper(OfflineSelfForcingWrapper):
    """Compile the fixed-size VAE decoder, not the duration-dependent outer loop."""

    def install_runtime(self, **kwargs):
        use_blocks = self.config.base.num_latent_frames > 81
        super().install_runtime(**kwargs, compile_vae_loop=not use_blocks)
        if use_blocks:
            kwargs["pipeline"].vae.model.decoder.compile(mode=self.config.compile_mode)
