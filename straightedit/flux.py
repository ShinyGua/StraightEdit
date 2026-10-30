# Adapted from diffusers' FluxPipeline.
# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FLUX backbone: FLUX.1-dev through diffusers' FluxPipeline."""

import os
from pathlib import Path

import numpy as np
import torch
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from PIL import Image
from torch import Tensor

from straightedit.sampler import Step

MODEL_ID = "black-forest-labs/FLUX.1-dev"
IMAGE_SIZE = 1024  # 5_Exp/5_experiments.tex:31


class FluxBackbone:
    def __init__(self, pipe: FluxPipeline):
        self.pipe = pipe
        self.img_ids = None

    @classmethod
    def load(cls, device: torch.device) -> "FluxBackbone":
        pipe = FluxPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, token=os.getenv("HUGGINGFACE_TOKEN")
        )
        return cls(pipe.to(device))

    def encode_prompt(self, prompt: str, guidance: float) -> dict:
        """Text condition c and the distilled guidance scale that goes with it."""
        device = self.pipe._execution_device
        embeds, pooled, txt_ids = self.pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, device=device, max_sequence_length=512
        )
        return {
            "encoder_hidden_states": embeds,
            "pooled_projections": pooled,
            "txt_ids": txt_ids,
            "guidance": torch.full([1], guidance, device=device, dtype=torch.float32),
        }

    def encode_image(self, path: Path) -> Tensor:
        """z_src: the packed VAE latent of the source image."""
        pipe = self.pipe
        device = pipe._execution_device
        image = pipe.image_processor.preprocess(Image.open(path), height=IMAGE_SIZE, width=IMAGE_SIZE)
        latents = pipe.vae.encode(image.to(pipe.vae.dtype).to(device)).latent_dist.mode()
        latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        channels = pipe.transformer.config.in_channels // 4
        z_src = pipe._pack_latents(latents, 1, channels, latents.shape[2], latents.shape[3])
        # prepare_latents also draws a noise latent, which is discarded. The draw advances the random
        # generator, so it keeps every ε_{t_i} identical to the experiment code.
        _, self.img_ids = pipe.prepare_latents(
            1, channels, IMAGE_SIZE, IMAGE_SIZE, latents.dtype, device, None
        )
        return z_src

    def schedule(self, num_steps: int, kept_steps: int) -> list[Step]:
        """The last `kept_steps` steps of FLUX's shifted num_steps-point grid."""
        pipe = self.pipe
        config = pipe.scheduler.config
        mu = calculate_shift(
            (IMAGE_SIZE // 16) ** 2,
            config.base_image_seq_len,
            config.max_image_seq_len,
            config.base_shift,
            config.max_shift,
        )
        sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        timesteps, _ = retrieve_timesteps(
            pipe.scheduler, num_steps, pipe._execution_device, sigmas=sigmas, mu=mu
        )
        sigmas = pipe.scheduler.sigmas.float()
        return [
            Step(sigmas[i], sigmas[i + 1], timesteps[i]) for i in range(num_steps - kept_steps, num_steps)
        ]

    def velocity(self, x: Tensor, t: Tensor, cond: dict) -> Tensor:
        return self.pipe.transformer(
            hidden_states=x,
            timestep=t.expand(x.shape[0]).to(x.dtype) / 1000,
            guidance=cond["guidance"],
            pooled_projections=cond["pooled_projections"],
            encoder_hidden_states=cond["encoder_hidden_states"],
            txt_ids=cond["txt_ids"],
            img_ids=self.img_ids,
            return_dict=False,
        )[0]

    def decode(self, z: Tensor) -> np.ndarray:
        """uint8 RGB image of the latent z."""
        pipe = self.pipe
        latents = pipe._unpack_latents(z, IMAGE_SIZE, IMAGE_SIZE, pipe.vae_scale_factor)
        latents = latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="np")[0]
        return (image * 255).astype(np.uint8)
