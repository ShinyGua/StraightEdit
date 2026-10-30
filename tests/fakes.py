"""Deterministic stand-ins for FLUX.1-dev behind the interfaces the samplers call, in float32 on CPU."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler, FluxPipeline
from diffusers.image_processor import VaeImageProcessor


def velocity_fn(x, t, cond):
    t = t.reshape(-1, *([1] * (x.ndim - 1))).to(x.dtype)
    return torch.tanh(x + cond) * (0.5 + t) - 0.2 * x + 0.1 * cond


def prompt_embedding(prompt, *shape):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(sum(map(ord, prompt))))


class FakeTransformer:
    """diffusers FluxTransformer2DModel."""

    def __init__(self):
        self.config = SimpleNamespace(in_channels=64, guidance_embeds=True)
        self.calls = []

    def cache_context(self, name):
        return torch.no_grad()

    def __call__(
        self, hidden_states, timestep, guidance, pooled_projections, encoder_hidden_states, **kwargs
    ):
        cond = encoder_hidden_states.mean() + 0.05 * guidance.mean() + pooled_projections.mean()
        self.calls.append(float(cond))
        return (velocity_fn(hidden_states, timestep, cond),)


class FakeVAE:
    dtype = torch.float32
    config = SimpleNamespace(shift_factor=0.1159, scaling_factor=0.3611)

    def encode(self, x):
        latent = (
            F.avg_pool2d(x, 8).repeat(1, 6, 1, 1)[:, :16] * torch.linspace(0.5, 1.5, 16)[None, :, None, None]
        )
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))

    def decode(self, z, **kwargs):
        return (torch.tanh(F.interpolate(z[:, :3], scale_factor=8)),)


class FakeFluxPipe:
    """The parts of diffusers' FluxPipeline that flow/flux.py and straightedit/flux.py use."""

    vae_scale_factor = 8
    _pack_latents = staticmethod(FluxPipeline._pack_latents)
    _unpack_latents = staticmethod(FluxPipeline._unpack_latents)
    _prepare_latent_image_ids = staticmethod(FluxPipeline._prepare_latent_image_ids)
    prepare_latents = FluxPipeline.prepare_latents

    def __init__(self):
        self.transformer = FakeTransformer()
        self.vae = FakeVAE()
        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=3.0,
            use_dynamic_shifting=True,
            base_image_seq_len=256,
            max_image_seq_len=4096,
            base_shift=0.5,
            max_shift=1.15,
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=16)
        self._execution_device = torch.device("cpu")
        self._current_timestep = None

    def encode_prompt(self, prompt, **kwargs):
        return prompt_embedding(prompt, 1, 8, 32), prompt_embedding(prompt + "/", 1, 16), torch.zeros(8, 3)

    def maybe_free_model_hooks(self):
        pass


class FakeFlux:
    """BFL Flux.forward."""

    def __init__(self):
        self.calls = []

    def __call__(self, img, img_ids, txt, txt_ids, y, timesteps, guidance):
        cond = txt.mean() + 0.05 * guidance.mean() + y.mean()
        self.calls.append(float(cond))
        return velocity_fn(img, timesteps, cond)

    def with_info(self, info, **kwargs):
        """The original ReFlow samplers pass and expect back an `info` dict."""
        return self(**kwargs), info


class FakeEmbedder:
    def __init__(self, *shape):
        self.shape = shape

    def __call__(self, prompts):
        return prompt_embedding(prompts[0], 1, *self.shape)


class FakeAutoEncoder:
    def encode(self, x):
        return (
            F.avg_pool2d(x, 8).repeat(1, 6, 1, 1)[:, :16] * torch.linspace(0.5, 1.5, 16)[None, :, None, None]
        )

    def decode(self, z):
        return torch.tanh(F.interpolate(z[:, :3], scale_factor=8))
