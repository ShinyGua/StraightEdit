"""ReFlow backbone: FLUX.1-dev through the BFL reference implementation, as set up by RF-Solver.

Model loading, the schedule and the packing follow black-forest-labs/flux (Apache-2.0) as
shipped by RF-Solver-Edit and FireFlow.
"""

import math
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from einops import rearrange, repeat
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from torch import Tensor

from straightedit.bfl.autoencoder import AutoEncoder, AutoEncoderParams
from straightedit.bfl.conditioner import HFEmbedder
from straightedit.bfl.model import Flux, FluxParams
from straightedit.sampler import Step

MODEL_ID = "black-forest-labs/FLUX.1-dev"
IMAGE_SIZE = 1024  # 5_Exp/5_experiments.tex:31

FLUX_DEV = FluxParams(
    in_channels=64,
    vec_in_dim=768,
    context_in_dim=4096,
    hidden_size=3072,
    mlp_ratio=4.0,
    num_heads=24,
    depth=19,
    depth_single_blocks=38,
    axes_dim=[16, 56, 56],
    theta=10_000,
    qkv_bias=True,
    guidance_embed=True,
)
FLUX_DEV_AE = AutoEncoderParams(
    resolution=256,
    in_channels=3,
    ch=128,
    out_ch=3,
    ch_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    z_channels=16,
    scale_factor=0.3611,
    shift_factor=0.1159,
)


def time_shift(mu: float, sigma: float, t: Tensor) -> Tensor:
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(num_steps: int, image_seq_len: int) -> list[float]:
    """num_steps + 1 points from 1 to 0, shifted towards high t for long sequences."""
    timesteps = torch.linspace(1, 0, num_steps + 1)
    mu = get_lin_function()(image_seq_len)
    return time_shift(mu, 1.0, timesteps).tolist()


class ReflowBackbone:
    def __init__(self, model: Flux, ae: AutoEncoder, t5: HFEmbedder, clip: HFEmbedder, device: torch.device):
        self.model, self.ae, self.t5, self.clip = model, ae, t5, clip
        self.device = device
        side = IMAGE_SIZE // 16
        img_ids = torch.zeros(side, side, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(side)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(side)[None, :]
        self.img_ids = repeat(img_ids, "h w c -> 1 (h w) c").to(device)

    @classmethod
    def load(cls, device: torch.device) -> "ReflowBackbone":
        token = os.getenv("HUGGINGFACE_TOKEN")
        t5 = HFEmbedder("google/t5-v1_1-xxl", max_length=512, is_clip=False, torch_dtype=torch.bfloat16).to(
            device
        )
        clip = HFEmbedder(
            "openai/clip-vit-large-patch14", max_length=77, is_clip=True, torch_dtype=torch.bfloat16
        ).to(device)
        with torch.device("meta"):
            model = Flux(FLUX_DEV).to(torch.bfloat16)
            ae = AutoEncoder(FLUX_DEV_AE)
        for module, filename in [(model, "flux1-dev.safetensors"), (ae, "ae.safetensors")]:
            weights = load_file(hf_hub_download(MODEL_ID, filename, token=token), device=str(device))
            module.load_state_dict(weights, assign=True)
        return cls(model, ae, t5, clip, device)

    def encode_prompt(self, prompt: str, guidance: float) -> dict:
        """Text condition c and the distilled guidance scale that goes with it."""
        txt = self.t5([prompt]).to(self.device)
        return {
            "txt": txt,
            "txt_ids": torch.zeros(1, txt.shape[1], 3, device=self.device),
            "y": self.clip([prompt]).to(self.device),
            "guidance": guidance,
        }

    def encode_image(self, path: Path) -> Tensor:
        """z_src: the packed autoencoder latent of the source image."""
        image = Image.open(path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS).convert("RGB")
        x = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1
        z = self.ae.encode(x.unsqueeze(0).to(self.device)).to(torch.bfloat16)
        return rearrange(z, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    def schedule(self, num_steps: int, kept_steps: int) -> list[Step]:
        """The last `kept_steps` steps of the shifted (num_steps + 1)-point grid."""
        t = get_schedule(num_steps, (IMAGE_SIZE // 16) ** 2)
        return [Step(t[i], t[i + 1], t[i]) for i in range(num_steps - kept_steps, num_steps)]

    def velocity(self, x: Tensor, t: float, cond: dict) -> Tensor:
        return self.model(
            img=x,
            img_ids=self.img_ids,
            txt=cond["txt"],
            txt_ids=cond["txt_ids"],
            y=cond["y"],
            timesteps=torch.full((x.shape[0],), t, dtype=x.dtype, device=x.device),
            guidance=torch.full((x.shape[0],), cond["guidance"], dtype=x.dtype, device=x.device),
        )

    def decode(self, z: Tensor) -> np.ndarray:
        """uint8 RGB image of the latent z."""
        side = IMAGE_SIZE // 16
        x = rearrange(z.float(), "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=side, w=side, ph=2, pw=2)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)
        x = rearrange(x.clamp(-1, 1).float()[0], "c h w -> h w c")
        return (127.5 * (x + 1.0)).cpu().byte().numpy()
