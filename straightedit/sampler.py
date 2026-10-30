"""StraightEdit sampling, Algorithm 1."""

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import torch
from torch import Tensor

from straightedit.corrections import noise_correction, rotational_correction


class Step(NamedTuple):
    """One step t_i → t_{i-1} of the decreasing time grid."""

    t: float | Tensor
    t_next: float | Tensor
    model_t: float | Tensor  # t_i in the form the backbone's transformer takes


Velocity = Callable[[Tensor, Any, Any], Tensor]


@torch.no_grad()
def straightedit(
    velocity: Velocity,
    z_src: Tensor,
    cond_src: Any,
    cond_tgt: Any,
    steps: Sequence[Step],
    anchor_interval: int,
    beta: float,
) -> Tensor:
    """Edit z_src from the source condition towards the target condition.

    velocity(x, model_t, cond) evaluates V_θ(x, t; c). The anchor set 𝒯 is every
    `anchor_interval`-th step (J) and the last step; the source branch is evaluated only there.
    After the first step, the target velocity is the mean of the current and previous target
    predictions; Algorithm 1 omits this. Returns z_edit.
    """
    z = z_src.clone()
    v_src_anchor = eps_anchor = v_tgt_prev = None
    for i, step in enumerate(steps):
        eps = torch.randn_like(z_src)
        q = (1 - step.t) * z_src + step.t * eps  # Eq. (5)
        p = z + q - z_src  # Eq. (6)
        v_tgt = velocity(p, step.model_t, cond_tgt)

        if i % anchor_interval == 0 or i == len(steps) - 1:
            v_src = velocity(q, step.model_t, cond_src)
            v_src_anchor, eps_anchor = v_src, eps
        else:
            v_src = noise_correction(v_src_anchor, eps, eps_anchor)  # Eq. (13)

        if v_tgt_prev is None:
            v_tgt_prev = v_tgt
        else:
            v_tgt, v_tgt_prev = 0.5 * (v_tgt + v_tgt_prev), v_tgt

        v_delta = (v_tgt - v_src).float()
        v_ref = eps - z_src  # Eq. (14)
        v_delta = rotational_correction(v_delta, v_src, v_ref, beta)  # Eq. (16)
        z = (z + (step.t_next - step.t) * v_delta).to(z_src.dtype)  # Eq. (17)
    return z
