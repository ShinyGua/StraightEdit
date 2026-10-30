"""Noise Correction (Sec. 4.1) and Rotational Correction (Sec. 4.2, supplementary Appendix A)."""

import torch
from torch import Tensor


def noise_correction(v_src_anchor: Tensor, eps: Tensor, eps_anchor: Tensor) -> Tensor:
    """Corrected reuse of the latest anchor's source velocity at a non-anchor step.

    V̄^src_{t_i} = V^src_{τ_j(i)} + (ε_{t_i} − ε_{τ_j(i)}), Eq. (13).
    """
    return v_src_anchor + (eps - eps_anchor)


def rotational_correction(
    v_delta: Tensor, v_src: Tensor, v_ref: Tensor, beta: float, eps: float = 1e-8
) -> Tensor:
    """The steering operator R(V^Δ; V̄^src, V^ref, β) of Eq. (16).

    Closed form of the supplementary appendix, Eqs. (18)–(23) and Algorithm 2, computed in float32
    with inner products over all non-batch dimensions.
    """
    v_delta, v_src, v_ref = v_delta.float(), v_src.float(), v_ref.float()
    dims = tuple(range(1, v_delta.ndim))
    u_pred = v_src / (torch.linalg.vector_norm(v_src, dim=dims, keepdim=True) + eps)  # Eq. (18)
    u_ref = v_ref / (torch.linalg.vector_norm(v_ref, dim=dims, keepdim=True) + eps)
    cos = (u_pred * u_ref).sum(dim=dims, keepdim=True)  # Eq. (19)
    g = (1 - cos).clamp(0, 1)  # Eq. (20)
    s = beta * g  # Eq. (21)
    alpha = (v_delta * u_pred).sum(dim=dims, keepdim=True)  # Eq. (22)
    return v_delta + s * alpha * (u_ref - u_pred)  # Eq. (23)
