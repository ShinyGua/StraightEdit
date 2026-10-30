import pytest
import torch

from straightedit.corrections import noise_correction, rotational_correction


def tensors(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(2, 16, 64, generator=g) for _ in range(n)]


def unit(v):
    return v / v.flatten(1).norm(dim=1)[:, None, None]


def dot(a, b):
    return (a * b).flatten(1).sum(dim=1)[:, None, None]


def test_noise_correction():
    v_anchor, eps, eps_anchor = tensors(3)
    assert torch.equal(noise_correction(v_anchor, eps, eps_anchor), v_anchor + (eps - eps_anchor))


@pytest.mark.parametrize("beta", [0.3, 1.0])
def test_rc_matches_closed_form(beta):
    """V̄^Δ = r + α((1 − s) u_pred + s u_ref): the residual r is kept, the parallel part is rotated."""
    v_delta, v_src, v_ref = tensors(3)
    u_pred, u_ref = unit(v_src), unit(v_ref)
    alpha = dot(v_delta, u_pred)
    s = beta * (1 - dot(u_pred, u_ref)).clamp(0, 1)
    residual = v_delta - alpha * u_pred
    expected = residual + alpha * ((1 - s) * u_pred + s * u_ref)
    torch.testing.assert_close(rotational_correction(v_delta, v_src, v_ref, beta), expected)


def test_rc_beta_zero_is_identity():
    v_delta, v_src, v_ref = tensors(3)
    assert torch.equal(rotational_correction(v_delta, v_src, v_ref, 0.0), v_delta)


def test_rc_keeps_the_component_outside_the_two_directions():
    v_delta, v_src, v_ref = tensors(3)
    e1 = unit(v_src)
    e2 = unit(v_ref - dot(v_ref, e1) * e1)

    def outside(v):
        return v - dot(v, e1) * e1 - dot(v, e2) * e2

    torch.testing.assert_close(outside(rotational_correction(v_delta, v_src, v_ref, 1.0)), outside(v_delta))


def test_rc_gap_is_clipped_to_one():
    """Opposite directions give 1 − cos θ = 2, clipped to g = 1, so s = β."""
    v_delta, v_src = tensors(2)
    beta = 0.7
    u = unit(v_src)
    expected = v_delta + beta * dot(v_delta, u) * (-u - u)
    torch.testing.assert_close(rotational_correction(v_delta, v_src, -v_src, beta), expected)


def test_rc_aligned_directions_change_nothing():
    v_delta, v_src = tensors(2)
    torch.testing.assert_close(rotational_correction(v_delta, v_src, 3 * v_src, 1.0), v_delta)
