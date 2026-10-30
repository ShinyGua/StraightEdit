import pytest
import torch
from fakes import FakeEmbedder, FakeFlux, FakeFluxPipe, velocity_fn

from straightedit.flux import FluxBackbone
from straightedit.reflow import ReflowBackbone
from straightedit.sampler import Step, straightedit


def toy_velocity(x, t, cond):
    return velocity_fn(x, torch.tensor([t]), cond)


def toy_steps(n=10):
    t = torch.linspace(1, 0, n + 1).tolist()
    return [Step(t[i], t[i + 1], t[i]) for i in range(n)]


def algorithm_1(velocity, z_src, c_src, c_tgt, steps, J, beta):
    """Algorithm 1 with the RC closed form (Appendix A) and the target averaging of the implementation."""
    z = z_src.clone()
    cache, v_tgt_prev = None, None
    for i, (t, t_next, model_t) in enumerate(steps):
        eps = torch.randn_like(z_src)
        q = (1 - t) * z_src + t * eps
        p = z + q - z_src
        v_tgt_raw = velocity(p, model_t, c_tgt)
        v_tgt = v_tgt_raw if v_tgt_prev is None else 0.5 * (v_tgt_raw + v_tgt_prev)
        v_tgt_prev = v_tgt_raw
        if i % J == 0 or i == len(steps) - 1:
            v_src = velocity(q, model_t, c_src)
            cache = (v_src, eps)
        else:
            v_src = cache[0] + (eps - cache[1])
        v_delta = v_tgt - v_src
        v_ref = eps - z_src
        u_pred = v_src / v_src.norm()
        u_ref = v_ref / v_ref.norm()
        s = beta * (1 - (u_pred * u_ref).sum()).clamp(0, 1)
        alpha = (v_delta * u_pred).sum()
        z = z + (t_next - t) * (v_delta + s * alpha * (u_ref - u_pred))
    return z


@pytest.mark.parametrize("J", [1, 3, 4])
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_matches_algorithm_1(J, beta):
    z_src = torch.randn(1, 16, 64, generator=torch.Generator().manual_seed(1))
    args = (toy_velocity, z_src, torch.tensor(0.3), torch.tensor(-0.2), toy_steps())
    torch.manual_seed(0)
    expected = algorithm_1(*args, J, beta)
    torch.manual_seed(0)
    torch.testing.assert_close(straightedit(*args, J, beta), expected)


def test_beta_is_not_rounded():
    z_src = torch.randn(1, 16, 64, generator=torch.Generator().manual_seed(1))
    outputs = []
    for beta in [0.0, 0.5, 1.0]:
        torch.manual_seed(0)
        outputs.append(
            straightedit(toy_velocity, z_src, torch.tensor(0.3), torch.tensor(-0.2), toy_steps(), 4, beta)
        )
    assert not torch.equal(outputs[0], outputs[1]) and not torch.equal(outputs[1], outputs[2])


def count_calls(backbone, model, num_steps, kept_steps):
    z_src = torch.zeros(1, 4096, 64)
    cond_src, cond_tgt = backbone.encode_prompt("a cat", 1.5), backbone.encode_prompt("a dog", 5.5)
    straightedit(
        backbone.velocity, z_src, cond_src, cond_tgt, backbone.schedule(num_steps, kept_steps), 4, 1.0
    )
    n_tgt = sum(c == model.calls[0] for c in model.calls)
    return len(model.calls), n_tgt


def reflow_backbone(model):
    return ReflowBackbone(model, None, FakeEmbedder(8, 32), FakeEmbedder(16), torch.device("cpu"))


def test_model_calls_flux_nfe30():
    pipe = FakeFluxPipe()
    backbone = FluxBackbone(pipe)
    backbone.img_ids = torch.zeros(4096, 3)
    assert count_calls(backbone, pipe.transformer, 28, 24) == (31, 24)


@pytest.mark.parametrize(("num_steps", "kept_steps", "calls"), [(28, 23, (30, 23)), (17, 14, (19, 14))])
def test_model_calls_reflow(num_steps, kept_steps, calls):
    model = FakeFlux()
    assert count_calls(reflow_backbone(model), model, num_steps, kept_steps) == calls
