"""The release against the experiment code, bit for bit in float32 on CPU.

Intended differences, checked elsewhere:
- FLUX truncated β to an integer (`int(config.get("beta"))`); the release uses it as given (test_sampler).
- The FLUX entry point did not pass `config` to its StraightEdit sampler; here it is passed.
- ReFlow outputs were watermarked before saving; the release saves the decoded image.
"""

import functools

import numpy as np
import pytest
import torch
from fakes import FakeAutoEncoder, FakeEmbedder, FakeFlux, FakeFluxPipe
from original import Config, load
from PIL import Image

from straightedit.flux import FluxBackbone
from straightedit.reflow import ReflowBackbone, get_schedule
from straightedit.sampler import straightedit

SOURCE, TARGET = "a cat sitting on a wooden chair", "a dog sitting on a wooden chair"


@pytest.fixture(scope="module")
def image_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("pie") / "source.jpg"
    rng = np.random.default_rng(0)
    Image.fromarray((rng.random((512, 512, 3)) * 255).astype(np.uint8)).save(path)
    return path


@pytest.mark.parametrize(("anchor_interval", "beta"), [(4, 1.0), (4, 0.0), (3, 1.0)])
def test_flux_end_to_end(image_path, monkeypatch, anchor_interval, beta):
    flow_flux = load("flow.flux")
    config = Config(
        model_key="black-forest-labs/FLUX.1-dev",
        image_height=1024,
        image_width=1024,
        src_guidance_scale=1.0,
        tar_guidance_scale=5.0,
        gamma=0.0,
        n_max=24,
        sample_method="straightedit",
        jump_frequency=anchor_interval,
        beta=beta,
    )
    monkeypatch.setattr(
        flow_flux, "straightedit_sampling", functools.partial(flow_flux.straightedit_sampling, config=config)
    )
    torch.manual_seed(0)
    expected = flow_flux.Flux_Edit(FakeFluxPipe(), config).editing(
        image_path, SOURCE, TARGET, num_inference_steps=28
    )
    expected = (expected * 255).astype(np.uint8)

    backbone = FluxBackbone(FakeFluxPipe())
    torch.manual_seed(0)
    cond_src, cond_tgt = backbone.encode_prompt(SOURCE, 1.0), backbone.encode_prompt(TARGET, 5.0)
    z_src = backbone.encode_image(image_path)
    z_edit = straightedit(
        backbone.velocity, z_src, cond_src, cond_tgt, backbone.schedule(28, 24), anchor_interval, beta
    )
    assert np.array_equal(backbone.decode(z_edit), expected)


@pytest.fixture
def reflow(image_path):
    backbone = ReflowBackbone(
        FakeFlux(), FakeAutoEncoder(), FakeEmbedder(8, 32), FakeEmbedder(16), torch.device("cpu")
    )
    return backbone, backbone.encode_image(image_path)


def test_reflow_inputs(reflow, image_path):
    flow_edit_sampling = load("reflow.flow_edit_sampling")
    backbone, z_src = reflow
    image = np.array(Image.open(image_path).resize((1024, 1024), Image.LANCZOS).convert("RGB"))
    image = backbone.ae.encode((torch.from_numpy(image).permute(2, 0, 1).float() / 127.5 - 1)[None])
    expected = flow_edit_sampling.prepare_flow_edit(
        backbone.t5, backbone.clip, image.to(torch.bfloat16), SOURCE, TARGET
    )
    cond_src, cond_tgt = backbone.encode_prompt(SOURCE, 1.5), backbone.encode_prompt(TARGET, 5.5)
    assert torch.equal(z_src, expected["img"])
    assert torch.equal(backbone.img_ids, expected["img_ids"])
    for key, cond in [("src", cond_src), ("tgt", cond_tgt)]:
        assert torch.equal(cond["txt"], expected[f"{key}_txt"])
        assert torch.equal(cond["txt_ids"], expected[f"{key}_txt_ids"])
        assert torch.equal(cond["y"], expected[f"{key}_vec"])
    assert get_schedule(28, 4096) == flow_edit_sampling.get_schedule(28, 4096)


@pytest.mark.parametrize(
    ("num_steps", "kept_steps", "anchor_interval", "beta"),
    [(28, 23, 4, 1.0), (28, 23, 4, 0.0), (28, 23, 3, 0.5), (17, 14, 4, 1.0)],
)
def test_reflow_sampler(reflow, num_steps, kept_steps, anchor_interval, beta):
    flow_edit_sampling = load("reflow.flow_edit_sampling")
    backbone, z_src = reflow
    z_src = z_src.float()
    cond_src, cond_tgt = backbone.encode_prompt(SOURCE, 1.5), backbone.encode_prompt(TARGET, 5.5)
    info = {"inject_step": 2, "start_layer_index": 20, "end_layer_index": 37, "reuse_v": 1, "feature": {}}
    torch.manual_seed(0)
    expected, _ = flow_edit_sampling.denoise_straightedit(
        backbone.model.with_info,
        z_src,
        backbone.img_ids,
        cond_src["txt"],
        cond_tgt["txt"],
        cond_src["txt_ids"],
        cond_tgt["txt_ids"],
        cond_src["y"],
        cond_tgt["y"],
        n_max=kept_steps + 1,
        timesteps=get_schedule(num_steps, 4096),
        info=info,
        gamma=0.0,
        config=Config(beta=beta),
        jump_frequency=anchor_interval,
        src_guidance=1.5,
        tgt_guidance=5.5,
    )
    torch.manual_seed(0)
    steps = backbone.schedule(num_steps, kept_steps)
    assert torch.equal(
        straightedit(backbone.velocity, z_src, cond_src, cond_tgt, steps, anchor_interval, beta), expected
    )


@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_rc_matches_original(beta):
    from straightedit.corrections import rotational_correction

    g = torch.Generator().manual_seed(5)
    v_delta, v_src, v_ref = (torch.randn(1, 4096, 64, generator=g) for _ in range(3))
    for module in ["flow.sampling", "reflow.flow_edit_sampling"]:
        expected = load(module).steer_vdelta_by_src_dir_gap(v_delta, v_src, v_ref, strength=beta)
        assert torch.equal(rotational_correction(v_delta, v_src, v_ref, beta), expected)
