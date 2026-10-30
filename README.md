# StraightEdit: Efficient Low-Step Inversion-Free Text-Guided Image Editing

Xiyu Wang\*, Mengyu Zheng\*, Daochang Liu, Chang Xu. ACM MM '26. [Paper](https://doi.org/10.1145/3767308.3835799)

StraightEdit speeds up inversion-free flow editing (FlowEdit) by running the source branch only at
sparse anchor steps, correcting each reused source velocity for the fresh noise draw (Noise
Correction), and steering every update towards the straight rectified-flow direction (Rotational
Correction).

## Install

Python 3.10 and an NVIDIA GPU with CUDA 12.4. The models run in bfloat16.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

GPU memory: about 34 GiB (FLUX) and 35 GiB (ReFlow) at peak, measured on an NVIDIA H20.

## Data

Download PIE-Bench from [PnPInversion](https://github.com/cure-lab/PnPInversion) and keep its layout:

```
data/
  mapping_file.json
  annotation_images/0_random_140/...
```

## Weights

FLUX.1-dev is gated. Accept its licence on
[Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-dev), then

```bash
export HUGGINGFACE_TOKEN=hf_...
export HF_HOME=/path/to/cache   # optional
```

Both backbones use the FLUX.1-dev weights, which download on first use. The `flux` backbone loads
the diffusers files (about 34 GB). The `reflow` backbone loads `flux1-dev.safetensors` and
`ae.safetensors` from the same repository (about 24 GB), plus `google/t5-v1_1-xxl` (about 45 GB)
and `openai/clip-vit-large-patch14`.

## Run

One command per StraightEdit row of Table 1:

| Table 1 row | Command | Paper NFE | Transformer calls |
|---|---|---|---|
| Ours, FLUX | `python edit.py --config configs/flux-nfe30.yaml --pie-bench data --output results/flux-nfe30` | 30 | 31 |
| Ours, ReFlow | `python edit.py --config configs/reflow-nfe30.yaml --pie-bench data --output results/reflow-nfe30` | 30 | 30 |
| Ours, ReFlow | `python edit.py --config configs/reflow-nfe18.yaml --pie-bench data --output results/reflow-nfe18` | 18 | 19 |

Each edit is saved as a 1024 × 1024 image at its PIE-Bench path under `--output`, next to the
`config.json` of the run. Options on the command line override the config; `python edit.py --help`
lists them.

## Evaluate

The paper uses the PIE-Bench metrics of PnPInversion (`evaluation/evaluate.py` there). They are
computed at 512 × 512, so resize the edits with PIL's default filter first.

## Paper → code

| Paper | Code |
|---|---|
| Algorithm 1 | `straightedit()` in `straightedit/sampler.py` |
| Coupled queries q_t, p_t, Eqs. (5)–(6) | `straightedit()` |
| Anchor set 𝒯 with jump frequency J | `straightedit(..., anchor_interval)`, `--anchor-interval` |
| Noise Correction, Eq. (13) | `noise_correction()` in `straightedit/corrections.py` |
| Reference velocity V^ref, Eq. (14) | `straightedit()` |
| Rotational Correction, Eq. (16); closed form, appendix Eqs. (18)–(23), Algorithm 2 | `rotational_correction()` in `straightedit/corrections.py` |
| Update, Eq. (17) | `straightedit()` |
| Strength β | `--beta` |
| V_θ on FLUX | `FluxBackbone.velocity()` in `straightedit/flux.py` |
| V_θ on ReFlow | `ReflowBackbone.velocity()` in `straightedit/reflow.py` |

## Implementation notes

- After the first step, the target velocity is the mean of the current and previous target
  predictions. Algorithm 1 omits this step.
- The source branch also runs at the last step. FLUX runs the last 24 steps of a 28-step grid
  (31 model calls). ReFlow runs the last 23 of 28 (30 calls), or the last 14 of 17 (19 calls).
- ReFlow is FLUX.1-dev in the reference implementation of Black Forest Labs, as set up by
  RF-Solver, with guidance (1.5, 5.5). RF-Solver's attention caching is not applied. In the paper's
  setup it was enabled only for the first two grid steps, which the kept-step schedule skips, so it
  had no effect.
- ReFlow outputs are saved without the invisible watermark of the BFL reference code.
- Seeds: 42, and 1228 for the 18-NFE row. The noise depends on each image's position in the mapping
  file.

## Development

```bash
pip install -r requirements-dev.txt
pytest
ruff check . && ruff format --check . && vulture
```

The tests run on CPU with stand-in velocity models. They also compare the code against the
experiment code when it is available (`STRAIGHTEDIT_ORIGINAL`).

## Acknowledgements and licence

The code is released under the [Apache License 2.0](LICENSE). It builds on
[FLUX](https://github.com/black-forest-labs/flux), [RF-Solver](https://github.com/wangjiangshan0725/RF-Solver-Edit),
[FireFlow](https://github.com/HolmesShuan/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing),
[diffusers](https://github.com/huggingface/diffusers) and [FlowEdit](https://github.com/fallenshock/FlowEdit);
see [NOTICE](NOTICE) and the [FlowEdit MIT licence](LICENSES/FlowEdit-MIT.txt).
PIE-Bench and its metrics come from [PnPInversion](https://github.com/cure-lab/PnPInversion).
The FLUX.1-dev weights are under the FLUX.1 [dev] Non-Commercial License.

## Citation

```bibtex
@inproceedings{wang2026straightedit,
  title     = {StraightEdit: Efficient Low-Step Inversion-Free Text-Guided Image Editing},
  author    = {Wang, Xiyu and Zheng, Mengyu and Liu, Daochang and Xu, Chang},
  booktitle = {Proceedings of the ACM International Conference on Multimedia (MM '26)},
  year      = {2026},
  address   = {Rio de Janeiro, Brazil},
  doi       = {10.1145/3767308.3835799}
}
```
