"""Edit the PIE-Bench images with StraightEdit."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from straightedit.flux import FluxBackbone
from straightedit.pie_bench import EditItem, load_items
from straightedit.reflow import ReflowBackbone
from straightedit.sampler import Step, straightedit

BACKBONES = {"flux": FluxBackbone, "reflow": ReflowBackbone}


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, help="YAML file of defaults for the options below")
    config_args, _ = config_parser.parse_known_args(argv)

    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    parser.add_argument("--backbone", choices=BACKBONES, default="flux")
    # Defaults: the FLUX row of Table 1.
    # 6_Appendix/6_appendix.tex:90
    parser.add_argument("--num-steps", type=int, default=28, help="steps of the full time grid")
    # 6_Appendix/6_appendix.tex:90
    parser.add_argument("--kept-steps", type=int, default=24, help="steps run, the last ones of the grid")
    # 6_Appendix/6_appendix.tex:89
    parser.add_argument("--guidance", type=float, nargs=2, default=[1.0, 5.0], metavar=("SRC", "TGT"))
    # 5_Exp/5_experiments.tex:33
    parser.add_argument(
        "--anchor-interval", type=positive_int, default=4, help="J: the source branch runs every J steps"
    )
    # 5_Exp/5_experiments.tex:33
    parser.add_argument("--beta", type=float, default=1.0, help="strength of Rotational Correction")
    parser.add_argument("--pie-bench", type=Path, required=True, help="folder with mapping_file.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    if config_args.config:
        parser.set_defaults(**yaml.safe_load(config_args.config.read_text()))
    return parser.parse_args(argv)


def edit_image(
    backbone: FluxBackbone | ReflowBackbone, item: EditItem, steps: list[Step], args: argparse.Namespace
) -> np.ndarray:
    cond_src = backbone.encode_prompt(item.source_prompt, args.guidance[0])
    cond_tgt = backbone.encode_prompt(item.target_prompt, args.guidance[1])
    z_src = backbone.encode_image(args.pie_bench / "annotation_images" / item.image_path)
    z_edit = straightedit(
        backbone.velocity, z_src, cond_src, cond_tgt, steps, args.anchor_interval, args.beta
    )
    return backbone.decode(z_edit)


@torch.inference_mode()
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    backbone = BACKBONES[args.backbone].load(torch.device(args.device))
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))

    steps = backbone.schedule(args.num_steps, args.kept_steps)
    for item in tqdm(load_items(args.pie_bench)):
        output = args.output / item.image_path
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(edit_image(backbone, item, steps, args)).save(output)


if __name__ == "__main__":
    main()
