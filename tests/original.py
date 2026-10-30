"""Read-only import of the experiment code the release was refactored from.

Set STRAIGHTEDIT_ORIGINAL to its folder; by default it is the supplementary material next to this
repository. Tests that use it are skipped when it is absent.
"""

import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(
    os.environ.get(
        "STRAIGHTEDIT_ORIGINAL",
        Path(__file__).resolve().parents[2] / "Supplementary Material" / "Code for StraightEdit",
    )
)


def load(module: str):
    if not (ROOT / "flow" / "sampling.py").exists():
        pytest.skip(f"original code not found at {ROOT}")
    sys.dont_write_bytecode = True
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    return importlib.import_module(module)


class Config(dict):
    """The attribute and .get access the original code uses on its OmegaConf config."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error
