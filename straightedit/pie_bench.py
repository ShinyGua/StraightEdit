"""PIE-Bench (Ju et al., PnPInversion): data/mapping_file.json and data/annotation_images/."""

import json
from pathlib import Path
from typing import NamedTuple


class EditItem(NamedTuple):
    image_path: str  # relative to annotation_images/
    source_prompt: str
    target_prompt: str


def load_items(root: Path) -> list[EditItem]:
    """The edits in mapping-file order, with the blended-word brackets removed from the prompts."""
    mapping = json.loads((root / "mapping_file.json").read_text())
    return [
        EditItem(
            item["image_path"],
            item["original_prompt"].replace("[", "").replace("]", ""),
            item["editing_prompt"].replace("[", "").replace("]", ""),
        )
        for item in mapping.values()
    ]
