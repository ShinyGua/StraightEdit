import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from edit import parse_args

CODE = Path(__file__).resolve().parents[1]
CONFIGS = sorted((CODE / "configs").glob("*.yaml"))
RESEARCH_KEYS = {"num_steps", "kept_steps", "guidance", "anchor_interval", "beta"}


def test_help():
    result = subprocess.run([sys.executable, "edit.py", "--help"], cwd=CODE, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--anchor-interval" in result.stdout


@pytest.mark.parametrize("config", CONFIGS, ids=lambda p: p.stem)
def test_config(config):
    values = yaml.safe_load(config.read_text())
    assert values.keys() <= vars(parse_args(["--pie-bench", "data", "--output", "out"])).keys()
    args = parse_args(["--config", str(config), "--pie-bench", "data", "--output", "out"])
    assert all(getattr(args, key) == value for key, value in values.items())
    for line in config.read_text().splitlines():
        key = line.split(":")[0]
        if key in RESEARCH_KEYS:
            assert re.search(r"# \S+\.tex:\d+", line), f"{key} cites no paper line"


def test_anchor_interval_is_at_least_one():
    with pytest.raises(SystemExit):
        parse_args(["--anchor-interval", "0", "--pie-bench", "data", "--output", "out"])


def test_readme_commands_parse():
    commands = re.findall(r"`python edit\.py ([^`]+)`", (CODE / "README.md").read_text())
    assert len(commands) >= 3
    for command in commands:
        if command != "--help":
            args = parse_args(command.split())
            assert args.config is None or args.config.name in {path.name for path in CONFIGS}
