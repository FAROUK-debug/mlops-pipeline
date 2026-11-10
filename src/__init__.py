from pathlib import Path

import yaml

# Project root, so every path works whatever folder the code is launched from.
ROOT = Path(__file__).resolve().parents[1]
TARGET = "target"


def load_params():
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)
