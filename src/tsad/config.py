from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    cfg_path = Path(path) if path else root / "configs" / "default.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(root)
    cfg["_project_root"] = str(root)
    return cfg


def resolve_path(cfg: dict, key: str) -> Path:
    root = Path(cfg["_root"])
    return root / cfg["paths"][key]
