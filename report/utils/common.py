from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def parse_model_entry(entry: Any) -> Tuple[str, str]:
    if isinstance(entry, str):
        return entry, entry
    if isinstance(entry, dict):
        name = str(entry["name"])
        display = str(entry.get("display") or name)
        return name, display
    raise ValueError(f"Invalid model entry: {entry!r}")


def format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
