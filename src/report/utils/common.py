from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


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
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def format_accuracy(value: Any) -> str:
    """与 format_metric_value 类似，但把空值渲染为 "-"（总览表格用）。"""
    if isinstance(value, float):
        return f"{value:.4f}"
    if value in (None, ""):
        return "-"
    return str(value)
