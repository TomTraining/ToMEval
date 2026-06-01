"""filter 公共工具：模型客户端加载。"""

from pathlib import Path
from typing import Any, Dict

import yaml

from src.llm.content_client import ContentClient

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_REQUIRED_ROLES = ("strong", "simple")


def _build_client(entry: Dict[str, Any], default_max_tokens: int = 512) -> ContentClient:
    return ContentClient(
        model_name=entry["model_name"],
        api_key=entry["api_key"],
        api_url=entry["api_url"],
        temperature=entry.get("temperature", 0.3),
        max_workers=entry.get("max_workers", 8),
        max_tokens=entry.get("max_tokens", default_max_tokens),
        enable_thinking=False,
    )


def load_answer_models() -> Dict[str, ContentClient]:
    """解析 config.yaml models dict，返回 {'strong': client, 'simple': client}。"""
    cfg = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    models = cfg.get("models")
    if not isinstance(models, dict) or any(r not in models for r in _REQUIRED_ROLES):
        raise ValueError(
            f"config.yaml models 必须为 dict 且含键 {_REQUIRED_ROLES}，得到: {type(models).__name__} keys={list(models) if isinstance(models, dict) else None}"
        )
    return {role: _build_client(models[role]) for role in _REQUIRED_ROLES}


def load_judge_client(role: str = "strong"):
    """复用 models.<role> 配置构造 StructureClient（用于 judge 和 repair）。"""
    from src.llm.structure_client import StructureClient

    cfg = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    models = cfg.get("models") or {}
    entry = models.get(role)
    if not isinstance(entry, dict):
        raise ValueError(f"config.yaml models.{role} 必须存在且为 dict")
    return StructureClient(
        model_name=entry["model_name"],
        api_key=entry["api_key"],
        api_url=entry["api_url"],
        temperature=entry.get("temperature", 0.3),
        max_workers=entry.get("max_workers", 8),
        max_tokens=entry.get("max_tokens", 4096),
        enable_thinking=False,
    )
