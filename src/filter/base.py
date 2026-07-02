"""filter 公共工具：模型客户端加载。

复用 src.runner.create_content_client / create_llm_client 构造客户端，
不再手工 new ContentClient/StructureClient；本模块只负责：
  1. 读取 src/filter/config.yaml 的 models.<role> 段；
  2. 注入 filter 侧默认值（temperature 0.3 / max_workers 8 / enable_thinking False /
     answer max_tokens 512、judge max_tokens 4096），config 显式给出的值优先；
  3. strong/simple 角色 → 配置段的映射。
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml

from src.runner import create_content_client, create_llm_client

# 默认读 src/filter/config.yaml；可用环境变量 FILTER_CONFIG 覆盖为其它配置路径，
# 便于跑隔离的 smoke/回归配置（如 scripts/smoke/filter.smoke.yaml）而不改动生产配置。
# pipeline.py 从本模块 import _CONFIG_PATH 复用同一路径，故只需在此处集中处理。
_CONFIG_PATH = Path(os.environ.get("FILTER_CONFIG") or Path(__file__).parent / "config.yaml")
_REQUIRED_ROLES = ("strong", "simple")

# filter 侧默认值（与历史手工构造保持一致；区别于 from_config 的 0.6/32/32768/True）。
# 注意：config.yaml 显式给出的同名字段优先于这里的默认。
_FILTER_DEFAULTS = {
    "temperature": 0.3,
    "max_workers": 8,
    "enable_thinking": False,
}


def _entry_with_defaults(entry: Dict[str, Any], default_max_tokens: int) -> Dict[str, Any]:
    """复制配置段并注入 filter 侧默认（不覆盖 config 已显式给出的字段）。"""
    config = dict(entry)
    for key, value in _FILTER_DEFAULTS.items():
        config.setdefault(key, value)
    config.setdefault("max_tokens", default_max_tokens)
    return config


def load_answer_models() -> Dict[str, Any]:
    """解析 config.yaml models dict，返回 {'strong': client, 'simple': client}。"""
    cfg = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    models = cfg.get("models")
    if not isinstance(models, dict) or any(r not in models for r in _REQUIRED_ROLES):
        raise ValueError(
            f"config.yaml models 必须为 dict 且含键 {_REQUIRED_ROLES}，得到: {type(models).__name__} keys={list(models) if isinstance(models, dict) else None}"
        )
    return {
        role: create_content_client(_entry_with_defaults(models[role], 512))
        for role in _REQUIRED_ROLES
    }


def load_judge_client(role: str = "strong"):
    """复用 models.<role> 配置构造 StructureClient（用于 judge 和 repair）。"""
    cfg = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    models = cfg.get("models") or {}
    entry = models.get(role)
    if not isinstance(entry, dict):
        raise ValueError(f"config.yaml models.{role} 必须存在且为 dict")
    return create_llm_client(_entry_with_defaults(entry, 4096))
