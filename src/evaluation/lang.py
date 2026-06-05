from __future__ import annotations

from typing import Any


def get_sample_lang(meta: Any) -> str:
    """从样本 meta 归一化语言标记。

    兼容 ToMBench 的 meta.lang 和 EmoBench 的 meta.language；
    zh / zh-CN / ZH 等统一归一化为 "zh"，其余（含缺失）一律视为 "en"。
    """
    if not isinstance(meta, dict):
        return "en"
    raw = str(meta.get("lang") or meta.get("language") or "en").strip().lower()
    return "zh" if raw.startswith("zh") else "en"
