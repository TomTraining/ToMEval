"""F034：从 data_eval/_audit/<DS>.json 推导每数据集 format 规则。

替换 eval_format.py 的硬编码 PROMPT_TYPE_WHITELIST / META_REQUIRED_FIELDS。

规则推导（D034-01）：
  prompt_types          := {k for k, v in prompt_type_counts if v > 0}
  correct_count_allowed := correct_count_distribution.keys()
  wrong_count_allowed   := {n - c for n in num_options_distribution
                                  for c in correct_count_distribution if c <= n}
  meta_required         := audit.meta 中非交叉键、sum==total、无 ''/'<empty>' 桶
                            （即原始数据集 100% non-null）

缺 audit 文件直接 raise（D034-02），不回落到硬编码。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet


@dataclass(frozen=True)
class FormatRules:
    dataset: str
    prompt_types: FrozenSet[str]
    correct_count_allowed: FrozenSet[int]
    wrong_count_allowed: FrozenSet[int]
    meta_required: tuple[str, ...]


def load_rules(dataset: str, audit_root: str = "data_eval/_audit") -> FormatRules:
    audit_path = Path(audit_root) / f"{dataset}.json"
    if not audit_path.exists():
        raise FileNotFoundError(
            f"audit JSON 缺失: {audit_path}（F034 不回落到硬编码，先跑 scripts/datasets_audit.py）"
        )
    info = json.loads(audit_path.read_text(encoding="utf-8"))

    pt_counts = info.get("prompt_type_counts", {})
    prompt_types = frozenset(k for k, v in pt_counts.items() if int(v) > 0)

    cc_dist = info.get("correct_count_distribution", {})
    correct_count_allowed = frozenset(int(k) for k in cc_dist.keys())

    no_dist = info.get("num_options_distribution", {})
    wrong_count_allowed = frozenset(
        int(n) - int(c)
        for n in no_dist.keys()
        for c in cc_dist.keys()
        if int(c) <= int(n)
    )

    total = int(info["total"])
    meta_block = info.get("meta", {})
    required: list[str] = []
    for key, val in meta_block.items():
        if " x " in key:
            continue
        if not isinstance(val, dict):
            continue
        s = sum(int(v) for v in val.values())
        empty = int(val.get("", 0)) + int(val.get("<empty>", 0))
        if s == total and empty == 0:
            required.append(key)

    return FormatRules(
        dataset=dataset,
        prompt_types=prompt_types,
        correct_count_allowed=correct_count_allowed,
        wrong_count_allowed=wrong_count_allowed,
        meta_required=tuple(required),
    )
