"""ToMato 分组指标。

标准化数据里 meta.dimension 实为单槽（仅心智状态一项，如 ['emotion']），此前按
slot1→slot2→slot3 嵌套出的三/四级维度全是占位空桶，纯属误导，已删除。改用真实字段：
mental_state（心智状态大类）/ order（推理阶数）/ false_belief（是否错误信念）。
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.evaluation.task_metrics import by_meta, hierarchical_metrics


def compute_metrics(records: List[Dict[str, Any]], per_sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return hierarchical_metrics(
        records,
        per_sample_results,
        [
            ("mental_state", by_meta("mental_state")),
            ("order", by_meta("order")),
            ("false_belief", by_meta("false_belief")),
        ],
    )
