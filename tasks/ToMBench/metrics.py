"""ToMBench 分组指标：官方同时报告 by_task(任务维度) 与 by_ability(能力维度)。

任务名取自 meta.filename(去掉 .jsonl 后缀)，与官方 "Task-oriented" 结果表对应；
原实现只有 by_ability，这里补上 by_task。
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.evaluation.task_metrics import by_meta, first_value, generic_group_metrics, meta


def _task_name(record: Dict[str, Any]) -> str:
    filename = first_value(meta(record).get("filename"))
    if filename.endswith(".jsonl"):
        filename = filename[: -len(".jsonl")]
    return filename or "unknown"


def compute_metrics(records: List[Dict[str, Any]], per_sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return generic_group_metrics(
        records,
        per_sample_results,
        [
            ("by_task", lambda record: [_task_name(record)]),
            ("by_ability", by_meta("ability")),
        ],
    )
