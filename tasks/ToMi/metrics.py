from __future__ import annotations

from typing import Any, Dict, List

from src.evaluation.task_metrics import by_meta, hierarchical_metrics


def compute_metrics(records: List[Dict[str, Any]], per_sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return hierarchical_metrics(
        records,
        per_sample_results,
        [
            # 真实字段：story_type（故事型：真/假信念、二阶假信念）
            # 与 question_type（题型：一/二阶 + 是否需要 ToM、memory、reality）。
            ("story_type", by_meta("story_type")),
            ("question_type", by_meta("question_type")),
        ],
    )
