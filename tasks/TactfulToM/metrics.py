"""TactfulToM 分组指标：按问题大类 / 细粒度 question_type / 白谎类型 / ToM 阶数切面。"""

from __future__ import annotations

from typing import Any, Dict, List

from src.evaluation.task_metrics import first_value, generic_group_metrics, meta


def compute_metrics(
    records: List[Dict[str, Any]],
    per_sample_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return generic_group_metrics(
        records,
        per_sample_results,
        [
            ("by_category", lambda r: [first_value(meta(r).get("category"))]),
            ("by_question_type", lambda r: [first_value(meta(r).get("question_type"))]),
            ("by_lie_type", lambda r: [first_value(meta(r).get("lie_type"))]),
            ("by_tom_type", lambda r: [first_value(meta(r).get("tom_type"))]),
        ],
    )
