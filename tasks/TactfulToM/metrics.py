"""TactfulToM 分组指标：按问题大类 / 细粒度 question_type / 白谎类型 / ToM 阶数切面，

外加二级指标 joint_comp_just —— 同一对话(set_id)的 Comprehension 与 Justification
都答对才算"真正理解了善意谎言"(Happé 双题判定)。该联合分无法从两个边际 accuracy
反推，必须按对话分组计算。
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.evaluation.task_metrics import (
    first_value,
    generic_group_metrics,
    group_all_correct,
    meta,
)


def compute_metrics(
    records: List[Dict[str, Any]],
    per_sample_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    metrics = generic_group_metrics(
        records,
        per_sample_results,
        [
            ("by_category", lambda r: [first_value(meta(r).get("category"))]),
            ("by_question_type", lambda r: [first_value(meta(r).get("question_type"))]),
            ("by_lie_type", lambda r: [first_value(meta(r).get("lie_type"))]),
            ("by_tom_type", lambda r: [first_value(meta(r).get("tom_type"))]),
        ],
    )
    # 二级指标：同一对话 comprehension ∧ justification 都对。
    metrics["joint_comp_just"] = group_all_correct(
        records,
        per_sample_results,
        key_fn=lambda r: first_value(meta(r).get("set_id")),
        member_fn=lambda r: first_value(meta(r).get("category")),
        required_members=["comprehension", "justification"],
    )
    return metrics
