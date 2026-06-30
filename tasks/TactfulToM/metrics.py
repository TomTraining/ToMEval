"""TactfulToM 分层指标：按问题大类 / 细粒度 question_type / 白谎类型 / ToM 阶数切分（切分型二级维度），

外加二级维度 joint_comp_just —— 同一对话(set_id)的 Comprehension 与 Justification
都答对才算"真正理解了善意谎言"(Happé 双题判定)。该联合分无法从两个边际 accuracy
反推，作为单 split（overall=全部对话）的二级维度收进 dimensions。
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.evaluation.task_metrics import (
    add_dimension,
    first_value,
    group_all_correct,
    hierarchical_metrics,
    make_split,
    meta,
)


def compute_metrics(
    records: List[Dict[str, Any]],
    per_sample_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    metrics = hierarchical_metrics(
        records,
        per_sample_results,
        [
            ("category", lambda r: [first_value(meta(r).get("category"))]),
            ("question_type", lambda r: [first_value(meta(r).get("question_type"))]),
            ("lie_type", lambda r: [first_value(meta(r).get("lie_type"))]),
            ("tom_type", lambda r: [first_value(meta(r).get("tom_type"))]),
        ],
    )

    # 同一对话 comprehension ∧ justification 都对。
    stats = group_all_correct(
        records,
        per_sample_results,
        key_fn=lambda r: first_value(meta(r).get("set_id")),
        member_fn=lambda r: first_value(meta(r).get("category")),
        required_members=["comprehension", "justification"],
    )
    add_dimension(metrics, "joint_comp_just", {
        "overall": make_split(stats["rate"], stats["total"]),
    })
    return metrics
