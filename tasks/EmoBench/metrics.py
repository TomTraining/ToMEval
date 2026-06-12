"""EmoBench 分组指标。

EU 子集已在预测阶段合并为 mcq_grouped(情绪+原因 bundled)，judge 要求两问都对才算对，
因此 grouped 记录的 is_correct 即官方 EU 联合判分；by_subset 的 emotional_understanding
即 EU 联合准确率。额外从 grouped 的 sub_results 聚合 emotion / cause 各自 accuracy 作诊断。
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.evaluation.task_metrics import (
    by_meta,
    generic_group_metrics,
    meta,
    safe_div,
    value_list,
)


def compute_metrics(records: List[Dict[str, Any]], per_sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = generic_group_metrics(
        records,
        per_sample_results,
        [
            ("by_subset", by_meta("subset")),
            ("by_language", by_meta("language")),
            ("by_question_subtype", by_meta("question_subtype")),
            ("by_coarse_category", by_meta("coarse_category")),
            ("by_finegrained_category", by_meta("finegrained_category")),
            ("by_dimension", lambda record: value_list(meta(record).get("dimension"))),
        ],
    )

    # EU 子问题诊断：从 grouped 记录的 sub_results 聚合 emotion / cause 各自 accuracy。
    sub_stats: Dict[str, Dict[str, int]] = {}
    for result in per_sample_results:
        for sub in result.get("sub_results") or []:
            subtype = sub.get("subtype") or "unknown"
            bucket = sub_stats.setdefault(subtype, {"correct": 0, "total": 0})
            bucket["total"] += 1
            if sub.get("is_correct"):
                bucket["correct"] += 1
    metrics["eu_subquestion_accuracy"] = {
        subtype: safe_div(bucket["correct"], bucket["total"])
        for subtype, bucket in sorted(sub_stats.items())
    }
    return metrics
