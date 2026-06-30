"""EmoBench 分层指标。

EU 子集已在预测阶段合并为 mcq_grouped(情绪+原因 bundled)，judge 要求两问都对才算对，
因此 grouped 记录的 is_correct 即官方 EU 联合判分；subset 维度的 emotional_understanding
即 EU 联合准确率。

维度层级：coarse_category → finegrained_category 为天然二→三级嵌套，其余 subset / language /
question_subtype / dimension 为平铺二级维度；eu_subquestion 为汇总型维度（从 grouped 的
sub_results 聚合 emotion / cause 各自 accuracy，作 EU 子问题诊断）。
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.evaluation.task_metrics import (
    add_dimension,
    by_meta,
    hierarchical_metrics,
    make_split,
    meta,
    value_list,
)


def compute_metrics(records: List[Dict[str, Any]], per_sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = hierarchical_metrics(
        records,
        per_sample_results,
        [
            ("subset", by_meta("subset")),
            ("language", by_meta("language")),
            ("question_subtype", by_meta("question_subtype")),
            # coarse → finegrained 天然嵌套（三级）。
            ("coarse_category", by_meta("coarse_category"), [
                ("finegrained_category", by_meta("finegrained_category")),
            ]),
            ("dimension", lambda record: value_list(meta(record).get("dimension"))),
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
    if sub_stats:
        add_dimension(metrics, "eu_subquestion", {
            subtype: make_split(bucket["correct"] / bucket["total"] if bucket["total"] else 0.0, bucket["total"])
            for subtype, bucket in sorted(sub_stats.items())
        })
    return metrics
