"""FANToM 指标：逐题型 accuracy + 官方 set 级 ALL(同一 info-set 内全题答对才算通过)。

set 级 ALL 是 FANToM 的头条指标(groupby(set_id).all().mean())。我们用 meta.snippet_id
(= 官方 set_id)分组。注意：标准化数据的 belief 为多选(MC)形式，无自由文本 belief，
因此本 ALL 对应官方 "All"(MC belief)，而非需要自由文本 belief 的 "All*"。
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.evaluation.task_metrics import (
    base_metric_payload,
    group_all_correct,
    meta,
    rate_dict,
    update_group,
)

# 参与 set 级 ALL 的 ToM 题型(不含 factQA —— 它是事实控制项，非 ToM 题)。
TOM_TYPES = [
    "beliefQAs",
    "answerabilityQA_list",
    "answerabilityQAs_binary",
    "infoAccessibilityQA_list",
    "infoAccessibilityQAs_binary",
]


def _question_type(record: Dict[str, Any]) -> str:
    record_meta = meta(record)
    question_type = str(record_meta.get("question_type") or "").strip()
    if question_type:
        return question_type
    sample_id = str(record_meta.get("id") or record.get("sample_id") or "")
    if "__" in sample_id:
        parts = sample_id.split("__")
        if len(parts) >= 2:
            return parts[1]
    return "unknown"


def _snippet_id(record: Dict[str, Any]) -> str:
    record_meta = meta(record)
    snippet_id = str(record_meta.get("snippet_id") or "").strip()
    if snippet_id:
        return snippet_id
    sample_id = str(record_meta.get("id") or record.get("sample_id") or "")
    return sample_id.split("__")[0] if "__" in sample_id else (sample_id or "unknown")


def compute_metrics(records: List[Dict[str, Any]], per_sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = base_metric_payload(per_sample_results)

    grouped_stats: Dict[str, Dict[str, int]] = {}
    for record, result in zip(records, per_sample_results):
        update_group(grouped_stats, _question_type(record), result["is_correct"])
    by_qt = rate_dict(grouped_stats)

    def all_rate(required: List[str]) -> float:
        return group_all_correct(records, per_sample_results, _snippet_id, _question_type, required)["rate"]

    metrics["by_category"] = {
        # set 级 ALL：同一 snippet 内全部 ToM 题型答对才算该 set 通过(官方头条指标)。
        "overall.ALL": all_rate(TOM_TYPES),
        "answerability.ALL": all_rate(["answerabilityQA_list", "answerabilityQAs_binary"]),
        "infoaccess.ALL": all_rate(["infoAccessibilityQA_list", "infoAccessibilityQAs_binary"]),
        # 各题型逐题 accuracy(诊断)。
        "belief.accuracy": by_qt.get("beliefQAs", 0.0),
        "answerability.list.accuracy": by_qt.get("answerabilityQA_list", 0.0),
        "answerability.yn.accuracy": by_qt.get("answerabilityQAs_binary", 0.0),
        "infoaccess.list.accuracy": by_qt.get("infoAccessibilityQA_list", 0.0),
        "infoaccess.yn.accuracy": by_qt.get("infoAccessibilityQAs_binary", 0.0),
        "fact.qa.accuracy": by_qt.get("factQA", 0.0),
    }
    # set 级通过数/总数，便于核对口径。
    metrics["set_all"] = group_all_correct(
        records, per_sample_results, _snippet_id, _question_type, TOM_TYPES
    )
    return metrics
