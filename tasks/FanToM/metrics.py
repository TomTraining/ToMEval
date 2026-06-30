"""FANToM 指标：分层维度。

- question_type：逐题型准确率（切分型二级维度）。
- set_all：官方 set 级 ALL（同一 info-set 内全题答对才算通过，groupby(set_id).all().mean()），
  是 FANToM 头条指标。按 overall / answerability / infoaccess 切成三个 split。
  注意：标准化数据的 belief 为多选(MC)形式，无自由文本 belief，因此本 ALL 对应官方
  "All"(MC belief)，而非需要自由文本 belief 的 "All*"。

set_all 无法从逐题型 ACC 反推，作为独立二级维度收进 dimensions（不再散落在顶层）。
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.evaluation.task_metrics import (
    add_dimension,
    group_all_correct,
    hierarchical_metrics,
    make_split,
    meta,
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
    metrics = hierarchical_metrics(
        records,
        per_sample_results,
        [("question_type", lambda r: [_question_type(r)])],
    )

    def all_split(required: List[str]):
        stats = group_all_correct(records, per_sample_results, _snippet_id, _question_type, required)
        return make_split(stats["rate"], stats["total"])

    # set 级 ALL：同一 snippet 内指定题型全对才算该 set 通过。
    add_dimension(metrics, "set_all", {
        "overall": all_split(TOM_TYPES),
        "answerability": all_split(["answerabilityQA_list", "answerabilityQAs_binary"]),
        "infoaccess": all_split(["infoAccessibilityQA_list", "infoAccessibilityQAs_binary"]),
    })
    return metrics
