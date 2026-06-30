"""FictionalQA 分层指标。

- style：按虚构文体切分（切分型二级维度）。
- grading：informed（模型在给定上下文下的 ACC）vs blind（无上下文盲评均分），
  两者之差即官方关注的 informed-vs-blind gap。汇总型维度，两个 split。
- macro_split_acc：按 event / document / style 分组后的宏平均 ACC（先组内再跨组平均，
  消除大组主导）。汇总型维度，每个 split 是一种分组口径下的宏平均。

event/document 基本一题一组，逐组明细无展示价值，只保留其宏平均；故全部口径统一收进
dimensions，顶层不再留散落的特殊键。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.evaluation.task_metrics import (
    add_dimension,
    first_value,
    hierarchical_metrics,
    make_split,
    meta,
    rate_dict,
    update_group,
)


def _ids(record: Dict[str, Any]) -> Tuple[str, str, str]:
    record_meta = meta(record)
    sample_id = str(record_meta.get("id") or record.get("sample_id") or "")
    event_id = "unknown"
    doc_id = "unknown"
    style = first_value(record_meta.get("fiction_type"))

    if "_style_" in sample_id:
        event_id = sample_id.split("_style_")[0] or "unknown"
        style_suffix = sample_id.split("_style_")[1].split("_")[0]
        if style == "unknown" and style_suffix:
            style = style_suffix
    if "_question_" in sample_id:
        doc_id = sample_id.split("_question_")[0] or "unknown"
    elif sample_id:
        doc_id = sample_id

    return event_id, doc_id, style


def _style(record: Dict[str, Any]) -> List[str]:
    return [_ids(record)[2]]


def _macro(stats: Dict[str, Dict[str, int]]) -> Tuple[float, int]:
    """宏平均（各组 ACC 再平均）与组数。"""
    values = list(rate_dict(stats).values())
    return (sum(values) / len(values) if values else 0.0, len(values))


def compute_metrics(records: List[Dict[str, Any]], per_sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = hierarchical_metrics(records, per_sample_results, [("style", _style)])

    by_event: Dict[str, Dict[str, int]] = {}
    by_document: Dict[str, Dict[str, int]] = {}
    by_style: Dict[str, Dict[str, int]] = {}
    blind_values: List[float] = []

    for record, result in zip(records, per_sample_results):
        event_id, doc_id, style = _ids(record)
        update_group(by_event, event_id, result["is_correct"])
        update_group(by_document, doc_id, result["is_correct"])
        update_group(by_style, style, result["is_correct"])

        blind_value = meta(record).get("blind_grade_avg")
        if isinstance(blind_value, (int, float)):
            blind_values.append(float(blind_value))

    # grading：informed（模型 ACC）vs blind（盲评均分），差值即 informed-vs-blind gap。
    grading: Dict[str, Any] = {"informed": make_split(metrics["accuracy"], metrics["total"])}
    if blind_values:
        grading["blind"] = make_split(sum(blind_values) / len(blind_values), len(blind_values))
    add_dimension(metrics, "grading", grading)

    # macro_split_acc：三种分组口径下的宏平均 ACC（n=组数）。
    event_macro, event_n = _macro(by_event)
    doc_macro, doc_n = _macro(by_document)
    style_macro, style_n = _macro(by_style)
    add_dimension(metrics, "macro_split_acc", {
        "event": make_split(event_macro, event_n),
        "document": make_split(doc_macro, doc_n),
        "style": make_split(style_macro, style_n),
    })
    return metrics
