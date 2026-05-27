from __future__ import annotations

from typing import Any, Dict, List, Sequence

from src.evaluation.task_metrics import base_metric_payload, meta, rate_dict, safe_div, update_group


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


def _all_metric(
    records: List[Dict[str, Any]],
    per_sample_results: List[Dict[str, Any]],
    required_types: Sequence[str],
) -> float:
    by_snippet: Dict[str, Dict[str, bool]] = {}
    for record, result in zip(records, per_sample_results):
        snippet_id = _snippet_id(record)
        question_type = _question_type(record)
        if snippet_id not in by_snippet:
            by_snippet[snippet_id] = {}
        current = by_snippet[snippet_id].get(question_type, True)
        by_snippet[snippet_id][question_type] = current and bool(result["is_correct"])

    hits = 0
    total = 0
    for grouped in by_snippet.values():
        if not all(question_type in grouped for question_type in required_types):
            continue
        total += 1
        if all(grouped[question_type] for question_type in required_types):
            hits += 1
    return safe_div(hits, total)


def compute_metrics(records: List[Dict[str, Any]], per_sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = base_metric_payload(per_sample_results)
    grouped_stats: Dict[str, Dict[str, int]] = {}
    for record, result in zip(records, per_sample_results):
        update_group(grouped_stats, _question_type(record), result["is_correct"])

    by_question_type = rate_dict(grouped_stats)

    metrics["by_category"] = {
        "overall.ALL": _all_metric(
            records,
            per_sample_results,
            [
                "beliefQAs_choice",
                "answerabilityQA_list",
                "answerabilityQAs_binary",
                "infoAccessibilityQA_list",
                "infoAccessibilityQAs_binary",
            ],
        ),
        "overall.ALL_star": _all_metric(
            records,
            per_sample_results,
            [
                "beliefQAs",
                "beliefQAs_choice",
                "answerabilityQA_list",
                "answerabilityQAs_binary",
                "infoAccessibilityQA_list",
                "infoAccessibilityQAs_binary",
            ],
        ),
        "belief.qa.accuracy": by_question_type.get("beliefQAs", 0.0),
        "belief.choice.accuracy": by_question_type.get("beliefQAs_choice", 0.0),
        "answerability.ALL": _all_metric(
            records,
            per_sample_results,
            ["answerabilityQA_list", "answerabilityQAs_binary"],
        ),
        "answerability.list.accuracy": by_question_type.get("answerabilityQA_list", 0.0),
        "answerability.yn.accuracy": by_question_type.get("answerabilityQAs_binary", 0.0),
        "infoaccess.ALL": _all_metric(
            records,
            per_sample_results,
            ["infoAccessibilityQA_list", "infoAccessibilityQAs_binary"],
        ),
        "infoaccess.list.accuracy": by_question_type.get("infoAccessibilityQA_list", 0.0),
        "infoaccess.yn.accuracy": by_question_type.get("infoAccessibilityQAs_binary", 0.0),
        "fact.qa.accuracy": by_question_type.get("factQA", 0.0),
    }
    return metrics
