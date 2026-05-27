from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.evaluation.task_metrics import base_metric_payload, count_dict, first_value, meta, rate_dict, update_group


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


def compute_metrics(records: List[Dict[str, Any]], per_sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = base_metric_payload(per_sample_results)
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

    event_acc_values = list(rate_dict(by_event).values())
    document_acc_values = list(rate_dict(by_document).values())
    style_acc_values = list(rate_dict(by_style).values())
    blind_avg = sum(blind_values) / len(blind_values) if blind_values else None

    metrics.update(
        {
            "event_split_acc": sum(event_acc_values) / len(event_acc_values) if event_acc_values else 0.0,
            "document_split_acc": sum(document_acc_values) / len(document_acc_values) if document_acc_values else 0.0,
            "style_split_acc": sum(style_acc_values) / len(style_acc_values) if style_acc_values else 0.0,
            "blind_avg": blind_avg,
            "informed_vs_blind_gap": metrics["accuracy"] - blind_avg if blind_avg is not None else None,
            "event_split_details": rate_dict(by_event),
            "document_split_details": rate_dict(by_document),
            "style_split_details": rate_dict(by_style),
            "event_counts": count_dict(by_event),
            "document_counts": count_dict(by_document),
            "style_counts": count_dict(by_style),
        }
    )
    return metrics
