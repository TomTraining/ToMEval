from __future__ import annotations

from typing import Any, Dict, List

from src.evaluation.task_metrics import (
    add_dimension,
    by_meta,
    first_value,
    hierarchical_metrics,
    make_split,
    meta,
    safe_div,
)


def _step_kind(record: Dict[str, Any]) -> str:
    """把 meta.step 归一成 belief_update / belief_matching 两类（与 BU/BM 口径一致）。"""
    step = first_value(meta(record).get("step"), default="")
    is_update = step.lower() in {"time_t+1", "time_t1", "t+1", "time_t_1"}
    return "belief_update" if is_update else "belief_matching"


def compute_metrics(records: List[Dict[str, Any]], per_sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = hierarchical_metrics(
        records,
        per_sample_results,
        [
            ("step", lambda r: [_step_kind(r)]),
            # modus：推理式（ponens 肯定前件 / tollens 否定后件）；
            # types_of_relation：条件规则的关系类型（事件→事件 / 事件→心理状态）。
            ("modus", by_meta("modus")),
            ("types_of_relation", by_meta("types_of_relation")),
        ],
    )

    bu_correct = bu_total = bm_correct = bm_total = 0
    for record, result in zip(records, per_sample_results):
        if _step_kind(record) == "belief_update":
            bu_total += 1
            bu_correct += int(result["is_correct"])
        else:
            bm_total += 1
            bm_correct += int(result["is_correct"])

    bu_acc = safe_div(bu_correct, bu_total)
    bm_acc = safe_div(bm_correct, bm_total)
    # BU/BM/BREU 是 Belief-R 官方头条指标：BREU 为整体（overall），BU/BM 为其两个分项 split。
    add_dimension(metrics, "belief_reasoning", {
        "BREU": make_split((bu_acc + bm_acc) / 2, bu_total + bm_total),
        "BU-Acc": make_split(bu_acc, bu_total),
        "BM-Acc": make_split(bm_acc, bm_total),
    })
    return metrics
