"""ToMato 分组指标：meta.dimension 是有序的三槽维度列表（slot1/slot2/slot3），
按 slot1 → slot2 → slot3 天然嵌套为二→三→四级维度。缺失槽位用占位符标记。
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.evaluation.task_metrics import hierarchical_metrics, meta, value_list


def _slot(index: int):
    """取 meta.dimension 第 index 个槽位的分组键；缺失用占位符。"""
    placeholder = "__none__" if index == 2 else "__missing__"

    def key_fn(record: Dict[str, Any]) -> List[str]:
        dims = value_list(meta(record).get("dimension"))
        return [dims[index] if len(dims) > index else placeholder]

    return key_fn


def compute_metrics(records: List[Dict[str, Any]], per_sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return hierarchical_metrics(
        records,
        per_sample_results,
        [
            # slot1 → slot2 → slot3 天然嵌套。
            ("dimension_1", _slot(0), [
                ("dimension_2", _slot(1), [
                    ("dimension_3", _slot(2)),
                ]),
            ]),
        ],
    )
