"""V3 Phase E：批量修复 pipeline。

输入：DataFrame（standardized 样本）+ labels.parquet（带 repair_type 字段）
输出：repaired.parquet —— 仅包含成功修复的样本，meta.history 追加修复事件

每条样本：
  1) 按 repair_type 选 prompt 模板
  2) 调 StructureClient + RepairedSample schema 强约束输出
  3) 解析失败 → 跳过（不进 repaired，但记录到 meta.history 的 unfixable）
  4) 解析成功 → 用新 story/question/correct/wrong 覆盖原字段，meta.id 保持，meta.source_iter += "_repair"

判定规则：
  label == easy / shortcut / bad → repair_type = easy / shortcut / unanswerable
  label == hard / medium → 不进 repair（被上游过滤掉）
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field

from src.filter.base import load_judge_client
from src.filter.repair.repair_prompts import build_repair_prompt
from src.filter.utils import resolve_sample_id, write_parquet
from src.evaluation.lang import get_sample_lang
from src.llm.structure_client import StructureClient

logger = logging.getLogger(__name__)


class RepairedSample(BaseModel):
    """V3 修复 prompt 强约束输出 schema。"""
    story: str = Field(description="Self-contained story")
    question: str = Field(description="Single question")
    correct_answers: List[str] = Field(description="Plain text correct answers, count matches original")
    wrong_answers: List[str] = Field(description="Plain text distractors, count matches original")


def _stringify_answers(values: Any) -> List[str]:
    """将 answer list 转换为字符串列表（用于 repair prompt）。"""
    if not isinstance(values, (list, tuple)):
        return []
    return [str(v) for v in values if v is not None]


def repair_samples(
    df: pd.DataFrame,
    labels_df: pd.DataFrame,
    dataset: str,
    iter_n: int,
    repair_client: Optional[StructureClient] = None,
    enabled_types: Optional[List[str]] = None,
) -> pd.DataFrame:
    """对 df 中标 easy/shortcut/bad 的样本批量调修复 prompt。

    Args:
        df: 当前轮 standardized 样本（含未修复样本，索引位与 labels_df 对齐）
        labels_df: 当前轮 labels.parquet（含 sample_id, label, repair_type, answerability 派生字段）
        dataset: 数据集名
        iter_n: 当前轮号（写入 meta.history）
        repair_client: StructureClient；不传则 load strong
        enabled_types: 启用的修复类型；None=全部三类

    Returns:
        DataFrame：修复成功的样本（standardized schema），meta.history 追加事件
    """
    if repair_client is None:
        repair_client = load_judge_client("strong")
    if enabled_types is None:
        enabled_types = ["unanswerable", "easy", "shortcut"]

    df = df.reset_index(drop=True)
    labels_df = labels_df.reset_index(drop=True)
    if len(df) != len(labels_df):
        raise ValueError(
            f"repair_samples: df ({len(df)}) 与 labels_df ({len(labels_df)}) 行数不匹配"
        )

    # 1) 选出待修复行
    target_indices: List[int] = []
    repair_types: List[str] = []
    for i, row in labels_df.iterrows():
        rt = row.get("repair_type")
        if rt in enabled_types:
            target_indices.append(i)
            repair_types.append(rt)

    if not target_indices:
        logger.info(f"[repair] {dataset} iter={iter_n} no rows to repair")
        return pd.DataFrame(columns=list(df.columns))

    # 2) 构造 prompts
    prompts: List[str] = []
    sample_id_per_target: List[str] = []
    for tidx, rt in zip(target_indices, repair_types):
        row = df.iloc[tidx]
        answer = row.get("answer") if isinstance(row.get("answer"), dict) else {}
        correct = _stringify_answers(answer.get("correct_answers"))
        wrong = _stringify_answers(answer.get("wrong_answers"))
        label_row = labels_df.iloc[tidx]
        prompts.append(build_repair_prompt(
            repair_type=rt,
            dataset=dataset,
            story=str(row.get("story", "") or ""),
            question=str(row.get("question", "") or ""),
            correct_answers=correct,
            wrong_answers=wrong,
            failure_label=str(label_row.get("answerability_label") or ""),
            failure_reason=str(label_row.get("answerability_reason") or ""),
            # 修复内容语言跟随原样本语言（meta.lang / meta.language）。
            lang=get_sample_lang(row.get("meta")),
        ))
        sample_id_per_target.append(resolve_sample_id(row, tidx))

    # 3) 批量调强模型 + 强约束 schema
    responses = repair_client.batch_generate_structure(
        prompts, RepairedSample, mode="create", desc=f"repair[{dataset}] iter{iter_n}"
    )

    # 4) 组装新 DataFrame
    repaired_rows: List[Dict[str, Any]] = []
    n_ok = 0
    n_fail = 0
    for tidx, rt, sid, resp in zip(target_indices, repair_types, sample_id_per_target, responses):
        repaired = resp.content if (resp is not None) else None
        if not isinstance(repaired, RepairedSample):
            n_fail += 1
            continue
        if not repaired.correct_answers:
            n_fail += 1
            continue

        original = df.iloc[tidx].to_dict()
        meta = original.get("meta") if isinstance(original.get("meta"), dict) else {}
        meta = dict(meta)
        history = list(meta.get("history") if meta.get("history") is not None else [])
        history.append({
            "iter": iter_n,
            "repair_type": rt,
        })
        meta["history"] = history
        meta["source_iter"] = f"eval_iter{iter_n}_repair"
        meta.setdefault("id", sid)

        repaired_rows.append({
            **original,
            "story": repaired.story,
            "question": repaired.question,
            "answer": {
                "correct_answers": list(repaired.correct_answers),
                "wrong_answers": list(repaired.wrong_answers),
            },
            "meta": meta,
        })
        n_ok += 1

    logger.info(
        f"[repair] {dataset} iter={iter_n} target={len(target_indices)} "
        f"ok={n_ok} fail={n_fail}"
    )

    return pd.DataFrame(repaired_rows)


def write_repaired_parquet(repaired_df: pd.DataFrame, out_path: Path) -> None:
    write_parquet(repaired_df, out_path, "repair")


__all__ = [
    "RepairedSample",
    "repair_samples",
    "write_repaired_parquet",
]
