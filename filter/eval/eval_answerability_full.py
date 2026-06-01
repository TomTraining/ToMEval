"""V3 Phase C：answerability 全量单条 judge（仅对 partial + all_failed 跑）。

输入：partial / all_failed 子集的 DataFrame
输出：DataFrame，每条样本一行，含 sample_id / label / reason / answerable

label ∈ {answerable, label_error, ambiguous, contradictory_premise, missing_info}
answerable = True iff label == "answerable"

复用：
  - filter.prompts.ANSWERABILITY_FULL_PROMPT
  - filter.base.load_answer_models（取 strong client）
  - src.llm.llm_utils.extract_json
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from filter.base import load_answer_models
from filter.prompts import ANSWERABILITY_FULL_PROMPT
from filter.utils import resolve_sample_id, stringify_answer_list, write_parquet
from src.llm.content_client import ContentClient
from src.llm.llm_utils import extract_json

logger = logging.getLogger(__name__)


_VALID_LABELS = {"answerable", "label_error", "ambiguous", "contradictory_premise", "missing_info"}


def _build_prompt(row: Any) -> str:
    answer = row.get("answer")
    if not isinstance(answer, dict):
        answer = {"correct_answers": [], "wrong_answers": []}
    ca = answer.get("correct_answers")
    wa = answer.get("wrong_answers")
    return ANSWERABILITY_FULL_PROMPT.format(
        story=str(row.get("story", "") or ""),
        question=str(row.get("question", "") or ""),
        correct_answers=stringify_answer_list(ca if ca is not None else []),
        wrong_answers=stringify_answer_list(wa if wa is not None else []),
    )


def _parse_response(text: Optional[str]) -> Dict[str, Any]:
    if not text:
        return {"label": None, "reason": "empty_response"}
    parsed = extract_json(text)
    if not isinstance(parsed, dict):
        return {"label": None, "reason": "json_parse_failed"}
    label = parsed.get("label")
    reason = str(parsed.get("reason") or "")[:240]
    if label not in _VALID_LABELS:
        return {"label": None, "reason": f"invalid_label:{label}"}
    return {"label": label, "reason": reason}


def run_answerability_on_df(
    df: pd.DataFrame,
    dataset: str,
    strong_client: Optional[ContentClient] = None,
) -> pd.DataFrame:
    """对 df 跑单条 answerability judge。

    Args:
        df: 待评估子集（partial + all_failed）
        dataset: 数据集名称（仅用于日志）
        strong_client: 可选注入；不传则从 config 加载 strong

    Returns:
        DataFrame：[sample_id, label, reason, answerable]
                   行序与 df 完全对齐
    """
    if strong_client is None:
        strong_client = load_answer_models()["strong"]

    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=["sample_id", "label", "reason", "answerable"])

    df = df.reset_index(drop=True)
    sample_ids: List[str] = []
    prompts: List[str] = []
    for idx, row in df.iterrows():
        sample_ids.append(resolve_sample_id(row, idx))
        prompts.append(_build_prompt(row))

    responses = strong_client.batch_generate(prompts, desc=f"answerability[{dataset}]")

    labels: List[Optional[str]] = []
    reasons: List[str] = []
    answerable: List[Optional[bool]] = []
    for resp in responses:
        text = resp.content if (resp is not None and resp.content is not None) else None
        parsed = _parse_response(text)
        labels.append(parsed["label"])
        reasons.append(parsed["reason"])
        answerable.append(True if parsed["label"] == "answerable" else (None if parsed["label"] is None else False))

    out = pd.DataFrame({
        "sample_id": sample_ids,
        "label": labels,
        "reason": reasons,
        "answerable": answerable,
    })

    n_ok = sum(1 for a in answerable if a == True)  # noqa: E712
    n_unans = sum(1 for a in answerable if a == False)  # noqa: E712
    n_err = sum(1 for a in answerable if a is None)
    logger.info(
        f"[answerability] {dataset} n={n} answerable={n_ok} unanswerable={n_unans} parse_error={n_err}"
    )
    return out


def write_answerability_parquet(ans_df: pd.DataFrame, out_path: Path) -> None:
    write_parquet(ans_df, out_path, "answerability")


__all__ = [
    "run_answerability_on_df",
    "write_answerability_parquet",
    "_VALID_LABELS",
]
