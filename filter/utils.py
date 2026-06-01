"""filter 共享工具函数。"""

from pathlib import Path
from typing import Any, Dict, List
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def resolve_sample_id(row: Any, fallback_idx: int) -> str:
    """从 row 的 meta.id 提取 sample_id，如果不存在则使用 row_{idx}。

    Args:
        row: DataFrame row 或 dict
        fallback_idx: 回退索引

    Returns:
        sample_id 字符串
    """
    meta = row.get("meta")
    if isinstance(meta, dict):
        sid = meta.get("id")
        if sid:
            return str(sid)
    return f"row_{fallback_idx}"


def row_to_sample(row: Any) -> Dict[str, Any]:
    """将 DataFrame row 转换为标准化的 sample dict。

    Args:
        row: DataFrame row

    Returns:
        dict 包含 story/question/answer 字段
    """
    answer = row.get("answer")
    if not isinstance(answer, dict):
        answer = {"correct_answers": [], "wrong_answers": []}
    correct_raw = answer.get("correct_answers")
    wrong_raw = answer.get("wrong_answers")
    correct = list(correct_raw) if correct_raw is not None else []
    wrong = list(wrong_raw) if wrong_raw is not None else []
    return {
        "story": str(row.get("story", "") or ""),
        "question": str(row.get("question", "") or ""),
        "answer": {"correct_answers": correct, "wrong_answers": wrong},
    }


def is_correct_mcq(prompt_type: str, response: Any, correct_letters: List[str]) -> bool:
    """判断 MCQ 答案是否正确。

    Args:
        prompt_type: "mcq_single" / "mcq_multi" / "open"
        response: LLM 响应对象
        correct_letters: 正确答案的字母列表

    Returns:
        是否正确
    """
    from src.evaluation.prompts import extract_prediction_value

    if response is None or response.content is None:
        return False
    pred = extract_prediction_value(prompt_type, response)
    if prompt_type == "mcq_single":
        return pred in correct_letters
    if prompt_type == "mcq_multi":
        return pred is not None and set(pred) == set(correct_letters)
    # open：宽容文本包含
    if pred is None:
        return False
    pred_norm = str(pred).strip().lower()
    return any(str(c).strip().lower() in pred_norm for c in correct_letters)


def write_parquet(df: pd.DataFrame, out_path: Path, desc: str = "") -> None:
    """写入 parquet 文件的统一接口。

    Args:
        df: 要写入的 DataFrame
        out_path: 输出路径
        desc: 描述（用于日志）
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    prefix = f"[{desc}] " if desc else ""
    logger.info(f"{prefix}wrote {len(df)} rows → {out_path}")


def normalize_answer_dict(answer: Any) -> Dict[str, List[str]]:
    """规范化 answer dict，确保 correct_answers 和 wrong_answers 是 list。

    Args:
        answer: answer 字段（可能是 dict 或其他类型）

    Returns:
        规范化的 dict
    """
    if not isinstance(answer, dict):
        return {"correct_answers": [], "wrong_answers": []}

    ca = answer.get("correct_answers")
    wa = answer.get("wrong_answers")

    return {
        "correct_answers": list(ca) if ca is not None else [],
        "wrong_answers": list(wa) if wa is not None else [],
    }


def stringify_answer_list(values: Any, separator: str = ", ") -> str:
    """将 answer list 转换为字符串。

    Args:
        values: list、tuple 或其他类型
        separator: 分隔符

    Returns:
        字符串表示
    """
    if values is None:
        return ""
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, (list, tuple)):
        return str(values) if values not in (None, "", 0, False) else ""
    if len(values) == 0:
        return ""
    return separator.join(str(v) for v in values)


__all__ = [
    "resolve_sample_id",
    "row_to_sample",
    "is_correct_mcq",
    "write_parquet",
    "normalize_answer_dict",
    "stringify_answer_list",
]
