"""F033: 共享 answerability 核心逻辑（题型分类 / 答案字母 / MCQ prompt / 答题判定 / LLM 加载）。

eval_answerability.py 与 run_train_eval.py 都从这里 import，避免重复实现。
prompt 文本与判题语义保持与 F032 之前的 eval_answerability.py / run_train_eval.py 完全一致。
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd

from data_eval.base import load_answer_models
from feedback_synthesis.stage1_load_predictions import (
    _extract_letter as extract_letter,
    _extract_letters_multi as extract_letters_multi,
)
from src.llm.content_client import ContentClient


MCQ_SINGLE_PROMPT = """\
Read the following story and question, then answer with ONLY the letter of the correct option (A, B, C, or D). Do not explain.

Story:
{story}

Question:
{question}

Answer (single letter only):"""

MCQ_MULTI_PROMPT = """\
Read the following story and question, then answer with ONLY the letters of all correct options, separated by commas (e.g. "A, C"). Do not explain.

Story:
{story}

Question:
{question}

Answer (letters only, comma-separated):"""


def _is_listlike(x: Any) -> bool:
    return isinstance(x, (list, tuple, np.ndarray, pd.Series))


def _to_list(x: Any) -> list:
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)


def classify_prompt_type(row: Any) -> str:
    """从 row 推断题型：mcq_single / mcq_multi / open。

    与 eval_answerability.py 旧 _classify_prompt_type 语义保持一致：
      - answer 非 dict → mcq_single
      - wrong_answers 空 → open
      - wrong_answers 长度 ≥3 → mcq_multi
      - 其他 → mcq_single
    """
    answer = row.get("answer")
    if not isinstance(answer, dict):
        return "mcq_single"
    wrong = answer.get("wrong_answers", [])
    wrong_list = _to_list(wrong) if _is_listlike(wrong) else []
    if len(wrong_list) == 0:
        return "open"
    if len(wrong_list) >= 3:
        return "mcq_multi"
    return "mcq_single"


def get_correct_letters(row: Any) -> List[str]:
    """从 row 提取正确答案字母列表（mcq 题）。

    与 run_train_eval.py 旧 _get_correct_letters 语义保持一致：
    按 correct + wrong 拼接顺序赋字母 A/B/C/...，找正确文本对应的字母。
    """
    answer = row.get("answer") if isinstance(row.get("answer"), dict) else {}
    correct_texts = answer.get("correct_answers", [])
    wrong_texts = answer.get("wrong_answers", [])
    correct_list = _to_list(correct_texts) if _is_listlike(correct_texts) else [correct_texts]
    wrong_list = _to_list(wrong_texts) if _is_listlike(wrong_texts) else [wrong_texts]
    all_texts = correct_list + wrong_list
    letter_map = {str(t).strip(): chr(ord("A") + i) for i, t in enumerate(all_texts)}
    result = [letter_map[str(ct).strip()] for ct in correct_list if str(ct).strip() in letter_map]
    return result if result else ["A"]


def build_answer_prompt(prompt_type: str, row: Any) -> str:
    story = str(row.get("story", ""))
    question = str(row.get("question", ""))
    if prompt_type == "mcq_multi":
        return MCQ_MULTI_PROMPT.format(story=story, question=question)
    return MCQ_SINGLE_PROMPT.format(story=story, question=question)


def check_correct_single(predicted: str, correct_letters: List[str]) -> bool:
    return extract_letter(predicted) in [c.upper() for c in correct_letters]


def check_correct_multi(predicted: str, correct_letters: List[str]) -> bool:
    return extract_letters_multi(predicted) == {c.upper() for c in correct_letters}


def check_correct(prompt_type: str, predicted: str, correct_letters: List[str]) -> bool:
    if prompt_type == "mcq_multi":
        return check_correct_multi(predicted, correct_letters)
    return check_correct_single(predicted, correct_letters)


def load_answer_clients() -> List[ContentClient]:
    """F039：保留有序 [strong, simple] 列表 API 给历史调用方；底层改读 dict schema。"""
    models = load_answer_models()
    return [models["strong"], models["simple"]]


__all__ = [
    "MCQ_SINGLE_PROMPT",
    "MCQ_MULTI_PROMPT",
    "classify_prompt_type",
    "get_correct_letters",
    "build_answer_prompt",
    "check_correct_single",
    "check_correct_multi",
    "check_correct",
    "load_answer_clients",
    "extract_letter",
    "extract_letters_multi",
]
