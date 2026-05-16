from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


PromptType = Literal["open", "mcq_single", "mcq_multi"]


class AnswerBlock(TypedDict):
    correct_answers: List[str]
    wrong_answers: List[str]


class StandardizedSample(TypedDict):
    sample_id: str
    story: str
    question: str
    answer: AnswerBlock
    meta: Dict[str, Any]


class PredictionRecord(TypedDict):
    sample_id: str
    sample_index: int
    repeat: int
    story: str
    question: str
    correct_answers: List[str]
    wrong_answers: List[str]
    prompt_type: PromptType
    prompt: str
    options: Optional[Dict[str, str]]
    correct_letters: List[str]
    wrong_letters: List[str]
    shuffle_seed: int
    meta: Dict[str, Any]
    pred: Dict[str, Any]
    raw_prediction: Any
