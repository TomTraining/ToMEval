from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


# mcq_grouped：一个 prompt 内含多道子问题(如 EmoBench EU 的情绪+原因)，
# 由 prepare_samples 预先合并，judge 要求每个子问题各对才算整体对。
PromptType = Literal["open", "mcq_single", "mcq_multi", "mcq_grouped"]


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
    # 协议评测下记录所用协议与 extractor；旧 jsonl 缺失这两字段时按 None/"legacy" 兼容读取。
    protocol: Optional[str]
    extractor: str
