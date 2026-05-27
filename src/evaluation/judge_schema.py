from __future__ import annotations

from pydantic import BaseModel, Field


class QAJudgeResult(BaseModel):
    """Judge 输出结果：只需要判断对错"""
    is_correct: bool = Field(description="Whether the model prediction is correct.")
