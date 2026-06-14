"""FictionalQA prompt(Kirchenbauer et al. 2025, arXiv:2506.05639)。

给一篇虚构文档，问其中的事实细节（开放短答）。FictionalQA 测记忆/知识获取，非 ToM，
作对照集。开放题：要求直接给出答案文本（走 open_judge: f1 判分）。
"""

from __future__ import annotations

from typing import Dict, Optional

from src.evaluation.lang import get_sample_lang
from src.evaluation.prompts import boxed_directive, prompt_type
from src.evaluation.protocols import reasoning_for
from src.evaluation.types import StandardizedSample

BODY = """Document:
{story}

Question:
{question}"""


def build_system_prompt(sample, protocol: str, lang: str, prompt_type: str) -> str:
    return boxed_directive(lang, prompt_type, reasoning_for(protocol))


def build_prompt(
    sample: StandardizedSample,
    option_map: Optional[Dict[str, str]],
    include_instruction: bool = True,
) -> str:
    lang = get_sample_lang(sample.get("meta"))
    body = BODY.format(story=sample["story"], question=sample["question"])
    if include_instruction:
        body += "\n\n" + boxed_directive(lang, prompt_type(sample["answer"]), reasoning=False)
    return body.rstrip()
