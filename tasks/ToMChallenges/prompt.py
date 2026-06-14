"""ToMChallenges prompt(Ma et al. 2023, CoNLL, arXiv:2305.15068)。

基于 Sally-Anne / Smarties 的多格式 ToM 测试。这里取选择题（mc）与开放问答（qa）两种：
mc 为单选，qa 为开放短答。保留官方 Context / Question 结构，答案格式用本框架 \\boxed{}
（开放题直接给答案文本）。
"""

from __future__ import annotations

from typing import Dict, Optional

from src.evaluation.lang import get_sample_lang
from src.evaluation.prompts import boxed_directive, prompt_type, render_options_block
from src.evaluation.protocols import reasoning_for
from src.evaluation.types import StandardizedSample

BODY_MCQ = """Story:
{story}

Question:
{question}

Options:
{options_block}"""

BODY_OPEN = """Story:
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
    if option_map:
        body = BODY_MCQ.format(
            story=sample["story"],
            question=sample["question"],
            options_block=render_options_block(option_map),
        )
    else:
        body = BODY_OPEN.format(story=sample["story"], question=sample["question"])
    if include_instruction:
        body += "\n\n" + boxed_directive(lang, prompt_type(sample["answer"]), reasoning=False)
    return body.rstrip()
