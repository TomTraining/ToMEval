"""HellaSwag prompt(Zellers et al. 2019, arXiv:1905.07830)。

常识句子补全：给一段上下文，从 4 个续写中选最合理的一个。HellaSwag 非 ToM，
作常识推理对照基线。官方未提供 LLM 评测 prompt，这里用社区约定的 Passage/续写四选一，
答案格式用本框架 \\boxed{}。
"""

from __future__ import annotations

from typing import Dict, Optional

from src.evaluation.lang import get_sample_lang
from src.evaluation.prompts import boxed_directive, prompt_type, render_options_block
from src.evaluation.protocols import reasoning_for
from src.evaluation.types import StandardizedSample

BODY = """Passage:
{story}

Question:
{question}

Options:
{options_block}"""


def build_system_prompt(sample, protocol: str, lang: str, prompt_type: str) -> str:
    return boxed_directive(lang, prompt_type, reasoning_for(protocol))


def build_prompt(
    sample: StandardizedSample,
    option_map: Optional[Dict[str, str]],
    include_instruction: bool = True,
) -> str:
    lang = get_sample_lang(sample.get("meta"))
    body = BODY.format(
        story=sample["story"],
        question=sample["question"],
        options_block=render_options_block(option_map or {}),
    )
    if include_instruction:
        body += "\n\n" + boxed_directive(lang, prompt_type(sample["answer"]), reasoning=False)
    return body.rstrip()
