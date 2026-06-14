"""ToMATO prompt(Shinoda et al. 2025, AAAI, arXiv:2501.08838)。

给一段角色扮演对话，问目标说话者的（高阶）心理状态，4 选 1。官方仅提供 CoT 提示
（"Let's think step by step."），评测主体为对话 + 问题 + 四选项；答案格式用本框架 \\boxed{}。
"""

from __future__ import annotations

from typing import Dict, Optional

from src.evaluation.lang import get_sample_lang
from src.evaluation.prompts import boxed_directive, prompt_type, render_options_block
from src.evaluation.protocols import reasoning_for
from src.evaluation.types import StandardizedSample

BODY = """Conversation:
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
