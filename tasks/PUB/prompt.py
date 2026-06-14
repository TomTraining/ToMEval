"""PUB prompt(Sravanthi et al. 2024, arXiv:2401.07078)。

官方 Multiple Choice Prompting（MCP）：给出语用语境（Context/Question/Response 等，
已包含在 story 中）+ 候选选项，让模型选最合适的一项。答案格式改用本框架 \\boxed{}。
"""

from __future__ import annotations

from typing import Dict, Optional

from src.evaluation.lang import get_sample_lang
from src.evaluation.prompts import boxed_directive, prompt_type, render_options_block
from src.evaluation.protocols import reasoning_for
from src.evaluation.types import StandardizedSample

BODY = """{story}

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
        story=sample["story"].rstrip(),
        question=sample["question"],
        options_block=render_options_block(option_map or {}),
    )
    if include_instruction:
        body += "\n\n" + boxed_directive(lang, prompt_type(sample["answer"]), reasoning=False)
    return body.rstrip()
