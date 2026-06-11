"""Hi-ToM 原论文 prompt(忠实复刻 ying-hui-he/Hi-ToM_dataset 的数据 prompt 字段)。

官方 prompt 结构：
  Read the following story and answer the multiple-choice question. <答案格式说明>
  Story:\n{story}\n\n\nQuestion: {question}\nChoices: A. x, B. y, ...(单行逗号分隔)\n\nNote: <4 条假设>

- Choices 为单行逗号分隔(非多行选项块)。
- 末尾固定追加 4 条假设 Note(标准化时曾丢失，这里补回，逐字复刻)。
- 官方 "Think step-by-step. Provide the answer first, and then explain it." 属答案格式/顺序指令，
  按统一策略移交 system prompt(换成 \\boxed{} 协议)。
"""

from __future__ import annotations

from typing import Dict, Optional

from src.evaluation.prompts import boxed_directive
from src.evaluation.protocols import reasoning_for
from src.evaluation.types import StandardizedSample

HITOM_NOTE = (
    "Note: You should assume the following. "
    "(1) An agent witnesses everything and every movement before exiting a location. "
    "(2) An agent A can infer another agent B's mental state only if A and B have been in the "
    "same location, or have private or public interactions. "
    "(3) Note that every agent tends to lie. What an agent A tells others doesn't affect A's "
    "actual belief. An agent tends to trust an agent that exited the room later than himself. "
    "The exit order is known to all agents. "
    "(4) Agents in private communications know that others won't hear them, but they know that "
    "anyone can hear any public claims."
)

TEMPLATE = """Read the following story and answer the multiple-choice question.
Story:
{story}


Question: {question}
Choices: {choices_inline}

{note}"""


def _inline_choices(option_map: Optional[Dict[str, str]]) -> str:
    return ", ".join(f"{letter}. {text}" for letter, text in (option_map or {}).items())


def build_system_prompt(sample, protocol: str, lang: str, prompt_type: str) -> str:
    return boxed_directive(lang, prompt_type, reasoning_for(protocol))


def build_prompt(
    sample: StandardizedSample,
    option_map: Optional[Dict[str, str]],
    include_instruction: bool = True,
) -> str:
    body = TEMPLATE.format(
        story=sample["story"],
        question=sample["question"],
        choices_inline=_inline_choices(option_map),
        note=HITOM_NOTE,
    )
    if include_instruction:
        body += "\n\n" + boxed_directive("en", "mcq_single", reasoning=False)
    return body.rstrip()
