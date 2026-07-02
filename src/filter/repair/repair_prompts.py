"""V3 Phase E：三种修复 prompt 模板（unanswerable / easy / shortcut）。

输出契约（强模型 StructureClient）：
  - story / question / correct_answers / wrong_answers 完整新样本
  - 保持原数据集风格（角色名/题型不变）

复用：
  - feedback.prompts.DATASET_SKILL_REGISTRY 提供数据集背景
"""

from __future__ import annotations

from typing import Dict, List

from src.feedback.prompts import DATASET_SKILL_REGISTRY


_SHARED_OUTPUT_SCHEMA = """\
Return ONLY valid JSON (no markdown fences) matching:
{{
  "story": "<full new story, self-contained>",
  "question": "<single question, exactly one '?' if interrogative>",
  "correct_answers": ["<correct option text>", ...],   # 1 element for single-choice; >1 for multi-choice
  "wrong_answers": ["<distractor text>", ...]          # keep the same count as the original sample
}}

Constraints:
  - {language_constraint}
  - Preserve the dataset's question style (length, agent names, register).
  - Keep correct_answers / wrong_answers as plain text (NOT letters).
  - Keep wrong_answers count identical to the original sample.
  - Do not include any commentary outside the JSON."""

# 修复内容的语言必须跟随原样本语言；JSON key 与结构约束保持英文。
_LANGUAGE_CONSTRAINTS = {
    "en": "Write all generated text (story, question, answers) in English, matching the original sample.",
    "zh": "所有生成的文本（story、question、answers 的内容）必须使用简体中文，与原样本语言一致。",
}


REPAIR_UNANSWERABLE_PROMPT = """\
You are repairing a Theory-of-Mind (ToM) question that an auditor flagged as NOT logically answerable.

Dataset: {dataset}
Dataset focus: {dataset_focus}

Failure label: {failure_label}
Auditor reason: {failure_reason}

Original sample (broken):
  Story: {story}
  Question: {question}
  Marked correct: {correct_answers}
  Marked wrong:   {wrong_answers}

Repair goal:
  - Fix the issue indicated by the failure label.
  - The new story MUST uniquely support exactly the marked correct answer(s).
  - All wrong answers must be unambiguously wrong given the new story.
  - Resolve any contradictions, ambiguity, missing info, or label errors.
  - Keep the ToM ability tested by the original question.

""" + _SHARED_OUTPUT_SCHEMA


REPAIR_EASY_PROMPT = """\
You are repairing a Theory-of-Mind (ToM) question that a small model answered correctly EVERY time \
(pass@k = k). It is too easy.

Dataset: {dataset}
Dataset focus: {dataset_focus}

Original sample (too easy):
  Story: {story}
  Question: {question}
  Marked correct: {correct_answers}
  Marked wrong:   {wrong_answers}

Repair goal:
  - Increase ToM reasoning depth: introduce one extra belief layer, asymmetric information, or a state change \
the agent did NOT witness.
  - The story must remain self-contained and the question must still have exactly one correct answer.
  - Keep the same agents and the same general scenario; only deepen the ToM demand.
  - Distractors should now be more plausible (close-but-wrong reasoning paths), not random fillers.

""" + _SHARED_OUTPUT_SCHEMA


REPAIR_SHORTCUT_PROMPT = """\
You are repairing a Theory-of-Mind (ToM) question that contains a SHORTCUT — the model can pick the right \
answer without genuinely reading the story (e.g. the answer is hinted in the question wording, the wrong \
options are obviously incoherent, or removing the story leaves the answer guessable from the options alone).

Dataset: {dataset}
Dataset focus: {dataset_focus}

Original sample (has shortcut):
  Story: {story}
  Question: {question}
  Marked correct: {correct_answers}
  Marked wrong:   {wrong_answers}

Repair goal:
  - Rewrite the story so the question MUST depend on a story-specific fact to be answerable.
  - Make distractors story-coherent: each distractor must be a fact that the story COULD have stated, \
and must remain wrong under the rewritten story.
  - Remove any cue in the question wording that gives the answer away.
  - Without the new story, the question must be unanswerable; with the new story, the answer must be unique.

""" + _SHARED_OUTPUT_SCHEMA


REPAIR_PROMPTS_BY_TYPE: Dict[str, str] = {
    "unanswerable": REPAIR_UNANSWERABLE_PROMPT,
    "easy": REPAIR_EASY_PROMPT,
    "shortcut": REPAIR_SHORTCUT_PROMPT,
}


def get_dataset_focus(dataset: str) -> str:
    return DATASET_SKILL_REGISTRY.get(dataset, "general Theory-of-Mind reasoning")


def build_repair_prompt(
    repair_type: str,
    dataset: str,
    story: str,
    question: str,
    correct_answers: List[str],
    wrong_answers: List[str],
    failure_label: str = "",
    failure_reason: str = "",
    lang: str = "en",
) -> str:
    template = REPAIR_PROMPTS_BY_TYPE.get(repair_type)
    if template is None:
        raise ValueError(f"unknown repair_type: {repair_type}")
    language_constraint = _LANGUAGE_CONSTRAINTS.get(lang, _LANGUAGE_CONSTRAINTS["en"])
    return template.format(
        dataset=dataset,
        dataset_focus=get_dataset_focus(dataset),
        story=story,
        question=question,
        correct_answers=", ".join(correct_answers),
        wrong_answers=", ".join(wrong_answers),
        failure_label=failure_label or "(unspecified)",
        failure_reason=failure_reason or "(unspecified)",
        language_constraint=language_constraint,
    )


__all__ = [
    "REPAIR_PROMPTS_BY_TYPE",
    "REPAIR_UNANSWERABLE_PROMPT",
    "REPAIR_EASY_PROMPT",
    "REPAIR_SHORTCUT_PROMPT",
    "build_repair_prompt",
    "get_dataset_focus",
]
