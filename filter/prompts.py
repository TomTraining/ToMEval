"""V3 数据飞轮 Phase C answerability 判断提示词。

V3 简化设计：
- 只保留 ANSWERABILITY_FULL_PROMPT（Phase C 使用）
- 修复提示词在 repair/repair_prompts.py
- 旧版本的 difficulty/representativeness 评估已废弃
"""

# ── V3 Phase C：answerability 全量单条 judge ──────────────────────────────────
# 对 partial + all_failed 样本判断是否可回答
# 输出：{"label": "answerable|label_error|ambiguous|contradictory_premise|missing_info", "reason": "..."}
# 决策树规则：answerable → 继续 Phase D；其余 4 类 → label=bad, repair_type=unanswerable

ANSWERABILITY_FULL_PROMPT = """\
You are auditing a Theory-of-Mind (ToM) question to decide whether it is logically answerable \
given only the story and question. Do NOT try to answer it; only judge whether the question is well-posed.

Story: {story}
Question: {question}
Correct answer: {correct_answers}
Wrong answers (distractors): {wrong_answers}

Choose EXACTLY ONE label:
  - answerable            : the story uniquely supports the marked correct answer; distractors are clearly wrong.
  - label_error           : the marked correct answer is wrong, or one of the distractors is actually correct.
  - ambiguous             : multiple options are equally defensible given the story.
  - contradictory_premise : the story contains an internal contradiction that blocks any consistent answer.
  - missing_info          : the story does not contain enough information to derive the correct answer.

Return ONLY valid JSON (no markdown fences):
{{
  "label": "<one of: answerable | label_error | ambiguous | contradictory_premise | missing_info>",
  "reason": "<one short sentence>"
}}"""


__all__ = ["ANSWERABILITY_FULL_PROMPT"]
