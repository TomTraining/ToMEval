"""data_eval prompt 注册表（三个评估维度占位）。"""

from typing import Dict

# 复用 feedback_synthesis 的维度技能描述，避免重复定义
from feedback_synthesis.prompts import DATASET_SKILL_REGISTRY  # noqa: F401

# ── 难度评估 prompt（F018）────────────────────────────────────────────────────
# 评分标准：
#   1-2分：字面推断，无需 ToM（commonsense/surface cue 即可）
#   3分：基本信念/情感推理（单步一阶 ToM）
#   4-5分：递归信念/多步/高阶 ToM
_DIFFICULTY_SYSTEM = (
    "You are an expert in Theory of Mind (ToM) cognitive science. "
    "Your task is to evaluate the difficulty of a synthesized question "
    "from the perspective of ToM reasoning depth."
)

_DIFFICULTY_SCHEMA = """\
Return ONLY valid JSON (no markdown fences):
{{
  "score": <integer 0-5>,
  "rationale": "<one sentence explaining why this score>",
  "is_training_useful": <true|false>
}}

Scoring rubric:
  0 — Malformed / not a valid ToM question (e.g. story missing, question unanswerable).
  1 — No ToM needed: answerable by surface text or commonsense alone.
  2 — Minimal ToM: single, obvious agent belief with no state change.
  3 — Basic ToM: one-step first-order belief or emotion attribution.
  4 — Intermediate ToM: multi-step or second-order belief tracking.
  5 — Advanced ToM: recursive belief chains (3+ orders), asymmetric information, or complex social inference.
is_training_useful: true if score >= 3 (meaningful ToM signal for training)."""

DIFFICULTY_PROMPTS: Dict[str, str] = {
    "BigToM": (
        _DIFFICULTY_SYSTEM + "\n\n"
        "Dataset context: BigToM tests first-order false belief vs true belief. "
        "Agent A's belief depends on whether they witnessed a state change.\n\n"
        "Question to evaluate:\n"
        "Story: {story}\n"
        "Question: {question}\n"
        "Correct answer: {correct_answers}\n\n"
        + _DIFFICULTY_SCHEMA
    ),
    "EmoBench": (
        _DIFFICULTY_SYSTEM + "\n\n"
        "Dataset context: EmoBench tests emotion attribution — "
        "identify the specific emotion an agent feels given a situation and their goal.\n\n"
        "Question to evaluate:\n"
        "Story: {story}\n"
        "Question: {question}\n"
        "Correct answer: {correct_answers}\n\n"
        + _DIFFICULTY_SCHEMA
    ),
    "FanToM": (
        _DIFFICULTY_SYSTEM + "\n\n"
        "Dataset context: FanToM tests belief tracking in multi-party conversations — "
        "who knows what after partial information sharing across participants.\n\n"
        "Question to evaluate:\n"
        "Story: {story}\n"
        "Question: {question}\n"
        "Correct answer: {correct_answers}\n\n"
        + _DIFFICULTY_SCHEMA
    ),
    "HiToM": (
        _DIFFICULTY_SYSTEM + "\n\n"
        "Dataset context: HiToM tests higher-order recursive belief (order 1-4) — "
        "A thinks B thinks C thinks D thinks... across 5 agents with partial observations.\n\n"
        "Question to evaluate:\n"
        "Story: {story}\n"
        "Question: {question}\n"
        "Correct answer: {correct_answers}\n\n"
        + _DIFFICULTY_SCHEMA
    ),
    "SimpleToM": (
        _DIFFICULTY_SYSTEM + "\n\n"
        "Dataset context: SimpleToM tests social behavior prediction and moral judgment "
        "in simple everyday scenarios with 1-2 agents.\n\n"
        "Question to evaluate:\n"
        "Story: {story}\n"
        "Question: {question}\n"
        "Correct answer: {correct_answers}\n\n"
        + _DIFFICULTY_SCHEMA
    ),
    "SocialIQA": (
        _DIFFICULTY_SYSTEM + "\n\n"
        "Dataset context: SocialIQA tests commonsense social reasoning — "
        "what people do/want/feel/react in everyday situations. NOT belief-tracking.\n\n"
        "Question to evaluate:\n"
        "Story: {story}\n"
        "Question: {question}\n"
        "Correct answer: {correct_answers}\n\n"
        + _DIFFICULTY_SCHEMA
    ),
    "ToMBench": (
        _DIFFICULTY_SYSTEM + "\n\n"
        "Dataset context: ToMBench tests multi-dimensional ToM — "
        "false beliefs, intentions, non-literal communication (faux pas/irony/white lies), "
        "second-order beliefs, knowledge attribution.\n\n"
        "Question to evaluate:\n"
        "Story: {story}\n"
        "Question: {question}\n"
        "Correct answer: {correct_answers}\n\n"
        + _DIFFICULTY_SCHEMA
    ),
}

# ── 可回答性评估 prompt（F019）────────────────────────────────────────────────
_ANSWERABILITY_SYSTEM = (
    "You are an expert in Theory of Mind (ToM) cognitive science and question design. "
    "Your task is to check whether a synthesized question is logically answerable — "
    "meaning the story provides sufficient and consistent information to reach the correct answer."
)

_ANSWERABILITY_SCHEMA = """\
Return ONLY valid JSON (no markdown fences):
{{
  "is_answerable": <true|false>,
  "issues": ["<issue description>", ...],
  "rationale": "<one sentence explaining your verdict>"
}}

Check ALL of the following:
1. Premise consistency: does the story contain any internal contradictions?
2. Answer uniqueness: is there exactly one correct answer supported by the story, \
with wrong answers clearly distinguishable?
3. No circular reasoning / undefined referents: are all agents and objects \
unambiguously defined and traceable?
4. Question-story alignment: does the question ask about information actually \
present or inferable from the story?
If all checks pass, set is_answerable=true and issues=[].
If any check fails, set is_answerable=false and list each issue."""

ANSWERABILITY_PROMPTS: Dict[str, str] = {
    "BigToM": (
        _ANSWERABILITY_SYSTEM + "\n\n"
        "Dataset context: BigToM tests first-order false belief vs true belief. "
        "The agent's belief depends on whether they witnessed a state change. "
        "Verify the witnessing condition in the story is unambiguous and that "
        "the correct answer follows strictly from it.\n\n"
        "Question to check:\n"
        "Story: {story}\n"
        "Question: {question}\n"
        "Correct answer: {correct_answers}\n"
        "Wrong answers: {wrong_answers}\n\n"
        + _ANSWERABILITY_SCHEMA
    ),
    "EmoBench": (
        _ANSWERABILITY_SYSTEM + "\n\n"
        "Dataset context: EmoBench tests emotion attribution — "
        "identify the specific emotion an agent feels given a situation and their goal. "
        "Check that the story clearly states the agent's goal and the triggering event, "
        "and that only one emotion from the options is the correct response.\n\n"
        "Question to check:\n"
        "Story: {story}\n"
        "Question: {question}\n"
        "Correct answer: {correct_answers}\n"
        "Wrong answers: {wrong_answers}\n\n"
        + _ANSWERABILITY_SCHEMA
    ),
    "FanToM": (
        _ANSWERABILITY_SYSTEM + "\n\n"
        "Dataset context: FanToM tests belief tracking in multi-party conversations. "
        "Participants leave and re-enter; who knows what depends strictly on who was present. "
        "Verify that the knowledge state derivable from the conversation matches "
        "the correct answer, and that wrong answers reflect plausible but incorrect beliefs.\n\n"
        "Question to check:\n"
        "Story: {story}\n"
        "Question: {question}\n"
        "Correct answer: {correct_answers}\n"
        "Wrong answers: {wrong_answers}\n\n"
        + _ANSWERABILITY_SCHEMA
    ),
    "HiToM": (
        _ANSWERABILITY_SYSTEM + "\n\n"
        "Dataset context: HiToM tests higher-order recursive belief (order 1-4) "
        "across up to 5 agents with partial observations. "
        "Check that each agent's observation record in the story is explicit and sufficient "
        "to derive the recursive belief chain required by the question.\n\n"
        "Question to check:\n"
        "Story: {story}\n"
        "Question: {question}\n"
        "Correct answer: {correct_answers}\n"
        "Wrong answers: {wrong_answers}\n\n"
        + _ANSWERABILITY_SCHEMA
    ),
    "SimpleToM": (
        _ANSWERABILITY_SYSTEM + "\n\n"
        "Dataset context: SimpleToM tests social behavior prediction and moral judgment "
        "in simple everyday scenarios with 1-2 agents. "
        "Check that the scenario provides enough social context to uniquely determine "
        "the predicted behavior or moral judgment.\n\n"
        "Question to check:\n"
        "Story: {story}\n"
        "Question: {question}\n"
        "Correct answer: {correct_answers}\n"
        "Wrong answers: {wrong_answers}\n\n"
        + _ANSWERABILITY_SCHEMA
    ),
    "SocialIQA": (
        _ANSWERABILITY_SYSTEM + "\n\n"
        "Dataset context: SocialIQA tests commonsense social reasoning — "
        "what people do/want/feel/react in everyday situations. "
        "Check that the correct answer is the commonsense-best response, "
        "not just one of several plausible answers.\n\n"
        "Question to check:\n"
        "Story: {story}\n"
        "Question: {question}\n"
        "Correct answer: {correct_answers}\n"
        "Wrong answers: {wrong_answers}\n\n"
        + _ANSWERABILITY_SCHEMA
    ),
    "ToMBench": (
        _ANSWERABILITY_SYSTEM + "\n\n"
        "Dataset context: ToMBench tests multi-dimensional ToM — "
        "false beliefs, intentions, non-literal communication (faux pas/irony/white lies), "
        "second-order beliefs, knowledge attribution. "
        "Check that the story contains the specific pragmatic or belief cues "
        "required to answer the question, and that wrong answers are genuinely wrong.\n\n"
        "Question to check:\n"
        "Story: {story}\n"
        "Question: {question}\n"
        "Correct answer: {correct_answers}\n"
        "Wrong answers: {wrong_answers}\n\n"
        + _ANSWERABILITY_SCHEMA
    ),
}

# ── F036 stage C：answerability 质量打分（5 标签）──────────────────────────────
ANSWERABILITY_QUALITY_PROMPT = """\
You are auditing a Theory-of-Mind (ToM) multiple-choice / open question that \
multiple LLM answerers consistently failed. Decide WHY they failed.

Story: {story}
Question: {question}
Correct answer: {correct_answers}
Wrong answers (distractors): {wrong_answers}

Choose EXACTLY ONE label that best explains the failure:
  - truly_hard            : the question is well-formed but genuinely requires deep ToM; failures reflect difficulty, not data quality.
  - label_error           : the marked correct answer is wrong / one of the distractors is actually correct.
  - ambiguous             : multiple options are equally defensible given the story.
  - contradictory_premise : the story contains an internal contradiction that blocks any consistent answer.
  - missing_info          : the story does not contain enough information to derive the correct answer.

Return ONLY valid JSON (no markdown fences):
{{
  "score": <integer 0-5>,
  "label": "<one of: truly_hard | label_error | ambiguous | contradictory_premise | missing_info>",
  "reason": "<one short sentence>"
}}

score rubric (informational, does NOT affect downstream score):
  5 = textbook-clean ToM item; 0 = unsalvageable. truly_hard items typically score 4-5; the four problem labels typically score 0-2."""


# ── 代表性评估 prompt（F020）─────────────────────────────────────────────────
_REPRESENTATIVENESS_SYSTEM = (
    "You are an expert in Theory of Mind (ToM) cognitive science. "
    "Your task is to verify that a synthesized question genuinely exercises "
    "the specific ToM ability dimension claimed in its metadata."
)

_REPRESENTATIVENESS_SCHEMA = """\
Return ONLY valid JSON (no markdown fences):
{{
  "score": <integer 0-5>,
  "dim": "<your inferred primary dimension this question actually exercises, or null>",
  "reason": "<one short sentence explaining the score>"
}}

Score rubric (0-5, integer):
  0 — Malformed / not answerable; does not test the claimed dimension at all.
  1 — Largely off-target: surface text or general commonsense suffices, the claimed ToM dimension is barely exercised.
  2 — Weakly aligned: the question touches the claimed dimension but can be solved by a shallow shortcut.
  3 — Aligned: answering the question correctly requires the claimed dimension at a basic level.
  4 — Strongly aligned: the claimed dimension is essential and exercised non-trivially.
  5 — Textbook representative: the question is a clean, focused probe of exactly the claimed dimension.

`dim` is YOUR inference of which ToM ability the question primarily tests \
(e.g. false_belief / second_order_belief / emotion_attribution / faux_pas / commonsense_social). \
Use null only if the question is malformed."""

REPRESENTATIVENESS_PROMPTS: Dict[str, str] = {
    "BigToM": (
        _REPRESENTATIVENESS_SYSTEM + "\n\n"
        "Dataset context: " + DATASET_SKILL_REGISTRY["BigToM"] + "\n"
        "Dimensions: condition_type (e.g. true_belief / false_belief) and "
        "dimension (the specific perceptual/cognitive condition tested).\n\n"
        "Question to evaluate:\n"
        "Story: {story}\n"
        "Question: {question}\n"
        "Correct answer: {correct_answers}\n"
        "Claimed condition_type: {condition_type}\n"
        "Claimed dimension: {dimension}\n\n"
        + _REPRESENTATIVENESS_SCHEMA
    ),
    "EmoBench": (
        _REPRESENTATIVENESS_SYSTEM + "\n\n"
        "Dataset context: " + DATASET_SKILL_REGISTRY["EmoBench"] + "\n"
        "Dimensions: subset (e.g. cause/reaction/desire), "
        "question_subtype (specific emotion category), language (en/zh).\n\n"
        "Question to evaluate:\n"
        "Story: {story}\n"
        "Question: {question}\n"
        "Correct answer: {correct_answers}\n"
        "Claimed subset: {subset}\n"
        "Claimed question_subtype: {question_subtype}\n\n"
        + _REPRESENTATIVENESS_SCHEMA
    ),
    "FanToM": (
        _REPRESENTATIVENESS_SYSTEM + "\n\n"
        "Dataset context: " + DATASET_SKILL_REGISTRY["FanToM"] + "\n"
        "Dimensions: question_type (e.g. belief/knowledge/memory) and "
        "order (0/1/2 — depth of belief nesting).\n\n"
        "Question to evaluate:\n"
        "Story: {story}\n"
        "Question: {question}\n"
        "Correct answer: {correct_answers}\n"
        "Claimed question_type: {question_type}\n"
        "Claimed order: {order}\n\n"
        + _REPRESENTATIVENESS_SCHEMA
    ),
    "HiToM": (
        _REPRESENTATIVENESS_SYSTEM + "\n\n"
        "Dataset context: " + DATASET_SKILL_REGISTRY["HiToM"] + "\n"
        "Dimension: order (0-5 — the depth of recursive belief chain required).\n\n"
        "Question to evaluate:\n"
        "Story: {story}\n"
        "Question: {question}\n"
        "Correct answer: {correct_answers}\n"
        "Claimed order: {order}\n\n"
        + _REPRESENTATIVENESS_SCHEMA
    ),
    "SimpleToM": (
        _REPRESENTATIVENESS_SYSTEM + "\n\n"
        "Dataset context: " + DATASET_SKILL_REGISTRY["SimpleToM"] + "\n"
        "Dimensions: dataset_source (origin), dimension (behavior/knowledge/judgment), "
        "difficulty (easy/medium/hard).\n\n"
        "Question to evaluate:\n"
        "Story: {story}\n"
        "Question: {question}\n"
        "Correct answer: {correct_answers}\n"
        "Claimed dimension: {dimension}\n"
        "Claimed difficulty: {difficulty}\n\n"
        + _REPRESENTATIVENESS_SCHEMA
    ),
    "SocialIQA": (
        _REPRESENTATIVENESS_SYSTEM + "\n\n"
        "Dataset context: " + DATASET_SKILL_REGISTRY["SocialIQA"] + "\n"
        "Dimension: dimension (commonsense relation type, e.g. xWant/xNeed/xReact/xEffect/xAttr).\n\n"
        "Question to evaluate:\n"
        "Story: {story}\n"
        "Question: {question}\n"
        "Correct answer: {correct_answers}\n"
        "Claimed dimension: {dimension}\n\n"
        + _REPRESENTATIVENESS_SCHEMA
    ),
    "ToMBench": (
        _REPRESENTATIVENESS_SYSTEM + "\n\n"
        "Dataset context: " + DATASET_SKILL_REGISTRY["ToMBench"] + "\n"
        "Dimension: ability (specific ToM sub-task, e.g. false_belief/faux_pas/second_order_belief).\n\n"
        "Question to evaluate:\n"
        "Story: {story}\n"
        "Question: {question}\n"
        "Correct answer: {correct_answers}\n"
        "Claimed ability: {ability}\n\n"
        + _REPRESENTATIVENESS_SCHEMA
    ),
}
