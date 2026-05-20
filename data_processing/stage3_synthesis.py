"""
Stage 3：从维度诊断报告合成新样本

读取 Stage 2 输出的 dimension_reports.jsonl，按报告生成新题目，
保存为与 train_data parquet 一致的格式（小写 story/question/answer/meta）。
"""

import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

from pydantic import BaseModel, Field
from tqdm import tqdm

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import runner
from .prompts import (
    build_stage2_generation_from_report_prompt,
    SYNTHESIS_FORMAT_REGISTRY,
    DATASET_SKILL_REGISTRY,
)
from .stage2_diagnosis import _extract_json, _batch_generate_json

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ============ 数据合成Schema ============

# ---- ToMBench ----
class ToMBenchQuestionFlat(BaseModel):
    story: str = Field(description="Complete story text as a plain string")
    question: str = Field(
        description="Question text with all 4 answer options embedded inline, "
                    "e.g. 'What does X think? A. ... B. ... C. ... D. ...'"
    )
    correct_letter: str = Field(
        description="The single correct answer letter: A, B, C, or D"
    )
    meta_ability: str = Field(
        description="The theory-of-mind ability tested, e.g. 'theory_of_mind', "
                    "'false_belief', 'perspective_taking'"
    )
    meta_id: str = Field(
        description="Unique identifier for this question, e.g. 'synthetic_0001'"
    )

    def to_storage_dict(self, synthesis_diagnosis=None) -> dict:
        q_text = self.question  # 包含 A. B. C. D. 的完整 question

        # 解析选项：支持中英文（不用 \b，改用负向后行断言排除字母前缀）
        correct_letter = self.correct_letter.strip().upper()
        options_map: dict = {}
        import re
        pattern = re.compile(r'(?<![A-Za-z])([A-D])[.．]\s*(.*?)(?=\s*[A-D][.．]|$)', re.DOTALL)
        for m in pattern.finditer(q_text):
            options_map[m.group(1)] = m.group(2).strip()

        correct_text = options_map.get(correct_letter, correct_letter)
        wrong_texts = [v for k, v in sorted(options_map.items()) if k != correct_letter]

        return {
            "story": self.story,
            "question": q_text,
            "answer": {
                "correct_answers": [correct_text],
                "wrong_answers": wrong_texts,
            },
            "meta": {
                "id": self.meta_id,
                "condition_type": "",
                "dimension": [self.meta_ability],
            },
        }

class ToMBenchSynthesisOutput(BaseModel):
    questions: List[ToMBenchQuestionFlat]


# ---- SocialIQA ----
class SocialIQAQuestionFlat(BaseModel):
    story_full: str = Field(description="Full story text describing a social situation")
    question: str = Field(description="Question about what someone wants, needs, or reacts to")
    correct_answer: str = Field(description="The correct answer text (full sentence)")
    wrong_answer_1: str = Field(description="First plausible but incorrect answer text")
    wrong_answer_2: str = Field(description="Second plausible but incorrect answer text")
    meta_id: str = Field(description="Unique identifier, e.g. 'synthetic_0001'")
    meta_dimension: str = Field(
        description="Social dimension being tested: xWant, xNeed, xReact, xEffect, xAttr, oWant, oReact, oEffect"
    )
    meta_difficulty: str = Field(description="Difficulty level: easy, medium, or hard")

    def to_storage_dict(self, synthesis_diagnosis=None) -> dict:
        return {
            "story": self.story_full,
            "question": self.question,
            "answer": {
                "correct_answers": [self.correct_answer],
                "wrong_answers": [self.wrong_answer_1, self.wrong_answer_2],
            },
            "meta": {
                "id": self.meta_id,
                "condition_type": "",
                "dimension": [self.meta_dimension],
            },
        }

class SocialIQASynthesisOutput(BaseModel):
    questions: List[SocialIQAQuestionFlat]


# ---- BigToM ----
class BigToMQuestionFlat(BaseModel):
    story_full: str = Field(description="Full story text involving an agent's belief state")
    question: str = Field(description="Question about what the agent believes / will do / perceives")
    correct_answer: str = Field(description="The correct answer text")
    wrong_answer: str = Field(description="Exactly 1 wrong answer — what a model outputs when it skips the required belief operation")
    meta_id: str = Field(
        description="Unique identifier, e.g. 'synthetic_001__backward_belief'"
    )
    meta_condition_type: str = Field(
        description=(
            "The BigToM condition type. Must be one of: "
            "forward_belief (what does agent believe will happen), "
            "backward_belief (what does agent believe caused a past event), "
            "forward_action (what will agent do given their belief), "
            "percept_to_belief (what does agent believe based on perception), "
            "true_belief (agent has correct belief), "
            "false_belief (agent has incorrect belief). "
            "Use the condition type that matches the diagnosed error pattern in the report."
        )
    )
    meta_dimension: str = Field(
        description="Belief order: first_order or second_order"
    )

    def to_storage_dict(self, synthesis_diagnosis=None) -> dict:
        return {
            "story": self.story_full,
            "question": self.question,
            "answer": {
                "correct_answers": [self.correct_answer],
                "wrong_answers": [self.wrong_answer],
            },
            "meta": {
                "id": self.meta_id,
                "condition_type": self.meta_condition_type,
                "dimension": [self.meta_dimension],
            },
        }

class BigToMSynthesisOutput(BaseModel):
    questions: List[BigToMQuestionFlat]


# ---- EmoBench ----
class EmoBenchQuestionFlat(BaseModel):
    story_full: str = Field(description="Full story describing an emotional situation")
    question: str = Field(description="Question about the cause or reaction to an emotion")
    correct_answer: str = Field(
        description="Correct emotion answer (exact text that will appear as an option)"
    )
    wrong_answer_1: str = Field(description="First wrong emotion option text")
    wrong_answer_2: str = Field(description="Second wrong emotion option text")
    wrong_answer_3: str = Field(
        default="",
        description="Third wrong option (leave empty string if only 2 wrong answers needed)"
    )
    meta_id: str = Field(description="Unique identifier, e.g. 'synthetic_0001'")
    meta_question_subtype: str = Field(
        description="Question subtype: cause, reaction, or desire"
    )
    meta_coarse_category: str = Field(
        description="Broad emotion category, e.g. 'joy', 'sadness', 'anger'"
    )
    meta_finegrained_category: str = Field(
        description="Specific emotion name, e.g. 'excitement', 'grief', 'frustration'"
    )
    meta_dimension: str = Field(
        description="Measured dimension, e.g. 'emotion_attribution'"
    )

    def to_storage_dict(self, synthesis_diagnosis=None) -> dict:
        wrong_answers = [self.wrong_answer_1, self.wrong_answer_2]
        if self.wrong_answer_3.strip():
            wrong_answers.append(self.wrong_answer_3)
        return {
            "story": self.story_full,
            "question": self.question,
            "answer": {
                "correct_answers": [self.correct_answer],
                "wrong_answers": wrong_answers,
            },
            "meta": {
                "id": self.meta_id,
                "condition_type": "",
                "dimension": [self.meta_dimension],
            },
        }

class EmoBenchSynthesisOutput(BaseModel):
    questions: List[EmoBenchQuestionFlat]


# ---- FanToM ----
class FanToMQuestionFlat(BaseModel):
    story_full: str = Field(description="Full conversation story text with multiple speakers")
    story_summary: str = Field(
        default="",
        description="Brief summary of the story (can be empty string)"
    )
    question: str = Field(
        description="Question about what a character knows or believes"
    )
    correct_answer: str = Field(description="Correct answer text")
    wrong_answer: str = Field(description="Exactly 1 wrong answer text")
    meta_id: str = Field(
        description="ID in format '{topic}__{question_type}__{index}', "
                    "e.g. 'synthetic_001__beliefQAs__1'. "
                    "question_type must be one of: beliefQAs, factQA, "
                    "answerabilityQAs_binary, infoAccessibilityQAs_binary"
    )
    meta_question_type: str = Field(
        description="Question type: beliefQAs, factQA, answerabilityQAs_binary, "
                    "or infoAccessibilityQAs_binary"
    )

    def to_storage_dict(self, synthesis_diagnosis=None) -> dict:
        return {
            "story": self.story_full,
            "question": self.question,
            "answer": {
                "correct_answers": [self.correct_answer],
                "wrong_answers": [self.wrong_answer],
            },
            "meta": {
                "id": self.meta_id,
                "condition_type": "",
                "dimension": [],
            },
        }

class FanToMSynthesisOutput(BaseModel):
    questions: List[FanToMQuestionFlat]


# ---- HiToM ----
class HiToMQuestionFlat(BaseModel):
    story_full: str = Field(
        description="Full story involving agents with higher-order nested beliefs"
    )
    question: str = Field(
        description="Higher-order belief question (15-choice, A through O)"
    )
    correct_answer: str = Field(description="The single correct answer text")
    wrong_1: str = Field(description="Wrong option 1 (of 14 wrong options)")
    wrong_2: str = Field(description="Wrong option 2")
    wrong_3: str = Field(description="Wrong option 3")
    wrong_4: str = Field(description="Wrong option 4")
    wrong_5: str = Field(description="Wrong option 5")
    wrong_6: str = Field(description="Wrong option 6")
    wrong_7: str = Field(description="Wrong option 7")
    wrong_8: str = Field(description="Wrong option 8")
    wrong_9: str = Field(description="Wrong option 9")
    wrong_10: str = Field(description="Wrong option 10")
    wrong_11: str = Field(description="Wrong option 11")
    wrong_12: str = Field(description="Wrong option 12")
    wrong_13: str = Field(description="Wrong option 13")
    wrong_14: str = Field(description="Wrong option 14 (of 14 wrong options)")
    meta_id: str = Field(description="Unique identifier, e.g. 'synthetic_0001'")
    meta_order: int = Field(
        description="ToM recursion depth (belief order): 1, 2, 3, or 4"
    )

    def to_storage_dict(self, synthesis_diagnosis=None) -> dict:
        wrong_list = [
            self.wrong_1, self.wrong_2, self.wrong_3, self.wrong_4,
            self.wrong_5, self.wrong_6, self.wrong_7, self.wrong_8,
            self.wrong_9, self.wrong_10, self.wrong_11, self.wrong_12,
            self.wrong_13, self.wrong_14,
        ]
        return {
            "story": self.story_full,
            "question": self.question,
            "answer": {
                "correct_answers": [self.correct_answer],
                "wrong_answers": wrong_list,
            },
            "meta": {
                "id": self.meta_id,
                "condition_type": "",
                "dimension": ["belief"],
            },
        }

class HiToMSynthesisOutput(BaseModel):
    questions: List[HiToMQuestionFlat]


# ---- SimpleToM ----
class SimpleToMQuestionFlat(BaseModel):
    story_full: str = Field(description="Full story text")
    story_summary: str = Field(
        default="", description="Story summary (can be empty string)"
    )
    story_background: str = Field(
        default="", description="Background context (can be empty string)"
    )
    question: str = Field(description="Social inference question (3-choice)")
    correct_answer: str = Field(description="Correct answer text")
    wrong_answer: str = Field(
        description="One wrong answer text. A third option 'Empty' is added automatically."
    )
    meta_id: str = Field(description="Unique identifier, e.g. 'synthetic_0001'")
    meta_dimension: str = Field(
        description="Social dimension, e.g. 'social_exchange', 'false_belief'"
    )
    meta_difficulty: str = Field(description="Difficulty level: easy, medium, or hard")

    def to_storage_dict(self, synthesis_diagnosis=None) -> dict:
        return {
            "story": self.story_full,
            "question": self.question,
            "answer": {
                "correct_answers": [self.correct_answer],
                "wrong_answers": [self.wrong_answer],
            },
            "meta": {
                "id": self.meta_id,
                "condition_type": "",
                "dimension": [self.meta_dimension],
            },
        }

class SimpleToMSynthesisOutput(BaseModel):
    questions: List[SimpleToMQuestionFlat]


SYNTHESIS_SCHEMA_REGISTRY: Dict[str, Any] = {
    "ToMBench": ToMBenchSynthesisOutput,
    "SocialIQA": SocialIQASynthesisOutput,
    "BigToM": BigToMSynthesisOutput,
    "EmoBench": EmoBenchSynthesisOutput,
    "FanToM": FanToMSynthesisOutput,
    "HiToM": HiToMSynthesisOutput,
    "SimpleToM": SimpleToMSynthesisOutput,
}

# 通用 fallback schema
class _FallbackSynthesisOutput(BaseModel):
    questions: List[Dict[str, Any]] = Field(description="Generated questions")


# ============ 从维度诊断报告合成新数据 ============

def synthesize_from_reports(
    reports: List[Dict[str, Any]],
    dataset_name: str,
    synthesis_llm_config: Dict[str, Any],
    samples_per_report: int = 2,
    max_retries: int = 3,
    iteration: int = 1,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """从维度级诊断报告合成新样本，每份报告生成 samples_per_report 条题目。

    跳过 pass@k 验证，直接合成后交守门员过滤。

    Args:
        reports: DimensionDiagnosisReport dict 列表
        dataset_name: 数据集名称
        synthesis_llm_config: 合成模型配置
        samples_per_report: 每份报告生成的样本数
        max_retries: 生成失败时最多重试次数
        iteration: 迭代轮次（用于 meta_id 命名）

    Returns:
        (synthesized_data, stats) 元组
    """
    if not reports:
        logger.warning("No reports provided for synthesis")
        return [], {"total_reports": 0, "total_target": 0, "succeeded": 0, "failed": 0}

    output_schema = SYNTHESIS_SCHEMA_REGISTRY.get(dataset_name, _FallbackSynthesisOutput)
    total_target = len(reports) * samples_per_report
    logger.info(
        f"synthesize_from_reports: {len(reports)} reports × {samples_per_report} = "
        f"{total_target} target questions, schema={output_schema.__name__}"
    )

    # pending_pairs: list of (report_idx, sample_idx)
    pending_pairs: List[Tuple[int, int]] = [
        (r_idx, s_idx)
        for r_idx in range(len(reports))
        for s_idx in range(samples_per_report)
    ]
    results: Dict[Tuple[int, int], Dict[str, Any]] = {}
    last_failed: Dict[Tuple[int, int], Dict[str, Any]] = {}
    generation_attempts: Dict[Tuple[int, int], int] = {p: 0 for p in pending_pairs}

    for round_idx in range(max_retries):
        if not pending_pairs:
            break

        logger.info(f"  Round {round_idx + 1}/{max_retries}: generating {len(pending_pairs)} questions...")

        gen_prompts = [
            build_stage2_generation_from_report_prompt(
                report=reports[r_idx],
                dataset_name=dataset_name,
                target_count=1,
                previous_question=last_failed.get((r_idx, s_idx)) if round_idx > 0 else None,
            )
            for (r_idx, s_idx) in pending_pairs
        ]
        gen_results = _batch_generate_json(
            gen_prompts, output_schema, synthesis_llm_config,
            desc=f"Synthesizing {dataset_name}"
        )

        success_this_round = 0
        for pair, r in zip(pending_pairs, gen_results):
            r_idx, s_idx = pair
            generation_attempts[pair] = generation_attempts.get(pair, 0) + 1
            # r 是 Pydantic 对象或 None（来自 _batch_generate_json）
            if r is not None and hasattr(r, 'questions') and r.questions:
                q = r.questions[0]
                if hasattr(q, "to_storage_dict"):
                    q_dict = q.to_storage_dict()
                else:
                    q_dict = q if isinstance(q, dict) else dict(q)
                    q_dict.setdefault("meta", {})

                # 强制覆盖 meta_id，确保全局唯一
                meta = q_dict.get("meta", {})
                if isinstance(meta, dict):
                    meta["id"] = f"synthetic_{dataset_name.lower()}_iter{iteration}_{r_idx:03d}_{s_idx}"

                q_dict["data_source"] = "synthesized"
                results[pair] = q_dict
                success_this_round += 1
            else:
                last_failed[pair] = {}

        logger.info(f"  Round {round_idx + 1}: {success_this_round}/{len(pending_pairs)} generated")
        pending_pairs = [p for p in pending_pairs if p not in results]

    final_results = list(results.values())
    stats = {
        "total_reports": len(reports),
        "total_target": total_target,
        "succeeded": len(final_results),
        "failed": total_target - len(final_results),
        "total_generation_calls": sum(generation_attempts.values()),
    }

    logger.info(f"synthesize_from_reports complete: {len(final_results)}/{total_target} questions produced")
    return final_results, stats


# ============ 保存合成数据 ============

def save_synthesized_data(
    synthesized: List[Dict[str, Any]],
    dataset_name: str,
    output_path: str = "data_output/synth_raw",
    model_name: str = "unknown_model",
    iteration: int = 1,
) -> Path:
    """保存合成数据为 JSONL（守门员过滤前的 raw 候选）

    Args:
        synthesized: 合成数据
        dataset_name: 数据集名称
        output_path: 输出目录
        model_name: 合成模型名称
        iteration: 迭代轮次

    Returns:
        输出文件路径
    """
    output_dir = Path(output_path) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_model = model_name.replace("/", "_").replace(":", "_").replace(" ", "_")
    output_file = output_dir / f"candidates_iter{iteration}_{safe_model}.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for item in synthesized:
            # 清理内部追踪字段
            clean = {k: v for k, v in item.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(synthesized)} synthesized questions (raw) to {output_file}")
    return output_file


# ============ 阶段三：从诊断报告合成数据 ============

def run_stage3_synthesis(
    reports_dir: str,
    dataset_name: str,
    synthesis_llm_config: Dict[str, Any],
    samples_per_report: int = 2,
    max_retries: int = 3,
    output_dir: str = "data_output/synth_raw",
    model_name: str = "unknown_model",
    iteration: int = 1,
) -> Optional[Path]:
    """从维度诊断报告（dimension_reports.jsonl）生成并保存原始候选数据

    Args:
        reports_dir:           diagnosis_reports 输出目录（含 dimension_reports.jsonl）
        dataset_name:          数据集名称
        synthesis_llm_config:  合成/生成模型配置
        samples_per_report:    每份报告生成的新样本数
        max_retries:           生成失败最多重试次数
        output_dir:            输出根目录（synth_raw，守门员过滤前）
        model_name:            合成模型名称
        iteration:             迭代轮次

    Returns:
        保存候选数据的文件 Path；若无报告则返回 None
    """
    reports_dir_ = Path(reports_dir)
    reports_path = reports_dir_ / "dimension_reports.jsonl"

    if not reports_path.exists():
        raise FileNotFoundError(f"dimension_reports.jsonl not found: {reports_path}")

    logger.info(f"Stage 3 Synthesis: {dataset_name}")
    logger.info(f"  Reading from: {reports_path}")

    reports: List[Dict[str, Any]] = []
    with open(reports_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            reports.append(json.loads(line))

    if not reports:
        logger.warning("  No reports found, skipping synthesis")
        return None

    logger.info(f"  Loaded {len(reports)} dimension reports")

    synthesized, stats = synthesize_from_reports(
        reports=reports,
        dataset_name=dataset_name,
        synthesis_llm_config=synthesis_llm_config,
        samples_per_report=samples_per_report,
        max_retries=max_retries,
        iteration=iteration,
    )

    if not synthesized:
        logger.warning("  No synthesized questions produced")
        return None

    out_path = save_synthesized_data(
        synthesized=synthesized,
        dataset_name=dataset_name,
        output_path=output_dir,
        model_name=model_name,
        iteration=iteration,
    )

    logger.info(f"Stage 3 done: {len(synthesized)}/{stats['total_target']} questions saved to {out_path}")
    return out_path

