"""F040：基于 train_eval_output 各子集 4 指标，生成 markdown 质量报告。

报告结构（与文档约定一致）：
1. 评估指标说明（详解每个指标的具体计算方式）
2. 4 指标分别排序的总览表（simple_pr / strong_diff / ans_score / repr_mean）
3. 各数据集详细分析与修复建议（由 report_model 自动生成自然语言）

report_model 仅做"汇总→自然语言"的归纳，不参与打分。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.llm.content_client import ContentClient

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_report_client() -> ContentClient:
    cfg = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    rm = cfg.get("report_model")
    if not isinstance(rm, dict):
        raise ValueError("config.yaml report_model 缺失或不是 dict")
    return ContentClient(
        model_name=rm["model_name"],
        api_key=rm["api_key"],
        api_url=rm["api_url"],
        temperature=rm.get("temperature", 0.3),
        max_workers=rm.get("max_workers", 4),
        max_tokens=rm.get("max_tokens", 4096),
        enable_thinking=False,
    )


# ── 指标说明（写死，由开发者维护，不交给 LLM 生成） ───────────────────────────

_METRICS_SECTION = """## 1. 评估指标详细说明

报告共统计 4 个独立指标，分别由 `data_eval/` 下四个评估器计算。**互不融合、不计算综合得分**——每个指标各自一张排序表，便于针对性诊断。

### 1.1 `simple_pr` —— 简模型重复通过率（F035）

- **计算流程**：用 config.yaml `eval_model.simple`（默认 qwen3-8b）对每个采样样本独立答题 5 次，统计每条样本答对次数 `k ∈ {0,1,2,3,4,5}`。
- **聚合公式**：`simple_pr = (Σ k_i) / (n × 5)`，其中 `n` 为采样样本数（由 `config.yaml:sample_rows` 控制）。
- **取值范围**：`0..1`。
- **怎么读**：
  - `simple_pr ≥ 0.95`：简模型几乎全对，**任务过易**——再练对该模型基本无收益；
  - `simple_pr ≤ 0.10`：简模型几乎全错，**疑似不可解 / 标签错位 / 选项不齐**；
  - 健康区间：`0.30 ~ 0.85`。
- **代码**：`data_eval/eval_difficulty.py:run_difficulty_eval_on_df` 中的 `simple_mean_pass_rate`。

### 1.2 `strong_diff` —— 强模型主观难度评分（F035）

- **计算流程**：用 config.yaml `eval_model.strong`（默认 deepseek-v4-flash）对每个样本读取 `(story, question, correct_answers)` 后，按 `data_eval/prompts.py:DIFFICULTY_PROMPTS` 给出 `0..5` 整数难度分（`0` 极易、`5` 极难）。
- **聚合公式**：`strong_diff = Σ score_i / 已成功打分的样本数`（解析失败的样本不计入分母，单独记 `strong_failed_count`）。
- **取值范围**：`0..5`。
- **怎么读**：
  - `strong_diff ≤ 1.5`：强模型也认为很简单，**训练增益有限**；
  - `strong_diff ≥ 4.0`：强模型也认为很难，结合 `simple_pr ≤ 0.10` 时高度怀疑**样本本身不可解**；
  - 健康区间：`2.0 ~ 3.5`。
- **代码**：`data_eval/eval_difficulty.py` 中的 `strong_difficulty_mean`。

### 1.3 `ans_score` —— 三阶级联可答性得分（F036）

- **计算流程**（三阶级联，`data_eval/eval_answerability.py`）：
  1. **阶段 A**：复用 F035 的 `simple_correct[5]`，将每条样本归入 `all_passed / partial_failed / all_failed`；
  2. **阶段 B**：仅对 `partial_failed + all_failed` 的样本，让强模型再答一次，分桶为 `upgraded_partial / still_failed`；
  3. **阶段 C**：仅对 `still_failed` 的样本，让强模型做"原因诊断"，输出标签：`truly_hard / label_error / ambiguous / contradictory_premise`。
- **聚合公式**：`ans_score = 1 − (label_error + ambiguous + contradictory_premise) / n`。`truly_hard` 不扣分（视为合理难题，不算质量缺陷）。
- **取值范围**：`0..1`。
- **怎么读**：`ans_score` 直接代表"非缺陷样本占比"，越高越好。`< 0.7` 通常意味着 ≥30% 的样本因标签错误/歧义/前提矛盾而不应进入训练集。
- **代码**：`data_eval/eval_answerability.py:run_answerability_eval_on_df` 中的 `answerability_score`。

### 1.4 `repr_mean` —— 代表性 0-5 均分（F037）

- **计算流程**：用强模型对每条样本沿固定维度（场景/角色/心智操作/语言风格等，详见 `data_eval/prompts.py` 中 `REPRESENTATIVENESS_PROMPTS`）打 `0..5` 综合代表性分，并要求模型按 `dimension_breakdown` 输出维度分解。
- **聚合公式**：`repr_mean = Σ score_i / n`。
- **取值范围**：`0..5`。越高代表样本对该数据集"任务画像"的覆盖度越好。
- **怎么读**：
  - `repr_mean < 2.5`：题干形式或场景过于单一，**多样性不足**，建议引入新模板；
  - `repr_mean ≥ 4.0`：覆盖度优秀；
  - 健康区间：`3.0 ~ 4.5`。
- **代码**：`data_eval/eval_representativeness.py:run_representativeness_eval_on_df` 中的 `mean_representativeness_score`。

### 1.5 附属指标

- **`fmt_pass / fmt_fail`**：format 校验是否全通过（基于 `synth_clean` audit 规则）。`FAIL` 多源自 `train_datasets` meta schema 与 `synth_clean` 的字段命名差异，并不必然代表样本本身损坏，但需要修复 schema 对齐才能复用下游 audit 工具链。
"""


def _flat_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把每个子集压平成单行 dict，便于排序。"""
    rows = []
    for r in results:
        f = r["format"]
        d = r["difficulty"]
        a = r["answerability"]
        rp = r["representativeness"]
        rows.append({
            "dataset": r["dataset"],
            "subset": r["subset"],
            "total_rows": r["total_rows"],
            "sample_rows": r["sample_rows"],
            "fmt_pass": f["pass"],
            "fmt_fail": f["failures_count"],
            "simple_pr": d["simple_mean_pass_rate"],
            "strong_diff": d["strong_difficulty_mean"],
            "ans_score": a["answerability_score"],
            "repr_mean": rp["mean_representativeness_score"],
        })
    return rows


def _label(row: Dict[str, Any], metric: str) -> str:
    sp = row["simple_pr"]
    sd = row["strong_diff"]
    if sp is None or sd is None:
        return ""
    if metric == "simple_pr":
        if sp >= 0.95:
            return "过易⚠️"
        if sp <= 0.10:
            return "疑似不可解⚠️"
        return ""
    if metric == "strong_diff":
        if sd <= 1.5:
            return "训练价值低⚠️"
        if sd >= 4.0:
            return "可能过难⚠️"
        return ""
    if metric == "ans_score":
        v = row["ans_score"]
        if v is not None and v < 0.7:
            return "可答性偏低⚠️"
        return ""
    if metric == "repr_mean":
        v = row["repr_mean"]
        if v is not None and v < 2.5:
            return "多样性不足⚠️"
        return ""
    return ""


def _sort_table(rows: List[Dict[str, Any]], metric: str, descending: bool) -> str:
    sortable = [r for r in rows if r.get(metric) is not None]
    sortable.sort(key=lambda r: r[metric], reverse=descending)

    fmt_str = lambda b: "PASS" if b else "FAIL"
    lines = []
    lines.append("| 排名 | 数据集 / 子集 | fmt | fmt_fail | simple_pr | strong_diff | ans_score | repr_mean | 标签 |")
    lines.append("|----:|---------------|:---:|---------:|----------:|------------:|----------:|----------:|------|")
    for i, r in enumerate(sortable, 1):
        # 用 ★ 标注当前排序键所在列
        cells = {
            "simple_pr": str(r["simple_pr"]),
            "strong_diff": str(r["strong_diff"]),
            "ans_score": str(r["ans_score"]),
            "repr_mean": str(r["repr_mean"]),
        }
        cells[metric] = f"**{cells[metric]}**"
        lines.append(
            f"| {i} | {r['dataset']} / {r['subset']} "
            f"| {fmt_str(r['fmt_pass'])} | {r['fmt_fail']} "
            f"| {cells['simple_pr']} | {cells['strong_diff']} | {cells['ans_score']} | {cells['repr_mean']} "
            f"| {_label(r, metric)} |"
        )
    return "\n".join(lines)


_RANKING_NOTES = {
    "simple_pr": "**升序**：值越小=简模型越答不出来，排在最前的若同时 `strong_diff` 高且 `ans_score` 低，则疑似存在不可解/标签错位样本。",
    "strong_diff": "**降序**：值越大=强模型也认为越难。排前若 `simple_pr` 同时 ≤0.10，需要核查样本是否 truly-impossible。",
    "ans_score": "**降序**：值越大=非缺陷样本占比越高，可直接读作“训练可用率”。",
    "repr_mean": "**降序**：值越大=场景/角色/心智操作覆盖越广。低于 2.5 的 split 建议补模板。",
}


def _ranking_section(rows: List[Dict[str, Any]]) -> str:
    parts = ["## 2. 四指标分别排序"]
    parts.append("")
    parts.append("以下四张表对**同一份子集列表**按各指标独立排序，互不融合。**当前排序键的数值用粗体标识**；标签栏的 ⚠️ 仅作信号提示，不影响排序。")
    parts.append("")

    parts.append("### 2.1 按 `simple_pr` 升序（先看疑似不可解的）")
    parts.append("")
    parts.append(_RANKING_NOTES["simple_pr"])
    parts.append("")
    parts.append(_sort_table(rows, "simple_pr", descending=False))
    parts.append("")

    parts.append("### 2.2 按 `strong_diff` 降序（先看强模型也觉得难的）")
    parts.append("")
    parts.append(_RANKING_NOTES["strong_diff"])
    parts.append("")
    parts.append(_sort_table(rows, "strong_diff", descending=True))
    parts.append("")

    parts.append("### 2.3 按 `ans_score` 降序（直接读作训练可用率）")
    parts.append("")
    parts.append(_RANKING_NOTES["ans_score"])
    parts.append("")
    parts.append(_sort_table(rows, "ans_score", descending=True))
    parts.append("")

    parts.append("### 2.4 按 `repr_mean` 降序（覆盖度排名）")
    parts.append("")
    parts.append(_RANKING_NOTES["repr_mean"])
    parts.append("")
    parts.append(_sort_table(rows, "repr_mean", descending=True))
    parts.append("")
    return "\n".join(parts)


_PROMPT_TEMPLATE = """你是一名数据质量分析师。下面是某 ToM 训练数据集的 4 项独立质量评估结果（不要计算综合得分）。请按要求输出该数据集的"详细分析与修复建议"小节，**用中文 markdown**。

数据集：{dataset}

各子集指标（JSON）：
```json
{rows_json}
```

指标含义（已有完整说明，无需展开重述）：
- simple_pr ∈ [0,1]：简模型 qwen3-8b repeat=5 通过率。≥0.95 过易；≤0.10 疑似不可解。
- strong_diff ∈ [0,5]：强模型 deepseek-v4-flash 主观难度。≤1.5 训练价值低；≥4.0 可能过难。
- ans_score ∈ [0,1]：可答性=1-质量缺陷率（truly_hard 不扣）。<0.7 偏低。
- repr_mean ∈ [0,5]：代表性均分。<2.5 多样性不足。
- fmt_pass：format 校验是否全通过（FAIL 多为 schema 字段命名差异）。

输出要求：
1. **第一段（80-120 字）**：先给该数据集整体画像（哪些子集质量高/低、是否存在系统性问题）。
2. **逐子集列表**：每个子集 1-2 句，点出最突出的问题或亮点（**必须基于具体数值**，例如“simple_pr=0.0 + strong_diff=4.86 → 疑似 truly-unsolvable”）。
3. **修复建议（要点列表，3-5 条）**：每条聚焦一个可执行的动作，按“问题→动作→预期收益”格式。建议按以下原则给出：
   - format 不通过 → schema normalize；
   - simple_pr 极低 + ans_score 低 → 用 F036 cascade 过滤 `label_error / ambiguous / contradictory_premise`；
   - simple_pr 极高 + strong_diff 极低 → 增加困难变体或在混训中降权；
   - repr_mean 低 → 扩展场景/角色/心智操作模板；
   - strong_diff 高 + simple_pr 极低 → 仅作 eval-only 或改 CoT/分步训练。
4. 严禁编造未在数据中出现的数值。严禁给出“综合 quality 得分”。

只输出该数据集小节本身（以 `### {dataset}` 作为小标题），不要重复指标说明，也不要包裹在 ``` 代码块中。"""


def _gen_dataset_section(client: ContentClient, dataset: str, ds_rows: List[Dict[str, Any]]) -> str:
    rows_json = json.dumps(ds_rows, ensure_ascii=False, indent=2)
    prompt = _PROMPT_TEMPLATE.format(dataset=dataset, rows_json=rows_json)
    resp = client.generate(prompt)
    text = resp.content.strip() if resp and resp.content else ""
    if not text:
        text = f"### {dataset}\n\n_[报告生成失败：模型返回空]_"
    if not text.lstrip().startswith("###"):
        text = f"### {dataset}\n\n" + text
    return text


def _by_dataset(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(r["dataset"], []).append(r)
    return out


def generate_quality_report(
    results: List[Dict[str, Any]],
    out_path: Path,
) -> Path:
    """生成 markdown 质量报告。

    Args:
        results: 每个子集的评估结果（与 run_eval.eval_subset 返回值同构）。
        out_path: 输出 .md 路径。
    """
    rows = _flat_rows(results)
    by_ds = _by_dataset(rows)

    client = _load_report_client()

    parts: List[str] = []
    parts.append("# Train Datasets 数据质量报告")
    parts.append("")
    parts.append("评估时间: 由 `run_eval.py` 生成（F040 收尾版，合并自原 run_train_eval.py）。")
    parts.append("")
    parts.append("本报告由两段构成：")
    parts.append("- §1 / §2 由 `data_eval/report.py` 直接基于 `train_eval_output/` 汇总写出（确定性）。")
    parts.append("- §3 各数据集详细分析与修复建议由 `config.yaml:report_model` 模型自动归纳生成。")
    parts.append("")
    parts.append(_METRICS_SECTION)
    parts.append("")
    parts.append(_ranking_section(rows))
    parts.append("")
    parts.append("## 3. 各数据集详细分析与修复建议")
    parts.append("")
    parts.append("> 以下小节由 report_model（`config.yaml:report_model`）基于该数据集所有子集的 4 指标自动生成；不计算综合得分。")
    parts.append("")

    for dataset in sorted(by_ds.keys()):
        section = _gen_dataset_section(client, dataset, by_ds[dataset])
        parts.append(section)
        parts.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path
