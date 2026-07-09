# Add a New Dataset

This repo now assumes datasets are standardized before evaluation.

## Required Sample Schema

Each sample must contain:

```json
{
  "story": "context text",
  "question": "question text",
  "answer": {
    "correct_answers": ["answer text"],
    "wrong_answers": ["wrong option text"]
  },
  "meta": {}
}
```

Rules:

- `correct_answers` must be a list of strings.
- `wrong_answers` must be a list of strings.
- `wrong_answers: []` means open QA.
- `wrong_answers` non-empty means choice QA.
- Any dataset-specific grouping info should go in `meta`.

## What You Need to Add

Create a task directory:

```text
tasks/MyDataset/
|-- config.yaml
|-- metrics.py
`-- run.py
```

### `config.yaml`

Minimal example (choice-QA only — MCQ needs no judge model):

```yaml
dataset: MyDataset
path: MyDataset/test
```

`path` is resolved under `normalized_datasets_path` from `experiment_config.yaml` (default: `datasets`).

#### Open-QA datasets: choose an `open_judge` mode

If the dataset has open questions (`wrong_answers: []`), pick how they are graded via `open_judge`
(registry: `src/evaluation/open_judge.py`):

- `f1` — no judge model; token/char F1 vs `correct_answers`, binarized by `f1_threshold` (default 0.5).
- `llm_simple` — **default**; a binary LLM judge (needs a `judge1` model).
- `rubric` — dataset-provided rubric prompts score a total; needs `judge1` (optional `judge2` averaged),
  a `rubric` block (`prompts_file` / `key_field` / `max_score`) and `open_threshold`.

Judge models are configured **here** (per dataset), not in `experiment_config.yaml`. Example (rubric, as used by SoMBench):

```yaml
dataset: MyDataset
path: MyDataset/test

open_judge: rubric
open_threshold: 7.0
rubric:
  prompts_file: judge_prompts.json   # relative to this dir; {key: {prompt}}
  key_field: dim                     # pick per-key prompt by meta.<key_field>
  max_score: 10

judge1:                              # required for llm_simple / rubric
  model_name: qwen3-8b
  api_key: ${DASHSCOPE_API_KEY}
  api_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  # 云端约束：thinking 开启只支持流式，judge 走非流式 → 关闭；dashscope qwen3 上限 8192
  enable_thinking: false
  max_tokens: 8192
```

### `run.py`

每个数据集的 `run.py` 内容完全一致——从 `__file__` 推导同目录的 `config.yaml`，无需写死数据集名，直接复制即可：

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation import run_standardized_qa_task

if __name__ == "__main__":
    run_standardized_qa_task(str(Path(__file__).parent / "config.yaml"))
```

### `metrics.py`

If plain accuracy is enough:

```python
from __future__ import annotations

from typing import Any, Dict, List

from src.evaluation.task_metrics import base_metric_payload


def compute_metrics(records: List[Dict[str, Any]], per_sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    del records
    return base_metric_payload(per_sample_results)
```

If you need grouped metrics, implement:

```python
from __future__ import annotations

from typing import Any, Dict, List


def compute_metrics(records: List[Dict[str, Any]], per_sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    ...
```

Inputs:

- `records`: rows from `prediction.jsonl` for one repeat
- `per_sample_results`: judge outputs for the same repeat

The returned dict must include:

- `accuracy`
- `correct`
- `total`
- `per_sample_results`

Using helpers from `src/evaluation/task_metrics.py` is recommended. Two helpers worth knowing:

- `generic_group_metrics(...)` — emits `by_<x>` accuracy breakdowns from `(name, key_fn)` pairs; the visualizer turns each into a chart automatically. For the common case of grouping by a single `meta` field, use the `by_meta("<field>")` helper instead of writing the lambda by hand, e.g. `("by_dimension", by_meta("dimension"))`.
- `group_all_correct(records, results, key_fn, member_fn, required_members)` — a group-level "all-correct" secondary metric: a group passes only if every `required_members` member is answered correctly (e.g. FanToM's set-level ALL, where all ToM question types in one snippet must be right; TactfulToM's Comprehension∧Justification joint score). Returns `{rate, passed, total}`.

## Custom Prompts (optional, faithful to the original paper)

By default prompts are generated uniformly from the protocol (see [protocols.md](protocols.md)). If you need to reproduce the paper's exact wording, add `tasks/<dataset>/prompt.py`. It is loaded by convention (`src/evaluation/prompt_loaders.py`); any hook you omit falls back to the generic implementation. Add a comment marker like `prompt_style: <name>` in `config.yaml` for discoverability (it is documentation only — no code reads it).

```python
# tasks/MyDataset/prompt.py
from typing import Dict, Optional
from src.evaluation.types import StandardizedSample


def build_prompt(sample: StandardizedSample, option_map: Optional[Dict[str, str]],
                 include_instruction: bool = True) -> str:
    """User-side body only: story / question / options layout, in the paper's wording.
    Same signature as src.evaluation.prompts.build_prompt. The answer-format hint is
    carried by the system prompt; only emit it here when include_instruction=True
    (legacy / no-protocol mode)."""
    ...


def build_system_prompt(sample, protocol: str, lang: str, prompt_type: str) -> str:
    """The official instruction block, with the answer format swapped to \\boxed{}."""
    ...


def prepare_samples(samples):
    """Whole-sample transform run before prediction. Use it to bundle sub-questions
    into one mcq_grouped sample: set meta.prompt_type_override = "mcq_grouped" and
    meta.sub_questions = [{subtype, correct_letters, ...}, ...]. rule_judge_grouped
    then requires every sub-question to be correct."""
    return samples
```

Recap of the split: **system prompt** = answer style + format instruction (how to answer); **user prompt** = the actual story / question / options (what to answer). See ToMBench / EmoBench / FanToM for working examples.

## Shared Behavior You Get Automatically

You do not need to reimplement:

- prompt building (generic by default; override per dataset via `prompt.py` as above)
- open-vs-choice task routing
- deterministic shuffle
- prediction generation
- judging (MCQ by `\boxed{}` rule; open QA by the dataset's `open_judge` mode)
- judge prompt construction
- result file writing
- visualization (figures from `metrics.json`)

Those all live in `src/evaluation/`.

## Register the Dataset

Add the dataset name to the `datasets` list in `experiment_config.yaml`.

## Smoke Test

Set `stage` in `experiment_config.yaml` (`predict` / `metric` / `visualize` / `all`), then:

```bash
python tasks/MyDataset/run.py                       # 跑 stage=predict 或 all
python tasks/MyDataset/run.py --exp-dir 20260515_120000   # stage=metric 时指定已有目录
```
