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

Judge models are configured **here** (per dataset), not in `experiment_config.yaml`. Example (rubric, as used by V4p2):

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

Minimal example:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation import run_standardized_qa_task


if __name__ == "__main__":
    run_standardized_qa_task("tasks/MyDataset/config.yaml")
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

Using helpers from `src/evaluation/task_metrics.py` is recommended.

## Shared Behavior You Get Automatically

You do not need to reimplement:

- prompt building
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
