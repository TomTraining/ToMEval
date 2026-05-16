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

Minimal example:

```yaml
dataset: MyDataset
path: MyDataset/test
```

`path` is resolved under `normalized_datasets_path` from `experiment_config.yaml` (default: `datasets`).

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
- judge parse calls
- judge prompt construction
- result file writing

Those all live in `src/evaluation/`.

## Register the Dataset

Add the dataset name to `DATASETS` in `run_all.py`.

## Smoke Test

```bash
python tasks/MyDataset/run.py --stage predict
python tasks/MyDataset/run.py --stage metric --exp-dir 20260515_120000
```
