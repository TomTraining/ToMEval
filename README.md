# TomTest

Standardized QA evaluation framework for Theory-of-Mind and related benchmarks.

## What This Repo Assumes

All included QA datasets are already normalized before evaluation.

Each sample should look like:

```json
{
  "story": "context text",
  "question": "question text",
  "answer": {
    "correct_answers": ["answer text"],
    "wrong_answers": ["wrong option 1", "wrong option 2"]
  },
  "meta": {
    "id": "optional-sample-id"
  }
}
```

Rules:

- `correct_answers` is always a list.
- `wrong_answers` is empty for open QA.
- If `wrong_answers` is non-empty, the sample is treated as choice QA.
- Dataset-specific grouping fields live in `meta`.

## Evaluation Design

There is one shared evaluation pipeline:

1. `predict`
2. `metric`
3. `tables`

Shared logic in `src/` handles:

- loading normalized data
- deterministic option shuffle
- two unified English prompt templates
- free-text prediction calls via `ContentClient` (`create`)
- structured LLM judge calls via `StructureClient` (`parse`)
- saving `prediction.jsonl` and `metrics.json`

Dataset-specific logic stays in `tasks/<dataset>/metrics.py`.

## Included Tasks

- `BigToM`
- `EmoBench`
- `FANToM`
- `HiToM`
- `SimpleTom`
- `SocialIQA`
- `ToMBench`

## Output Behavior

- Open QA: model outputs answer text.
- Single-choice QA: model outputs one option letter.
- Multi-choice QA: model outputs a list of option letters.
- All correctness is decided by the judge stage.

For choice QA, prediction records include the shuffled option mapping and gold letters so results are reproducible.

## Repo Layout

```text
TomTest/
|-- experiment_config.yaml
|-- run_all.py
|-- src/
|   |-- runner.py
|   |-- evaluation/
|   |   |-- __init__.py
|   |   |-- pipeline.py
|   |   |-- data.py
|   |   |-- prediction.py
|   |   |-- judge.py
|   |   |-- judge_schema.py
|   |   |-- prompts.py
|   |   |-- storage.py
|   |   |-- paths.py
|   |   |-- metrics.py
|   |   |-- task_metrics.py
|   |   `-- types.py
|   `-- llm/
|-- tasks/
|   `-- <dataset>/
|       |-- config.yaml
|       |-- metrics.py
|       `-- run.py
|-- report/
|-- docs/
`-- tests/
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Set model and path config in `experiment_config.yaml`, then run:

```bash
python run_all.py
```

Or run one dataset:

```bash
python tasks/ToMQA/run.py
```

Run only prediction:

```bash
python run_all.py --stage predict
```

Re-run only metrics on an existing experiment:

```bash
python run_all.py --stage metric --exp-dir 20260515_120000
```

Generate tables:

```bash
python report/generate_dataset_tables.py
python report/generate_summary.py
```

## Adding a New Dataset

1. Normalize the dataset into the standard schema.
2. Add `tasks/<dataset>/config.yaml`.
3. Add `tasks/<dataset>/metrics.py` if the dataset needs custom grouped metrics.
4. Add `tasks/<dataset>/run.py`.
5. Register the dataset name in `run_all.py`.

See [docs/add_new_dataset.md](docs/add_new_dataset.md).
