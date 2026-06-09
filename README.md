# ToMEval

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
- protocol-driven sampling, system prompts, answer extraction, and voting (see [docs/protocols.md](docs/protocols.md))
- deterministic option shuffle (disabled for the `del_tom` voting protocol)
- free-text prediction calls via `ContentClient` (`create`)
- structured LLM judge calls via `StructureClient` (`parse`), only when open-ended questions are present
- saving `prediction.jsonl` and `metrics.json`

The evaluation **protocol** (`direct` / `direct_think` / `cot` / `del_tom`) is selected with the `protocol`
field in `experiment_config.yaml` and drives sampling params, prompts, extractor, and majority voting.
See [docs/protocols.md](docs/protocols.md) for the full parameter and prompt reference.

Dataset-specific logic stays in `tasks/<dataset>/metrics.py`.

## Included Tasks

- `BigToM`
- `EmoBench`
- `FanToM`
- `HiToM`
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
ToMEval/
|-- experiment_config.yaml
|-- run_eval.py
|-- run_feedback.py
|-- run_filter.py
|-- requirements.txt
|-- src/
|   |-- evaluation/
|   |   |-- __init__.py
|   |   |-- pipeline.py
|   |   |-- data.py
|   |   |-- prediction.py
|   |   |-- protocols.py        # 协议:采样参数/system prompt/extractor
|   |   |-- voting.py           # del_tom 多数投票
|   |   |-- judge.py
|   |   |-- judge_schema.py
|   |   |-- prompts.py
|   |   |-- storage.py
|   |   |-- paths.py
|   |   |-- metrics.py
|   |   |-- task_metrics.py
|   |   `-- types.py
|   |-- llm/
|   |   |-- client.py
|   |   `-- ...
|   `-- dataloader/
|-- tasks/
|   `-- <dataset>/
|       |-- config.yaml
|       |-- metrics.py
|       `-- run.py
|-- datasets/                  # 标准化后的测试数据集
|-- train_datasets/            # 合成的训练数据集
|-- feedback/                  # 数据合成模块（bad case → 诊断 → 合成）
|   |-- config.yaml
|   |-- README.md
|   `-- ...
|-- filter/                    # 数据质量评估模块（V3 飞轮）
|   |-- config.yaml
|   |-- README.md
|   |-- eval/
|   |-- repair/
|   `-- ...
|-- report/                    # 报告生成脚本
|   |-- config.yaml
|   |-- generate_dataset_tables.py
|   |-- generate_summary.py
|   `-- generate_html_report.py
|-- tables/                    # 生成的表格和报告
|-- results/                   # 评测结果
|-- docs/                      # 文档
`-- logs/                      # 日志文件
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Set model, protocol, `stage` and `datasets` in `experiment_config.yaml`, then run:

```bash
python run_eval.py
```

Or run one dataset (still reads `stage` from the config):

```bash
python tasks/BigToM/run.py
```

Run only prediction: set `stage: predict` in `experiment_config.yaml`, then run `python run_eval.py`.

Re-run only metrics on an existing experiment: set `stage: metric`, then:

```bash
python run_eval.py --exp-dir 20260515_120000
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
5. Add the dataset name to the `datasets` list in `experiment_config.yaml`.

See [docs/add_new_dataset.md](docs/add_new_dataset.md).
