# Bad Case Analysis

`report/report_client.py` analyzes one model's bad cases for one dataset or multiple datasets.

## What It Reads

- `results/.../metrics.json`
- `results/.../prediction.jsonl`

Correctness comes from `metrics.json` judge results.
`prediction.jsonl` is only used for prompt/prediction/reasoning display.

## Usage

```bash
python report/report_client.py
python report/report_client.py report/report_config.yaml
```

## Config

Example:

```yaml
results_dir: results
tables_dir: tables
output_dir: analysis

model:
  name: Qwen3-8B
  display: Qwen3-8B

baseline:
  name: Qwen3-4B
  display: Qwen3-4B

dataset: ToMBench

bad_cases:
  n: 10
  seed: 42

no_llm_analysis: false
output_report: true
```

## Behavior

- Loads model metrics and optional baseline metrics
- Samples bad cases by tier using judged correctness
- Optionally calls an LLM to explain the failures
- Optionally writes a markdown report to `analysis/`
