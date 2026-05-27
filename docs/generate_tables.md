# Generate Dataset Tables

`report/generate_dataset_tables.py` converts `metrics.json` files under `results/` into markdown tables.

## Input

The script reads:

- `results/{dataset}/{model}/exp_*/metrics.json`
- `results/{dataset}/{model}/exp_*/config.json` for metadata copy

It does not read raw datasets or run any evaluation logic.

## Output

For each dataset it writes:

- `tables/{Dataset}/基础指标.md`
- `tables/{Dataset}/其他指标.md`
- `tables/{Dataset}/{ModelDisplay}/config.json`

## Usage

```bash
python report/generate_dataset_tables.py
python report/generate_dataset_tables.py report/tables_config.yaml
```

## Config

Example:

```yaml
results_dir: results
output_dir: tables
exp_suffix:
dataset:
  - ToMBench
models:
  - name: Qwen3-8B
    display: Qwen3-8B-Think
```

Rules:

- `exp_suffix` empty means use the latest `exp_*`
- `dataset` empty means process all datasets
- `models` empty means process all models
