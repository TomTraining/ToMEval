# Report Tools

`report/` now acts as a thin result-consumption layer.

It does not run evaluation logic. It only reads:

- `results/.../metrics.json`
- `results/.../prediction.jsonl` for bad case display
- generated markdown tables under `tables/`

## Files

- `generate_dataset_tables.py`: build per-dataset markdown tables from `metrics.json`
- `generate_summary.py`: build `tables/SUMMARY.md` from per-dataset basic tables
- `report_client.py`: compare metrics and analyze bad cases for one model
- `generate_html_report.py`: render an HTML view from generated markdown tables
- `utils/`: shared helpers for loading results, parsing markdown tables, and sampling bad cases

## Flow

1. Run evaluation to produce `prediction.jsonl` and `metrics.json`
2. Run `python report/generate_dataset_tables.py`
3. Run `python report/generate_summary.py`
4. Optionally run `python report/report_client.py` for bad case analysis

## Principles

- tables only read `metrics.json`
- bad case analysis uses `metrics.json` as the source of correctness
- `prediction.jsonl` is only used to show prompts, predictions, and reasoning
- no old task-specific evaluation logic lives in `report/`
