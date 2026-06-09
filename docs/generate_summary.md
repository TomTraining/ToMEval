# Generate Summary Table

`report/generate_summary.py` reads per-dataset basic tables and builds a cross-dataset summary table.

## Usage

```bash
python report/generate_summary.py
python report/generate_summary.py --stdout
```

## Expected Flow

```bash
python run_eval.py
python report/generate_dataset_tables.py
python report/generate_summary.py
```

## Output

The script writes:

```text
tables/SUMMARY.md
```
