"""
Data evaluation module - Phase B/C/D evaluation logic.

Modules:
- eval_passk: Phase B pass@k evaluation (weak model k trials)
- eval_answerability_full: Phase C answerability judge
- eval_shortcut: Phase D 3-dimension shortcut detection
"""

from .eval_answerability_full import run_answerability_on_df, write_answerability_parquet
from .eval_passk import (
    _BUCKET_ALL_FAILED,
    _BUCKET_ALL_PASSED,
    _BUCKET_PARTIAL,
    run_passk_on_df,
    write_passk_parquet,
)
from .eval_shortcut import run_shortcut_on_df, write_shortcut_parquet

__all__ = [
    "run_passk_on_df",
    "write_passk_parquet",
    "_BUCKET_ALL_PASSED",
    "_BUCKET_PARTIAL",
    "_BUCKET_ALL_FAILED",
    "run_answerability_on_df",
    "write_answerability_parquet",
    "run_shortcut_on_df",
    "write_shortcut_parquet",
]
