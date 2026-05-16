import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation import run_standardized_qa_task


if __name__ == "__main__":
    run_standardized_qa_task("tasks/EmoBench/config.yaml")
