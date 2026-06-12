import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation import run_standardized_qa_task

if __name__ == "__main__":
    run_standardized_qa_task(str(Path(__file__).parent / "config.yaml"))
