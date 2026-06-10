import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation import run_standardized_qa_task


if __name__ == "__main__":
    # V4p2 的 Q4 开放题判分通过共享 pipeline 的 open_judge=rubric 模式完成
    # （rubric prompt 见同目录 q4_judge_prompts.json，配置见 config.yaml）。
    run_standardized_qa_task("tasks/V4p2/config.yaml")
