"""SoMBench（socmind v5.3）评测入口。

Q4 开放题判分通过共享 pipeline 的 open_judge=rubric 模式完成
（rubric prompt 见同目录 q4_judge_prompts.json，配置见 config.yaml）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation import run_standardized_qa_task

if __name__ == "__main__":
    run_standardized_qa_task(str(Path(__file__).parent / "config.yaml"))
