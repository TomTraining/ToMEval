"""Unified evaluation entrypoint for standardized QA datasets.

stage 和 datasets 这两个参数现在统一从 experiment_config.yaml 读取:
- stage: predict | metric | all
- datasets: 要批量评测的数据集名称列表
命令行只保留 --experiment-config(指向配置)和 --exp-dir(metric-only 重跑指定目录)。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def run_dataset(dataset: str, experiment_config_path: str, exp_dir: Optional[str]) -> bool:
    run_script = Path(f"tasks/{dataset}/run.py")
    if not run_script.exists():
        print(f"[{dataset}] run.py not found, skipping.")
        return False

    print(f"\n{'=' * 60}")
    print(f"Running: {dataset}")
    print(f"{'=' * 60}")

    # stage 由子进程从 --experiment-config 里读取，这里不再透传 --stage。
    command = [
        sys.executable,
        str(run_script),
        "--experiment-config",
        experiment_config_path,
    ]
    if exp_dir:
        command.extend(["--exp-dir", exp_dir])

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=False,
            env=os.environ,
        )
        return True
    except subprocess.CalledProcessError as error:
        print(f"[{dataset}] Error: {error}")
        print(f"Return code: {error.returncode}")
        return False
    except Exception as error:
        print(f"[{dataset}] Unexpected error: {error}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-config",
        default="experiment_config.yaml",
        help="Path to the experiment config file.",
    )
    parser.add_argument(
        "--exp-dir",
        default=None,
        help="Existing experiment directory suffix for metric-only reruns.",
    )
    args = parser.parse_args()

    # stage / datasets 统一从 experiment_config 读取。
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src import runner

    experiment_config = runner.load_experiment_config(args.experiment_config)
    stage = experiment_config["stage"]
    datasets = experiment_config["datasets"]

    if not datasets:
        print("experiment_config 里的 datasets 为空，没有可评测的数据集。请在配置里填写 datasets 列表。")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["RUN_TIMESTAMP"] = timestamp
    print(f"Run timestamp: {timestamp}")
    print(f"Experiment config: {args.experiment_config}")
    print(f"Stage: {stage}")
    print(f"Datasets: {', '.join(datasets)}")

    for dataset in datasets:
        run_dataset(dataset, args.experiment_config, args.exp_dir)

    print(f"\n{'=' * 60}")
    print("All datasets completed.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
