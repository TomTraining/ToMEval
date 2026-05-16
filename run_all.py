"""Unified evaluation entrypoint for standardized QA datasets."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


DATASETS = [
    "BigToM",
    "EmoBench",
    "FANToM",
    "HiToM",
    "SimpleTom",
    "SocialIQA",
    "ToMBench"
]


def run_dataset(dataset: str, experiment_config_path: str, stage: str, exp_dir: Optional[str]) -> bool:
    run_script = Path(f"tasks/{dataset}/run.py")
    if not run_script.exists():
        print(f"[{dataset}] run.py not found, skipping.")
        return False

    print(f"\n{'=' * 60}")
    print(f"Running: {dataset} ({stage})")
    print(f"{'=' * 60}")

    command = [
        sys.executable,
        str(run_script),
        "--experiment-config",
        experiment_config_path,
        "--stage",
        stage,
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
        "--stage",
        choices=["predict", "metric", "all"],
        default="all",
        help="Which stage to run.",
    )
    parser.add_argument(
        "--exp-dir",
        default=None,
        help="Existing experiment directory suffix for metric-only reruns.",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["RUN_TIMESTAMP"] = timestamp
    print(f"Run timestamp: {timestamp}")
    print(f"Experiment config: {args.experiment_config}")
    print(f"Stage: {args.stage}")

    for dataset in DATASETS:
        run_dataset(dataset, args.experiment_config, args.stage, args.exp_dir)

    print(f"\n{'=' * 60}")
    print("All datasets completed.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
