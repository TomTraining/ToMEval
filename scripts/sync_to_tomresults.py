#!/usr/bin/env python3
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_RESULT_FILES = ("config.json", "metrics.json", "prediction.jsonl")


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, data):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def exp_timestamp(exp_dir):
    name = exp_dir.name
    return name.removeprefix("exp_") if name.startswith("exp_") else name


def is_valid_result_dir(path):
    return path.is_dir() and all((path / filename).is_file() for filename in REQUIRED_RESULT_FILES)


def find_latest_valid_exp(results_root, dataset, model):
    source_root = results_root / dataset / model
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source result root not found: {source_root}")

    candidates = [
        path
        for path in source_root.glob("exp_*")
        if is_valid_result_dir(path)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No valid exp_* result directories found under: {source_root}"
        )

    return max(candidates, key=lambda path: exp_timestamp(path))


def read_existing_timestamp(target_dir):
    config_path = target_dir / "config.json"
    if not config_path.is_file():
        return None

    try:
        return load_json(config_path).get("run_timestamp")
    except json.JSONDecodeError:
        return None


def should_copy(source_timestamp, target_timestamp, force):
    if force:
        return True, "force enabled"
    if target_timestamp is None:
        return True, "no existing target result"
    if source_timestamp > target_timestamp:
        return True, f"source timestamp {source_timestamp} is newer than target {target_timestamp}"
    return False, f"target timestamp {target_timestamp} is same or newer"


def sync_result(source_exp_dir, tomresults_dir, dataset, model, force):
    run_timestamp = exp_timestamp(source_exp_dir)
    source_exp_name = source_exp_dir.name
    result_path = f"results/{model}/{dataset}"
    result_id = f"{model}/{dataset}/{run_timestamp}"
    target_dir = tomresults_dir / result_path
    target_timestamp = read_existing_timestamp(target_dir)
    copy_allowed, reason = should_copy(run_timestamp, target_timestamp, force)

    print(f"Selected source exp dir: {source_exp_dir}")
    print(f"Source run_timestamp: {run_timestamp}")
    print(f"Target result dir: {target_dir}")

    if not copy_allowed:
        print(f"Skipped: {reason}")
        return False, target_dir

    target_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(source_exp_dir / "config.json")
    config.update(
        {
            "run_timestamp": run_timestamp,
            "source_exp_dir": source_exp_name,
            "model": model,
            "dataset": dataset,
            "result_id": result_id,
            "result_path": result_path,
        }
    )
    write_json(target_dir / "config.json", config)

    written_files = [target_dir / "config.json"]
    for filename in ("metrics.json", "prediction.jsonl"):
        source_path = source_exp_dir / filename
        target_path = target_dir / filename
        shutil.copy2(source_path, target_path)
        written_files.append(target_path)

    print(f"Copied: {reason}")
    print("Written target files:")
    for path in written_files:
        print(f"  - {path}")

    return True, target_dir


def metric_value(metrics, key):
    avg_metrics = metrics.get("avg_metrics", {})
    value = avg_metrics.get(key)
    return value


def int_metric_value(metrics, key):
    value = metric_value(metrics, key)
    if isinstance(value, (int, float)):
        return int(value)
    return value


def build_entry(config_path):
    result_dir = config_path.parent
    metrics_path = result_dir / "metrics.json"
    if not metrics_path.is_file():
        return None

    config = load_json(config_path)
    metrics = load_json(metrics_path)
    experiment_config = config.get("experiment_config", {})

    dataset = config.get("dataset") or result_dir.name
    model = config.get("model") or result_dir.parent.name
    result_path = config.get("result_path") or f"results/{model}/{dataset}"
    accuracy = metric_value(metrics, "accuracy")
    overall_score = accuracy * 100 if isinstance(accuracy, (int, float)) else None

    return {
        "dataset": dataset,
        "model": model,
        "run_timestamp": config.get("run_timestamp"),
        "protocol": experiment_config.get("protocol"),
        "max_samples": experiment_config.get("max_samples"),
        "stage": experiment_config.get("stage"),
        "accuracy": accuracy,
        "overall_score": overall_score,
        "correct": int_metric_value(metrics, "correct"),
        "total": int_metric_value(metrics, "total"),
        "q4_mean_score": metric_value(metrics, "q4_mean_score"),
        "result_path": result_path,
    }


def rebuild_leaderboard(tomresults_dir):
    results_dir = tomresults_dir / "results"
    entries = []
    if results_dir.is_dir():
        for config_path in sorted(results_dir.glob("*/*/config.json")):
            entry = build_entry(config_path)
            if entry is not None:
                entries.append(entry)

    entries.sort(
        key=lambda item: (
            item.get("dataset") or "",
            item.get("model") or "",
            item.get("run_timestamp") or "",
        )
    )

    leaderboard = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entries": entries,
    }
    leaderboard_path = tomresults_dir / "leaderboard.json"
    write_json(leaderboard_path, leaderboard)

    print(f"Rebuilt leaderboard: {leaderboard_path}")
    print(f"leaderboard.json entries: {len(entries)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sync the latest ToMEval result into ToMResults and rebuild leaderboard.json."
    )
    parser.add_argument("--tomresults-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--force", action="store_true", help="Overwrite even when target is same or newer.")
    return parser.parse_args()


def main():
    args = parse_args()
    tomresults_dir = args.tomresults_dir.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve()

    source_exp_dir = find_latest_valid_exp(results_root, args.dataset, args.model)
    sync_result(source_exp_dir, tomresults_dir, args.dataset, args.model, args.force)
    rebuild_leaderboard(tomresults_dir)

    print("Next step: run git status / commit / push inside ToMResults when ready.")


if __name__ == "__main__":
    main()
