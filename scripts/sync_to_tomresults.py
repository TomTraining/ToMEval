#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_RESULT_FILES = ("config.json", "metrics.json", "prediction.jsonl")
LEGACY_METRIC_FIELDS = (
    "by_dim1",
    "by_dim1_counts",
    "by_dim2",
    "by_dim2_counts",
    "by_dim3",
    "by_dim3_counts",
    "by_dim3_qtype",
    "by_dim3_qtype_counts",
    "by_qtype",
    "by_qtype_counts",
    "by_perspective",
    "by_perspective_counts",
    "by_length",
    "by_length_counts",
    "by_variant",
    "by_variant_counts",
    "q4_mean_score",
    "q4_count",
    "q4_mean_score_by_dim",
)
QTYPE_CODE_NAMES = {
    "Q1": "Single Choice",
    "Q2": "Multiple Choice",
    "Q3": "Judgment & Reasoning",
    "Q4": "Open Analysis",
}
BREAKDOWN_DEFS = (
    ("dimension_level_1", "Dimension Level 1", "dim1", "by_dim1", "by_dim1_counts", "dim1"),
    ("dimension_level_2", "Dimension Level 2", "dim2", "by_dim2", "by_dim2_counts", "dim2"),
    ("dimension_level_3", "Dimension Level 3", "dim3", "by_dim3", "by_dim3_counts", "dim3"),
    ("question_type", "Question Type", "qtype", "by_qtype", "by_qtype_counts", "qtype"),
    ("perspective", "Perspective", "perspective", "by_perspective", "by_perspective_counts", None),
    ("length", "Length", "length", "by_length", "by_length_counts", None),
    ("variant", "Variant", "variant", "by_variant", "by_variant_counts", "variant"),
)


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


def default_branch_name(model, dataset, run_timestamp):
    return f"results/{model}-{dataset}-{run_timestamp}"


def run_git(tomresults_dir, args, check=True):
    command = ["git", *args]
    result = subprocess.run(
        command,
        cwd=tomresults_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        command_text = " ".join(command)
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{command_text} failed: {message}")
    return result


def checkout_result_branch(tomresults_dir, branch_name):
    exists = run_git(
        tomresults_dir,
        ["rev-parse", "--verify", "--quiet", branch_name],
        check=False,
    ).returncode == 0

    if exists:
        run_git(tomresults_dir, ["checkout", branch_name])
        print(f"Checked out existing branch: {branch_name}")
    else:
        run_git(tomresults_dir, ["checkout", "-b", branch_name])
        print(f"Created and checked out branch: {branch_name}")


def allowed_git_paths(model, dataset):
    result_path = f"results/{model}/{dataset}"
    return [
        "leaderboard.json",
        f"{result_path}/config.json",
        f"{result_path}/metrics.json",
        f"{result_path}/prediction.jsonl",
    ]


def commit_synced_files(tomresults_dir, model, dataset, run_timestamp):
    paths = allowed_git_paths(model, dataset)
    run_git(tomresults_dir, ["add", *paths])

    diff_result = run_git(tomresults_dir, ["diff", "--cached", "--quiet"], check=False)
    if diff_result.returncode == 0:
        print("Commit skipped: no staged changes.")
        return None

    message = f"Add results for {model} on {dataset} at {run_timestamp}"
    run_git(tomresults_dir, ["commit", "-m", message])
    commit_hash = run_git(tomresults_dir, ["rev-parse", "--short", "HEAD"]).stdout.strip()
    print(f"Committed synced files: {commit_hash}")
    return commit_hash


def push_result_branch(tomresults_dir, branch_name):
    if not branch_name:
        branch_name = run_git(tomresults_dir, ["branch", "--show-current"]).stdout.strip()
    run_git(tomresults_dir, ["push", "-u", "origin", branch_name])
    print(f"Pushed branch: {branch_name}")
    return branch_name


def pr_title(model, dataset, run_timestamp):
    return f"Add results for {model} on {dataset} at {run_timestamp}"


def pr_body(model, dataset, run_timestamp, result_path, source_exp_dir):
    return "\n".join(
        [
            "This PR syncs a ToMEval result into ToMResults.",
            "",
            f"- model: {model}",
            f"- dataset: {dataset}",
            f"- run_timestamp: {run_timestamp}",
            f"- result_path: {result_path}",
            f"- source_exp_dir: {source_exp_dir}",
            "- leaderboard.json regenerated: yes",
            "- generated by: scripts/sync_to_tomresults.py",
        ]
    )


def build_pr_payload(repo, base_branch, head_branch, model, dataset, run_timestamp, result_path, source_exp_dir):
    return {
        "repo": repo,
        "title": pr_title(model, dataset, run_timestamp),
        "body": pr_body(model, dataset, run_timestamp, result_path, source_exp_dir),
        "head": head_branch,
        "base": base_branch,
    }


def print_pr_dry_run(payload):
    print("Dry-run PR payload:")
    print(f"  repo: {payload['repo']}")
    print(f"  title: {payload['title']}")
    print(f"  head: {payload['head']}")
    print(f"  base: {payload['base']}")
    print("  body:")
    for line in payload["body"].splitlines():
        print(f"    {line}")


def create_github_pr(payload, token):
    api_url = f"https://api.github.com/repos/{payload['repo']}/pulls"
    request_body = json.dumps(
        {
            "title": payload["title"],
            "body": payload["body"],
            "head": payload["head"],
            "base": payload["base"],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=request_body,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "ToMEval-sync-to-tomresults",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"GitHub PR creation failed with HTTP {error.code}: {error_body}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"GitHub PR creation failed: {error.reason}") from error

    print(f"Created PR: {response_payload.get('html_url', 'URL unavailable')}")
    return response_payload


def metric_value(metrics, key):
    avg_metrics = metrics.get("avg_metrics", {})
    value = avg_metrics.get(key)
    return value


def int_metric_value(metrics, key):
    value = metric_value(metrics, key)
    if isinstance(value, (int, float)):
        return int(value)
    return value


def load_socialmind_name_maps():
    config_path = Path(__file__).resolve().parents[1] / "tasks" / "SocialMind" / "dim_config.py"
    maps = {"dim1": {}, "dim2": {}, "dim3": {}, "qtype": dict(QTYPE_CODE_NAMES), "variant": {}}
    if not config_path.is_file():
        return maps

    spec = importlib.util.spec_from_file_location("socialmind_dim_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    maps["dim1"] = dict(getattr(module, "DIM1_NAMES_EN", {}) or getattr(module, "DIM1_NAMES", {}) or {})
    maps["dim2"] = dict(getattr(module, "DIM2_NAMES_EN", {}) or getattr(module, "DIM2_NAMES", {}) or {})
    maps["dim3"] = dict(getattr(module, "DIM3_NAMES_EN", {}) or getattr(module, "DIM3_NAMES", {}) or {})
    maps["variant"] = dict(getattr(module, "VARIANT_NAMES_EN", {}) or getattr(module, "VARIANT_NAMES", {}) or {})

    qtype_names = getattr(module, "QTYPE_NAMES_EN", {}) or {}
    maps["qtype"].update({
        "Q1": qtype_names.get("单选", QTYPE_CODE_NAMES["Q1"]),
        "Q2": qtype_names.get("多选", QTYPE_CODE_NAMES["Q2"]),
        "Q3": qtype_names.get("判断推理", QTYPE_CODE_NAMES["Q3"]),
        "Q4": qtype_names.get("开放分析", QTYPE_CODE_NAMES["Q4"]),
    })
    return maps


def display_name(name_maps, map_key, item_id):
    if map_key and item_id in name_maps.get(map_key, {}):
        return name_maps[map_key][item_id]
    return item_id


def sorted_items(mapping):
    return sorted(mapping.items(), key=lambda item: str(item[0]))


def item_from_split(item_id, split, name_maps, map_key):
    if not isinstance(split, dict):
        return None
    score = split.get("acc")
    count = split.get("n")
    if score is None and count is None:
        return None
    item = {
        "id": str(item_id),
        "name": display_name(name_maps, map_key, str(item_id)),
    }
    if score is not None:
        item["score"] = score
    if count is not None:
        item["count"] = int(count) if isinstance(count, (int, float)) else count
    return item


def items_from_scores(scores, counts, name_maps, map_key):
    if not isinstance(scores, dict):
        return []
    counts = counts if isinstance(counts, dict) else {}
    items = []
    for item_id, score in sorted_items(scores):
        item = {
            "id": str(item_id),
            "name": display_name(name_maps, map_key, str(item_id)),
            "score": score,
        }
        if item_id in counts:
            count = counts[item_id]
            item["count"] = int(count) if isinstance(count, (int, float)) else count
        items.append(item)
    return items


def add_breakdown(breakdowns, key, label, items):
    items = [item for item in items if item is not None]
    if items:
        breakdowns[key] = {"label": label, "items": items}


def dimension_items(dimensions, dim_name, name_maps, map_key):
    source = dimensions.get(dim_name)
    if not isinstance(source, dict):
        return []
    return [
        item_from_split(item_id, split, name_maps, map_key)
        for item_id, split in sorted_items(source)
    ]


def nested_dimension_items(dimensions, path, name_maps, map_key):
    def walk(node, remaining):
        if not remaining or not isinstance(node, dict):
            return []
        dim_name = remaining[0]
        if dim_name not in node:
            return []
        dim = node[dim_name]
        if not isinstance(dim, dict):
            return []
        if len(remaining) == 1:
            return [
                item_from_split(item_id, split, name_maps, map_key)
                for item_id, split in sorted_items(dim)
            ]

        items = []
        for split in dim.values():
            child = split.get("dimensions") if isinstance(split, dict) else None
            items.extend(walk(child, remaining[1:]))
        return items

    by_id = {}
    for item in walk(dimensions, path):
        if item is not None:
            by_id[item["id"]] = item
    return [by_id[item_id] for item_id in sorted(by_id)]


def build_breakdowns_from_dimensions(avg_metrics, name_maps):
    dimensions = avg_metrics.get("dimensions")
    if not isinstance(dimensions, dict) or not dimensions:
        return {}

    breakdowns = {}
    add_breakdown(
        breakdowns,
        "dimension_level_1",
        "Dimension Level 1",
        dimension_items(dimensions, "dim1", name_maps, "dim1"),
    )
    add_breakdown(
        breakdowns,
        "dimension_level_2",
        "Dimension Level 2",
        nested_dimension_items(dimensions, ("dim1", "dim2"), name_maps, "dim2"),
    )
    add_breakdown(
        breakdowns,
        "dimension_level_3",
        "Dimension Level 3",
        nested_dimension_items(dimensions, ("dim1", "dim2", "dim3"), name_maps, "dim3"),
    )

    for key, label, dim_name, _scores_key, _counts_key, map_key in BREAKDOWN_DEFS[3:]:
        add_breakdown(
            breakdowns,
            key,
            label,
            dimension_items(dimensions, dim_name, name_maps, map_key),
        )
    return breakdowns


def build_breakdowns_from_legacy(avg_metrics, name_maps):
    breakdowns = {}
    for key, label, _dim_name, scores_key, counts_key, map_key in BREAKDOWN_DEFS:
        add_breakdown(
            breakdowns,
            key,
            label,
            items_from_scores(avg_metrics.get(scores_key), avg_metrics.get(counts_key), name_maps, map_key),
        )
    return breakdowns


def breakdown_to_legacy_scores(breakdowns, breakdown_key):
    breakdown = breakdowns.get(breakdown_key)
    if not isinstance(breakdown, dict):
        return None, None
    scores = {}
    counts = {}
    for item in breakdown.get("items") or []:
        item_id = item.get("id")
        if item_id is None:
            continue
        if "score" in item:
            scores[str(item_id)] = item["score"]
        if "count" in item:
            counts[str(item_id)] = item["count"]
    return (scores or None), (counts or None)


def derive_legacy_fields_from_dimensions(avg_metrics, breakdowns):
    derived = {}
    for breakdown_key, scores_key, counts_key in (
        ("dimension_level_1", "by_dim1", "by_dim1_counts"),
        ("dimension_level_2", "by_dim2", "by_dim2_counts"),
        ("dimension_level_3", "by_dim3", "by_dim3_counts"),
        ("question_type", "by_qtype", "by_qtype_counts"),
        ("perspective", "by_perspective", "by_perspective_counts"),
        ("length", "by_length", "by_length_counts"),
        ("variant", "by_variant", "by_variant_counts"),
    ):
        scores, counts = breakdown_to_legacy_scores(breakdowns, breakdown_key)
        if scores is not None:
            derived[scores_key] = scores
        if counts is not None:
            derived[counts_key] = counts

    q4_score = (avg_metrics.get("dimensions") or {}).get("q4_score")
    if isinstance(q4_score, dict):
        overall = q4_score.get("overall")
        if isinstance(overall, dict):
            if overall.get("acc") is not None:
                derived["q4_mean_score"] = overall.get("acc")
            if overall.get("n") is not None:
                derived["q4_count"] = overall.get("n")
        by_dim = {
            key: split.get("acc")
            for key, split in q4_score.items()
            if key != "overall" and isinstance(split, dict) and split.get("acc") is not None
        }
        if by_dim:
            derived["q4_mean_score_by_dim"] = by_dim
    return derived


def build_entry(config_path):
    result_dir = config_path.parent
    metrics_path = result_dir / "metrics.json"
    if not metrics_path.is_file():
        return None

    config = load_json(config_path)
    metrics = load_json(metrics_path)
    avg_metrics = metrics.get("avg_metrics", {})
    experiment_config = config.get("experiment_config", {})
    name_maps = load_socialmind_name_maps() if (config.get("dataset") or result_dir.name) == "SocialMind" else {
        "dim1": {},
        "dim2": {},
        "dim3": {},
        "qtype": dict(QTYPE_CODE_NAMES),
        "variant": {},
    }

    dataset = config.get("dataset") or result_dir.name
    model = config.get("model") or result_dir.parent.name
    result_path = config.get("result_path") or f"results/{model}/{dataset}"
    accuracy = metric_value(metrics, "accuracy")
    overall_score = accuracy * 100 if isinstance(accuracy, (int, float)) else None

    breakdowns = build_breakdowns_from_dimensions(avg_metrics, name_maps)
    if not breakdowns:
        breakdowns = build_breakdowns_from_legacy(avg_metrics, name_maps)
    derived_legacy_fields = derive_legacy_fields_from_dimensions(avg_metrics, breakdowns)

    entry = {
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
        "result_path": result_path,
        "source_exp_dir": config.get("source_exp_dir"),
    }

    for key in LEGACY_METRIC_FIELDS:
        value = metric_value(metrics, key)
        if value is None:
            value = derived_legacy_fields.get(key)
        if value is not None:
            entry[key] = value

    if breakdowns:
        entry["breakdowns"] = breakdowns

    return entry


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
    parser.add_argument("--results-root", default=Path("results"), type=Path)
    parser.add_argument("--force", action="store_true", help="Overwrite even when target is same or newer.")
    parser.add_argument(
        "--git-branch",
        action="store_true",
        help="Create or switch to a result branch inside ToMResults before syncing.",
    )
    parser.add_argument(
        "--branch-name",
        help="Result branch name. Defaults to results/<model>-<dataset>-<run_timestamp>.",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Commit the synced ToMResults files after syncing.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the committed ToMResults branch to origin.",
    )
    parser.add_argument(
        "--open-pr",
        action="store_true",
        help="Create a GitHub PR from the pushed result branch to the base branch.",
    )
    parser.add_argument(
        "--dry-run-pr",
        action="store_true",
        help="Print the GitHub PR payload without calling the GitHub API.",
    )
    parser.add_argument(
        "--github-token-env",
        default="GITHUB_TOKEN",
        help="Environment variable name that contains the GitHub token.",
    )
    parser.add_argument(
        "--repo",
        default="TomTraining/ToMResults",
        help="GitHub repository full name.",
    )
    parser.add_argument(
        "--base-branch",
        default="main",
        help="GitHub PR base branch.",
    )
    args = parser.parse_args()
    if args.push and not args.commit:
        parser.error("--push requires --commit")
    if args.open_pr and not args.push:
        parser.error("--open-pr requires --push")
    return args


def main():
    args = parse_args()
    tomresults_dir = args.tomresults_dir.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve()

    source_exp_dir = find_latest_valid_exp(results_root, args.dataset, args.model)
    run_timestamp = exp_timestamp(source_exp_dir)
    branch_name = args.branch_name or default_branch_name(args.model, args.dataset, run_timestamp)
    result_path = f"results/{args.model}/{args.dataset}"
    commit_hash = None
    pushed_branch = None

    print(f"Branch name: {branch_name if args.git_branch else 'not requested'}")
    if args.git_branch:
        checkout_result_branch(tomresults_dir, branch_name)

    copied, _target_dir = sync_result(
        source_exp_dir,
        tomresults_dir,
        args.dataset,
        args.model,
        args.force,
    )
    if copied:
        rebuild_leaderboard(tomresults_dir)
    else:
        print("No result update; leaderboard.json was not rebuilt.")

    if args.commit:
        if copied:
            commit_hash = commit_synced_files(
                tomresults_dir,
                args.model,
                args.dataset,
                run_timestamp,
            )
        else:
            print("Commit skipped: no newer result was synced.")
    else:
        print("Commit: not requested")

    if args.push:
        if commit_hash is None:
            print("Push skipped: no commit was created.")
        else:
            pushed_branch = push_result_branch(tomresults_dir, branch_name if args.git_branch else None)
    else:
        print("Push: not requested")

    should_prepare_pr = args.open_pr or args.dry_run_pr
    if should_prepare_pr:
        if not copied:
            print("PR skipped: no result update was synced.")
        elif args.commit and commit_hash is None:
            print("PR skipped: no commit was created.")
        elif args.open_pr and pushed_branch is None:
            print("PR skipped: no branch was pushed.")
        else:
            head_branch = pushed_branch or branch_name
            payload = build_pr_payload(
                args.repo,
                args.base_branch,
                head_branch,
                args.model,
                args.dataset,
                run_timestamp,
                result_path,
                source_exp_dir.name,
            )
            if args.dry_run_pr:
                print_pr_dry_run(payload)
            if args.open_pr:
                token = os.environ.get(args.github_token_env)
                if not token:
                    raise RuntimeError(
                        f"--open-pr requires a GitHub token in environment variable {args.github_token_env}"
                    )
                create_github_pr(payload, token)

    if args.git_branch:
        print(f"GitHub PR hint: create a PR from {branch_name} to main.")
    else:
        print("Next step: run git status / commit / push inside ToMResults when ready.")


if __name__ == "__main__":
    main()
