from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from report.utils import find_experiment_dir, load_bad_cases, load_metrics_payload, load_yaml, parse_model_entry
from src.llm.content_client import ContentClient


BAD_CASE_ANALYSIS_PROMPT = """You are analyzing a failed Theory-of-Mind QA example.

Story and question prompt shown to the model:
{prompt}

Sample metadata:
{meta}

Gold answer:
{gold_answer}

Model prediction:
{prediction}

Model reasoning excerpt:
{reasoning}

Please answer in three short parts:
[Dimension]
[Failure Reason]
[Improvement Suggestion]"""


def _flatten_extra_metrics(avg_metrics: Dict[str, Any]) -> List[tuple[str, Any]]:
    rows: List[tuple[str, Any]] = []
    for key, value in sorted(avg_metrics.items()):
        if key in {"accuracy", "correct", "total", "per_sample_results"}:
            continue
        if isinstance(value, dict):
            for sub_key, sub_value in sorted(value.items()):
                rows.append((f"{key}.{sub_key}", sub_value))
        else:
            rows.append((key, value))
    return rows


def _prediction_text(case: Dict[str, Any]) -> str:
    return json.dumps(case.get("raw_prediction"), ensure_ascii=False)


def _build_analysis_prompt(case: Dict[str, Any]) -> str:
    reasoning = ((case.get("pred") or {}).get("reasoning") or "")[:1000]
    return BAD_CASE_ANALYSIS_PROMPT.format(
        prompt=case.get("prompt", ""),
        meta=json.dumps(case.get("meta", {}), ensure_ascii=False, indent=2),
        gold_answer=case.get("gold_answer", ""),
        prediction=_prediction_text(case),
        reasoning=reasoning or "(empty)",
    )


def _print_basic_metrics(dataset: str, model_display: str, model_metrics: Dict[str, Any], baseline_display: Optional[str], baseline_metrics: Optional[Dict[str, Any]]) -> None:
    print("=" * 72)
    print(f"Dataset: {dataset}")
    print(f"Model: {model_display}")
    if baseline_display:
        print(f"Baseline: {baseline_display}")
    print("=" * 72)
    model_acc = model_metrics.get("accuracy")
    baseline_acc = baseline_metrics.get("accuracy") if baseline_metrics else None
    if baseline_display and baseline_acc is not None and model_acc is not None:
        print(f"Accuracy: {model_acc:.4f}  |  Baseline: {baseline_acc:.4f}  |  Diff: {model_acc - baseline_acc:+.4f}")
    elif model_acc is not None:
        print(f"Accuracy: {model_acc:.4f}")
    print(f"Correct: {model_metrics.get('correct', '-')}")
    print(f"Total: {model_metrics.get('total', '-')}")


def _print_extra_metrics(model_metrics: Dict[str, Any], baseline_metrics: Optional[Dict[str, Any]]) -> None:
    rows = _flatten_extra_metrics(model_metrics)
    if not rows:
        print("\nNo extra metrics.")
        return
    print("\nExtra metrics:")
    for name, value in rows:
        if baseline_metrics:
            baseline_value = baseline_metrics
            for part in name.split("."):
                if not isinstance(baseline_value, dict):
                    baseline_value = None
                    break
                baseline_value = baseline_value.get(part)
            if isinstance(value, (int, float)) and isinstance(baseline_value, (int, float)):
                print(f"- {name}: {value:.4f}  |  baseline {baseline_value:.4f}  |  diff {value - baseline_value:+.4f}")
            else:
                print(f"- {name}: {value}")
        else:
            print(f"- {name}: {value}")


def _print_bad_cases(bad_cases: List[Dict[str, Any]], analyses: List[Optional[str]]) -> None:
    print(f"\nBad cases: {len(bad_cases)}")
    for index, (case, analysis) in enumerate(zip(bad_cases, analyses), start=1):
        print("-" * 72)
        print(f"[{index}] Tier {case['_tier']} | Wrong repeats {case['_wrong_count']}/{case['_max_repeat']}")
        print(f"Group: {case['_group_display']}")
        print(f"Gold: {case.get('gold_answer', '')}")
        print(f"Prediction: {_prediction_text(case)}")
        print(f"Judge: {(case.get('judge_result') or {}).get('reason', '')}")
        if analysis:
            print(analysis)


def _save_markdown_report(
    output_dir: str,
    dataset: str,
    model_display: str,
    baseline_display: Optional[str],
    model_metrics: Dict[str, Any],
    baseline_metrics: Optional[Dict[str, Any]],
    bad_cases: List[Dict[str, Any]],
    analyses: List[Optional[str]],
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = Path(output_dir) / dataset / model_display
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / f"{timestamp}.md"

    lines = [
        "# Bad Case Analysis",
        "",
        f"- Dataset: {dataset}",
        f"- Model: {model_display}",
    ]
    if baseline_display:
        lines.append(f"- Baseline: {baseline_display}")
    lines.append("")
    lines.extend(
        [
            "## Basic Metrics",
            "",
            f"- Accuracy: {model_metrics.get('accuracy', '-')}",
            f"- Correct: {model_metrics.get('correct', '-')}",
            f"- Total: {model_metrics.get('total', '-')}",
            "",
        ]
    )
    if baseline_metrics:
        lines.extend(
            [
                "## Baseline Metrics",
                "",
                f"- Accuracy: {baseline_metrics.get('accuracy', '-')}",
                f"- Correct: {baseline_metrics.get('correct', '-')}",
                f"- Total: {baseline_metrics.get('total', '-')}",
                "",
            ]
        )

    lines.extend(["## Extra Metrics", ""])
    for name, value in _flatten_extra_metrics(model_metrics):
        lines.append(f"- {name}: {value}")
    lines.append("")

    lines.extend(["## Bad Cases", ""])
    for index, (case, analysis) in enumerate(zip(bad_cases, analyses), start=1):
        lines.extend(
            [
                f"### Case {index}",
                "",
                f"- Tier: {case['_tier']}",
                f"- Group: {case['_group_display']}",
                f"- Wrong repeats: {case['_wrong_count']}/{case['_max_repeat']}",
                f"- Gold: {case.get('gold_answer', '')}",
                f"- Prediction: {_prediction_text(case)}",
                "",
                "```text",
                case.get("prompt", ""),
                "```",
                "",
            ]
        )
        if analysis:
            lines.extend([analysis, ""])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "report_config.yaml")
    config = load_yaml(config_path)

    model_name, model_display = parse_model_entry(config["model"])
    baseline_entry = config.get("baseline")
    baseline_name = baseline_display = None
    if baseline_entry:
        baseline_name, baseline_display = parse_model_entry(baseline_entry)

    datasets = [config["dataset"]] if config.get("dataset") else []
    if not datasets:
        results_root = Path(str(config.get("results_dir", "results")))
        datasets = [path.name for path in sorted(results_root.iterdir()) if path.is_dir()]

    llm_client = None
    if not config.get("no_llm_analysis", False):
        llm_cfg = config.get("llm") or {}
        if llm_cfg.get("api_url"):
            llm_client = ContentClient.from_config(llm_cfg)

    for dataset in datasets:
        model_exp_dir = find_experiment_dir(str(config.get("results_dir", "results")), dataset, model_name)
        if model_exp_dir is None:
            print(f"Skip {dataset}: no experiment for {model_name}")
            continue
        model_payload = load_metrics_payload(model_exp_dir)
        model_metrics = model_payload.get("avg_metrics", {})

        baseline_metrics = None
        if baseline_name:
            baseline_exp_dir = find_experiment_dir(str(config.get("results_dir", "results")), dataset, baseline_name)
            if baseline_exp_dir is not None:
                baseline_metrics = load_metrics_payload(baseline_exp_dir).get("avg_metrics", {})

        _print_basic_metrics(dataset, model_display, model_metrics, baseline_display, baseline_metrics)
        _print_extra_metrics(model_metrics, baseline_metrics)

        bad_cases = load_bad_cases(
            model_exp_dir,
            limit=int((config.get("bad_cases") or {}).get("n", 10)),
            seed=int((config.get("bad_cases") or {}).get("seed", 42)),
        )
        analyses: List[Optional[str]] = [None] * len(bad_cases)
        if llm_client and bad_cases:
            prompts = [_build_analysis_prompt(case) for case in bad_cases]
            responses = llm_client.batch_generate(prompts)
            analyses = [response.content if response and response.content is not None else None for response in responses]

        _print_bad_cases(bad_cases, analyses)

        if config.get("output_report", True):
            output_path = _save_markdown_report(
                output_dir=str(config.get("output_dir", "analysis")),
                dataset=dataset,
                model_display=model_display,
                baseline_display=baseline_display,
                model_metrics=model_metrics,
                baseline_metrics=baseline_metrics,
                bad_cases=bad_cases,
                analyses=analyses,
            )
            print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
