from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from src import runner

from .data import load_standardized_data, load_task_config, analyze_question_types
from .judge import judge_repeat
from .metrics import aggregate_metrics
from .paths import build_experiment_dir, find_latest_experiment_dir, save_config
from .prediction import predict_records
from .storage import read_prediction_file, save_metrics, write_prediction_file

logger = logging.getLogger(__name__)


def run_prediction_stage(
    task_config: Dict[str, Any],
    experiment_config: Dict[str, Any],
    output_dir: Path,
) -> Path:
    # predict 阶段只生成 prediction.jsonl，不做 judge 和 metric。
    logger.info("=" * 80)
    logger.info("STAGE 1: PREDICTION")
    logger.info("=" * 80)

    samples = load_standardized_data(task_config, experiment_config)

    # 分析题型分布
    type_info = analyze_question_types(samples)
    max_samples = experiment_config.get("max_samples", 0)

    if max_samples > 0:
        logger.info(f"Dataset: {task_config['dataset']} ({type_info['total']}/{type_info['total']} samples - limited by max_samples={max_samples})")
    else:
        logger.info(f"Dataset: {task_config['dataset']} ({type_info['total']} samples - full dataset)")

    logger.info(f"Question Type: {type_info['question_type']}")

    # 展示第一个样本的输入示例
    if samples:
        example = samples[0]
        logger.info("\n[INPUT EXAMPLE - Sample #0]")
        logger.info(f"Story: {example['story'][:200]}...")
        logger.info(f"Question: {example['question']}")
        logger.info(f"Correct answers: {example['answer']['correct_answers']}")
        logger.info(f"Wrong answers: {example['answer']['wrong_answers'][:2] if len(example['answer']['wrong_answers']) > 2 else example['answer']['wrong_answers']}")

    client = runner.create_content_client(experiment_config["llm_config"], task_config)
    logger.info(f"\nStarting batch prediction (repeats={experiment_config['repeats']})...")

    records = predict_records(samples, task_config["dataset"], client, experiment_config["repeats"])

    # 展示第一个预测结果的输出示例
    if records:
        example_record = records[0]
        logger.info("\n[OUTPUT EXAMPLE - Sample #0, Repeat 0]")
        logger.info(f"Prompt type: {example_record['prompt_type']}")
        logger.info(f"Prompt: {example_record['prompt'][:300]}...")
        logger.info(f"Model response: {json.dumps(example_record['pred']['content'], ensure_ascii=False)}")
        if example_record['pred'].get('reasoning'):
            logger.info(f"Reasoning: {example_record['pred']['reasoning'][:200]}...")

    prediction_path = write_prediction_file(output_dir, records)
    logger.info(f"\n✓ Saved {len(records)} predictions to: {prediction_path}")
    logger.info("=" * 80 + "\n")
    return prediction_path


def run_metric_stage(
    task_config: Dict[str, Any],
    experiment_config: Dict[str, Any],
    output_dir: Path,
) -> Path:
    logger.info("=" * 80)
    logger.info("STAGE 2: JUDGE & METRICS")
    logger.info("=" * 80)

    prediction_path = output_dir / "prediction.jsonl"
    if not prediction_path.exists():
        raise FileNotFoundError(f"prediction.jsonl not found in {output_dir}")

    records = read_prediction_file(prediction_path)
    logger.info(f"Loaded {prediction_path}")

    # MCQ 题走 \boxed{} 提取规则判分，只有存在 open 题时才需要创建 judge client。
    has_open = any(record["prompt_type"] == "open" for record in records)
    if has_open:
        judge_config = experiment_config.get("judge_config") or experiment_config["llm_config"]
        judge_client = runner.create_llm_client(judge_config, task_config)
    else:
        judge_client = None
        logger.info("All records are MCQ: rule-based grading via \\boxed{} extraction, judge client skipped.")
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["repeat"])].append(record)

    # metric 按 repeat 独立执行 judge 和聚合，最后统一写入 all_metrics + avg_metrics。
    all_metrics: List[Dict[str, Any]] = []
    for repeat in sorted(grouped):
        logger.info(f"\n--- Repeat {repeat + 1}/{len(grouped)} ---")
        repeat_records = sorted(grouped[repeat], key=lambda item: int(item["sample_index"]))

        # 展示第一个样本的判分输入：open 展示 judge prompt，MCQ 展示提取规则比对信息。
        if repeat == 0 and repeat_records:
            example = repeat_records[0]
            logger.info("\n[JUDGE INPUT EXAMPLE - Sample #0]")
            logger.info(f"Prompt type: {example['prompt_type']}")
            if example["prompt_type"] == "open":
                from .judge import judge_prompt
                judge_input = judge_prompt(example)
                logger.info(f"Judge prompt: {judge_input[:400]}...")
            else:
                from .prompts import extract_prediction_from_text
                content = (example.get("pred") or {}).get("content") or ""
                extracted = extract_prediction_from_text(example["prompt_type"], str(content))
                logger.info(f"Rule-based grading: extracted={extracted} vs correct={example['correct_letters']}")

        per_sample_results = judge_repeat(repeat_records, judge_client)

        # 展示第一个样本的判分输出
        if repeat == 0 and per_sample_results:
            example_result = per_sample_results[0]
            logger.info("\n[JUDGE OUTPUT EXAMPLE - Sample #0]")
            logger.info(f"Is correct: {example_result['is_correct']}")
            if example_result.get('error_reason'):
                logger.info(f"Error reason: {example_result['error_reason']}")

        metrics = aggregate_metrics(task_config["dataset"], repeat_records, per_sample_results)
        all_metrics.append(metrics)
        logger.info(
            f"✓ Run {repeat + 1}: Accuracy={metrics['accuracy']:.4f}, "
            f"Correct={metrics['correct']}/{metrics['total']}"
        )

    metrics_path = save_metrics(output_dir, all_metrics)
    logger.info(f"\n✓ Saved metrics to: {metrics_path}")
    logger.info("=" * 80 + "\n")
    return metrics_path


def run_standardized_qa_task(config_path: str) -> None:
    task_config = load_task_config(config_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", default="experiment_config.yaml")
    parser.add_argument("--stage", choices=["predict", "metric", "all"], default="all")
    parser.add_argument("--exp-dir", default=None)
    args = parser.parse_args()

    experiment_config = runner.load_experiment_config(args.experiment_config)

    # 配置日志：同时输出到控制台和文件
    from datetime import datetime
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    model_name = experiment_config["llm_config"]["model_name"]
    dataset_name = task_config["dataset"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"实验_{dataset_name}_{model_name}_{timestamp}.log"

    # 清除之前的 handlers，避免重复配置
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # 配置日志格式和输出
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),  # 输出到控制台
            logging.FileHandler(log_file, encoding='utf-8')  # 输出到文件
        ]
    )

    logger.info(f"日志文件: {log_file}")

    # 美化 stage 显示
    stage_display = {
        "predict": "Prediction",
        "metric": "Judge & Metrics",
        "all": "Prediction + Judge & Metrics"
    }

    logger.info("\n" + "=" * 80)
    logger.info(f"🚀 Starting Evaluation")
    logger.info("=" * 80)
    logger.info(f"Config: {args.experiment_config}")
    logger.info(f"Stage: {stage_display[args.stage]}")
    logger.info("=" * 80 + "\n")

    model_name = experiment_config["llm_config"]["model_name"]
    if args.stage == "metric" and args.exp_dir is None:
        target_output_dir = find_latest_experiment_dir(
            task_config["dataset"],
            model_name,
            experiment_config["results_path"],
        )
    else:
        target_output_dir = build_experiment_dir(
            task_config["dataset"],
            model_name,
            experiment_config["results_path"],
            exp_dir=args.exp_dir,
            create=True,
        )

    logger.info(f"Output directory: {target_output_dir}\n")
    save_config(target_output_dir, task_config, experiment_config)

    if args.stage in {"predict", "all"}:
        run_prediction_stage(task_config, experiment_config, target_output_dir)
    if args.stage in {"metric", "all"}:
        run_metric_stage(task_config, experiment_config, target_output_dir)
