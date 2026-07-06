from __future__ import annotations

import argparse
import json
import logging
import time
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
        logger.info(f"Dataset: {task_config['dataset']} ({type_info['total']} samples - limited by max_samples={max_samples})")
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

    logger.info(f"\nStarting batch prediction (repeats={experiment_config['repeats']})...")

    if experiment_config.get("agent_mode"):
        records = _predict_via_agent(samples, task_config, experiment_config, output_dir)
    else:
        client = runner.create_content_client(experiment_config["llm_config"])
        records = predict_records(
            samples,
            task_config["dataset"],
            client,
            experiment_config["repeats"],
            protocol=experiment_config.get("protocol"),
        )

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


def _predict_via_agent(
    samples: List[Dict[str, Any]],
    task_config: Dict[str, Any],
    experiment_config: Dict[str, Any],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """agent 模式预测:本地拉起统一运行时(agents/server.py)加载参赛方 solution.py → 发题 → 写 efficiency.json。

    提交逻辑:参赛方仓库被拉到 agents/ 下(含 solution.py),agent.solution_dir 指向它。框架在评测前
    起 agents/server.py(注入 SOLUTION_DIR + 从 api_url 解析的 PORT)、等 /health 就绪、评完关掉。

    model 凭证(我们部署的统一后端)随每条请求体的 `model` 字段下发给 agent;框架不中转模型请求,
    token 口径回归我们部署的后端侧。这里 efficiency.json 只记框架能直接看到的墙钟与
    agent 端点调用数/违约数。效率只记录、不进排名。
    """
    from src.llm import AgentClient

    from .agent_launcher import launch_local_agent

    agent_config = experiment_config.get("agent_config") or {}
    llm_config = experiment_config["llm_config"]
    if not agent_config.get("solution_dir"):
        raise ValueError("agent 配置缺少 solution_dir(参赛方仓库根目录,含 solution.py)。")

    client = AgentClient.from_config(agent_config, llm_config)

    # 本地拉起统一运行时:起 agents/server.py、等就绪、评完关掉。
    with launch_local_agent(agent_config, client):
        wall_start = time.time()
        records = predict_records(
            samples,
            task_config["dataset"],
            client,
            experiment_config["repeats"],
            protocol=None,
            agent_mode=True,
        )
        wall_clock = time.time() - wall_start

    total = len(samples) or 1
    predict_calls = client.get_usage().total_calls
    efficiency: Dict[str, Any] = {
        "dataset": task_config["dataset"],
        "num_samples": len(samples),
        "wall_clock_seconds": round(wall_clock, 3),
        "seconds_per_sample": round(wall_clock / total, 3),
        # agent /predict 侧的调用数(一条样本一次;失败=重试耗尽/不可重试)。
        "agent_predict_calls": predict_calls,
        "agent_predict_failed": client.get_usage().failed_calls,
        # prediction 违反严格格式(mcq_single 单字符 / mcq_multi 字母数组等)的样本数。
        "format_violations": client.format_violations,
    }

    efficiency_path = output_dir / "efficiency.json"
    efficiency_path.write_text(json.dumps(efficiency, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"✓ 效率指标 -> {efficiency_path}: {json.dumps(efficiency, ensure_ascii=False)}")
    return records


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

    # 协议优先从记录里读(兼容单独跑 stage=metric)，兜底用 experiment_config。
    protocol = (records[0].get("protocol") if records else None) or experiment_config.get("protocol")

    # MCQ 题走 \boxed{} 提取规则判分，只有存在 open 题时才需要按 open_judge 模式建上下文。
    # open_judge 模式 + judge1/rubric 参数来自本数据集 config.yaml。
    from .open_judge import DEFAULT_OPEN_JUDGE, build_open_ctx
    open_mode = task_config.get("open_judge", DEFAULT_OPEN_JUDGE)
    has_open = any(record["prompt_type"] == "open" for record in records)
    if has_open:
        task_dir = Path(f"tasks/{task_config['dataset']}")
        open_ctx = build_open_ctx(task_config, open_mode, task_dir=task_dir)
        logger.info(f"Open QA judging: mode={open_mode} (judge_clients={len(open_ctx.judge_clients)}).")
    else:
        open_ctx = None
        logger.info("All records are MCQ: rule-based grading via \\boxed{} extraction, open judge skipped.")

    # del_tom:把同一 sample 的多次 repeat 折叠成「每样本一条」后多数投票，单次聚合。
    from .protocols import is_voting_protocol
    if is_voting_protocol(protocol):
        from .voting import vote_collapse
        logger.info(f"Protocol '{protocol}': majority voting across {len(records)} predictions...")
        voted_records, voted_results = vote_collapse(records, open_ctx)
        metrics = aggregate_metrics(task_config["dataset"], voted_records, voted_results)
        all_metrics = [metrics]
        logger.info(
            f"✓ Voted: Accuracy={metrics['accuracy']:.4f}, "
            f"Correct={metrics['correct']}/{metrics['total']}"
        )
        metrics_path = save_metrics(output_dir, all_metrics)
        logger.info(f"\n✓ Saved metrics to: {metrics_path}")
        logger.info("=" * 80 + "\n")
        return metrics_path

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
                # judge_prompt 仅适用于 llm_simple；f1/rubric 模式只标注模式名,避免触发 assert。
                logger.info(f"Open judge mode: {open_mode}")
                if open_mode == "llm_simple":
                    from .judge import judge_prompt
                    judge_input = judge_prompt(example)
                    logger.info(f"Judge prompt: {judge_input[:400]}...")
            else:
                from .prompts import extract_prediction_from_text
                content = (example.get("pred") or {}).get("content") or ""
                extracted = extract_prediction_from_text(
                    example["prompt_type"], str(content), example.get("extractor", "legacy")
                )
                logger.info(f"Rule-based grading: extracted={extracted} vs correct={example['correct_letters']}")

        per_sample_results = judge_repeat(repeat_records, open_ctx=open_ctx)

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


def run_visualize_stage(
    task_config: Dict[str, Any],
    experiment_config: Dict[str, Any],
    output_dir: Path,
) -> Path:
    # visualize 阶段只读 output_dir/metrics.json，按数据集分组自动出图（数据集无关）。
    logger.info("=" * 80)
    logger.info("STAGE 3: VISUALIZE")
    logger.info("=" * 80)

    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found in {output_dir}; 请先跑 stage=metric 或 all。")

    from src.visualization import load_metrics, render_single

    payload = load_metrics(metrics_path)
    model_name = experiment_config["llm_config"]["model_name"]
    figures_dir = Path(experiment_config["figures_path"]) / task_config["dataset"] / model_name
    figures_dir.mkdir(parents=True, exist_ok=True)

    written = render_single(payload, figures_dir, prefix="")
    logger.info(f"✓ 生成 {len(written)} 张图 -> {figures_dir}")
    for path in written:
        logger.info(f"    {path}")
    logger.info("=" * 80 + "\n")
    return figures_dir


def run_standardized_qa_task(config_path: str) -> None:
    task_config = load_task_config(config_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", default="experiment_config.yaml")
    parser.add_argument("--exp-dir", default=None)
    args = parser.parse_args()

    experiment_config = runner.load_experiment_config(args.experiment_config)
    # stage 由配置驱动(predict/metric/visualize/all)。
    stage = experiment_config["stage"]
    if stage not in {"predict", "metric", "visualize", "all"}:
        raise ValueError(f"experiment_config.stage must be predict/metric/visualize/all, got {stage!r}")

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
        "visualize": "Visualize",
        "all": "Prediction + Judge & Metrics + Visualize"
    }

    logger.info("\n" + "=" * 80)
    logger.info(f"🚀 Starting Evaluation")
    logger.info("=" * 80)
    logger.info(f"Config: {args.experiment_config}")
    logger.info(f"Stage: {stage_display[stage]}")
    logger.info(f"Protocol: {experiment_config.get('protocol') or '(none / legacy)'}")
    logger.info("=" * 80 + "\n")

    model_name = experiment_config["llm_config"]["model_name"]
    # metric / visualize 只读已有产物，默认复用最新 exp 目录，不新建。
    if stage in {"metric", "visualize"} and args.exp_dir is None:
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

    if stage in {"predict", "all"}:
        run_prediction_stage(task_config, experiment_config, target_output_dir)
    if stage in {"metric", "all"}:
        run_metric_stage(task_config, experiment_config, target_output_dir)
    if stage in {"visualize", "all"}:
        run_visualize_stage(task_config, experiment_config, target_output_dir)
