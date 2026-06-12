"""
Dataset Tables Generator

从 results/ 目录读取 metrics.json 文件，为每个数据集生成详细表格：
1. 基础指标.md：accuracy, correct, total
2. 其他指标.md：其他所有指标（包括字典类型的子指标）
3. 复制 config.json 到输出目录

支持通过 --config yaml 文件或命令行参数指定过滤条件（dataset / models）。
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from report.utils import parse_markdown_sections, parse_markdown_table


ModelRun = Dict[str, Optional[str]]


def resolve_exp_dir(model_dir: Path, exp_suffix: Optional[str], dataset_name: str, model_name: str) -> Optional[Path]:
    """Resolve the experiment directory for one dataset/model pair."""
    if exp_suffix:
        exp_dir = model_dir / f"exp_{exp_suffix}"
        if not exp_dir.exists():
            raise FileNotFoundError(
                f"指定的实验目录不存在: {exp_dir}\n"
                f"  数据集: {dataset_name}  模型: {model_name}\n"
                f"  可用实验: {[d.name for d in sorted(model_dir.glob('exp_*'))]}"
            )
        return exp_dir

    exp_dirs = sorted(model_dir.glob("exp_*"))
    if not exp_dirs:
        return None
    return exp_dirs[-1]


def collect_metrics(
    results_dir: str,
    exp_suffix: str = None,
    dataset_filter: Optional[Union[str, List[str]]] = None,
    models_filter: Optional[List[str]] = None,
    model_runs: Optional[List[ModelRun]] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """收集所有数据集和模型的 metrics

    Args:
        results_dir: results 目录路径
        exp_suffix: 实验时间后缀，用于筛选特定实验结果。如果为 None，自动选择每个数据集最新的实验
        dataset_filter: 只收集该数据集（None 表示收集全部）
        models_filter: 只收集这些模型（None 表示收集全部）
        model_runs: 每个模型运行的显式绑定配置，支持为同一模型指定不同 exp_suffix/display

    Returns:
        {dataset_name: {model_or_display_name: metrics_data}}
    """
    results_path = Path(results_dir)
    metrics_data = {}

    for dataset_dir in results_path.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
            continue

        dataset_name = dataset_dir.name

        # 过滤数据集
        if dataset_filter:
            allowed = {dataset_filter} if isinstance(dataset_filter, str) else set(dataset_filter)
            if dataset_name not in allowed:
                continue

        metrics_data[dataset_name] = {}

        if model_runs is not None:
            for run in model_runs:
                model_name = run["name"]
                display_name = run.get("display") or model_name
                run_exp_suffix = run.get("exp_suffix") or exp_suffix
                model_dir = dataset_dir / model_name
                if not model_dir.is_dir():
                    continue

                exp_dir = resolve_exp_dir(model_dir, run_exp_suffix, dataset_name, model_name)
                if exp_dir is None:
                    continue

                metrics_file = exp_dir / "metrics.json"
                if metrics_file.exists():
                    if display_name in metrics_data[dataset_name]:
                        raise ValueError(
                            f"数据集 [{dataset_name}] 中表格列名 [{display_name}] 重复。"
                            "同一个生成模型绑定多个 exp_suffix 时，请为每个条目设置不同 display。"
                        )
                    with open(metrics_file, 'r', encoding='utf-8') as f:
                        metrics_data[dataset_name][display_name] = json.load(f)
            continue

        for model_dir in dataset_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue

            model_name = model_dir.name

            # 过滤模型
            if models_filter and model_name not in models_filter:
                continue

            exp_dir = resolve_exp_dir(model_dir, exp_suffix, dataset_name, model_name)
            if exp_dir is None:
                continue

            metrics_file = exp_dir / "metrics.json"

            if metrics_file.exists():
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    metrics_data[dataset_name][model_name] = json.load(f)

    return metrics_data


def get_all_metrics_names(metrics_data: Dict[str, Dict[str, Dict[str, Any]]]) -> List[str]:
    """获取所有出现的指标名称（标量类型）

    Args:
        metrics_data: 收集的 metrics 数据

    Returns:
        所有标量指标名称的列表
    """
    metric_names = set()

    for dataset_metrics in metrics_data.values():
        for model_metrics in dataset_metrics.values():
            if "avg_metrics" in model_metrics:
                for key, value in model_metrics["avg_metrics"].items():
                    if not isinstance(value, dict):
                        metric_names.add(key)

    return sorted(metric_names)


def get_dict_metrics(metrics_data: Dict[str, Dict[str, Dict[str, Any]]]) -> List[str]:
    """获取所有字典类型的指标名称

    Args:
        metrics_data: 收集的 metrics 数据

    Returns:
        所有字典类型指标名称的列表
    """
    dict_metrics = set()

    for dataset_metrics in metrics_data.values():
        for model_metrics in dataset_metrics.values():
            if "avg_metrics" in model_metrics:
                for key, value in model_metrics["avg_metrics"].items():
                    if isinstance(value, dict):
                        dict_metrics.add(key)

    return sorted(dict_metrics)


def format_value(value: Any) -> str:
    """格式化指标值

    Args:
        value: 指标值

    Returns:
        格式化后的字符串
    """
    if isinstance(value, float):
        return f"{value:.4f}"
    elif isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    else:
        return str(value)


def merge_table_data(
    existing: Dict[str, Dict[str, str]],
    new_models: List[str],
    new_rows: Dict[str, Dict[str, str]],
) -> tuple:
    """将新数据合并进已有表格数据

    Args:
        existing: 已有表格数据 {row_key: {model: value}}
        new_models: 新数据涉及的模型列表
        new_rows: 新数据 {row_key: {model: value}}

    Returns:
        (merged_data, all_models) — 合并后的数据及完整模型列表
    """
    # 收集已有的所有模型列
    all_models_set: Set[str] = set()
    for row_vals in existing.values():
        all_models_set.update(row_vals.keys())
    all_models_set.update(new_models)
    # 保持原有顺序，新 model 追加到末尾
    existing_models = [m for m in sorted(all_models_set) if m not in new_models]
    all_models = existing_models + [m for m in new_models if m in all_models_set]
    # 按原有列顺序重排（先从已有表格取列顺序）
    old_order: List[str] = []
    for row_vals in existing.values():
        for m in row_vals:
            if m not in old_order:
                old_order.append(m)
    merged_models: List[str] = []
    for m in old_order:
        if m not in new_models:
            merged_models.append(m)
    for m in new_models:
        merged_models.append(m)
    for m in all_models:
        if m not in merged_models:
            merged_models.append(m)

    # 合并行数据
    merged: Dict[str, Dict[str, str]] = {}
    all_row_keys: List[str] = list(existing.keys())
    for row_key in new_rows:
        if row_key not in all_row_keys:
            all_row_keys.append(row_key)

    for row_key in all_row_keys:
        merged[row_key] = {}
        for model in merged_models:
            if row_key in new_rows and model in new_rows[row_key]:
                merged[row_key][model] = new_rows[row_key][model]
            elif row_key in existing and model in existing[row_key]:
                merged[row_key][model] = existing[row_key][model]
            else:
                merged[row_key][model] = "-"

    return merged, merged_models


def build_table_lines(header_col0: str, models: List[str], rows: Dict[str, Dict[str, str]], row_order: List[str]) -> List[str]:
    """将合并后的数据渲染成 Markdown 表格行"""
    lines = []
    lines.append("| " + header_col0 + " | " + " | ".join(models) + " |")
    lines.append("|" + "|".join(["---"] * (len(models) + 1)) + "|")
    for row_key in row_order:
        vals = rows.get(row_key, {})
        row = [row_key] + [vals.get(m, "-") for m in models]
        lines.append("| " + " | ".join(row) + " |")
    return lines


def generate_basic_metrics_table(
    dataset_name: str,
    models: List[str],
    metrics_data: Dict[str, Dict[str, Dict[str, Any]]],
    existing_path: Optional[Path] = None,
    overwrite: bool = True,
) -> str:
    """生成基础指标表格：accuracy, correct, total

    Args:
        dataset_name: 数据集名称
        models: 模型列表（显示名）
        metrics_data: 收集的 metrics 数据
        existing_path: 已有表格路径，存在时增量合并
        overwrite: True 则覆盖同名模型的已有数据，False 则保留

    Returns:
        Markdown 表格字符串
    """
    basic_metrics = ["accuracy", "correct", "total"]

    # 构建新数据
    new_rows: Dict[str, Dict[str, str]] = {}
    for metric in basic_metrics:
        new_rows[metric] = {}
        for model in models:
            if model in metrics_data.get(dataset_name, {}):
                value = metrics_data[dataset_name][model].get("avg_metrics", {}).get(metric, "")
                new_rows[metric][model] = format_value(value) if value != "" else "-"
            else:
                new_rows[metric][model] = "-"

    lines = [f"# {dataset_name} - 基础指标", ""]

    if existing_path and existing_path.exists():
        existing_table = parse_markdown_table(existing_path.read_text(encoding="utf-8"))
        existing_model_set: Set[str] = {col for row_vals in existing_table.values() for col in row_vals}
        # 不覆盖时，只合并新增模型
        models_to_merge = models if overwrite else [m for m in models if m not in existing_model_set]
        filtered_rows = {
            metric: {m: v for m, v in vals.items() if m in models_to_merge}
            for metric, vals in new_rows.items()
        }
        merged, all_models = merge_table_data(existing_table, models_to_merge, filtered_rows)
        row_order = [m for m in basic_metrics if m in merged] + [m for m in merged if m not in basic_metrics]
        lines.extend(build_table_lines("指标 \\ 模型", all_models, merged, row_order))
    else:
        lines.extend([
            "| 指标 \\ 模型 | " + " | ".join(models) + " |",
            "|" + "|".join(["---"] * (len(models) + 1)) + "|",
        ])
        for metric in basic_metrics:
            row = [metric] + [new_rows.get(metric, {}).get(m, "-") for m in models]
            lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return "\n".join(lines)


def generate_other_metrics_table(
    dataset_name: str,
    models: List[str],
    metrics_data: Dict[str, Dict[str, Dict[str, Any]]],
    existing_path: Optional[Path] = None,
    overwrite: bool = True,
) -> str:
    """生成其他指标表格

    Args:
        dataset_name: 数据集名称
        models: 模型列表（显示名）
        metrics_data: 收集的 metrics 数据
        existing_path: 已有表格路径，存在时增量合并
        overwrite: True 则覆盖同名模型的已有数据，False 则保留

    Returns:
        Markdown 表格字符串
    """
    metric_names = get_all_metrics_names({dataset_name: metrics_data[dataset_name]})
    basic_metrics = ["accuracy", "correct", "total"]
    other_metrics = [m for m in metric_names if m not in basic_metrics]
    dict_metrics = get_dict_metrics({dataset_name: metrics_data[dataset_name]})

    # 解析已有文件（按 ## 分 section）
    existing_sections: Dict[str, Dict[str, Dict[str, str]]] = {}
    if existing_path and existing_path.exists():
        existing_sections = parse_markdown_sections(existing_path.read_text(encoding="utf-8"))

    # 从已有文件推断已有模型集合（任意 section 的列名均可）
    existing_model_set: Set[str] = set()
    for section_data in existing_sections.values():
        for row_vals in section_data.values():
            existing_model_set.update(row_vals.keys())

    models_to_merge = models if overwrite else [m for m in models if m not in existing_model_set]

    lines = [f"# {dataset_name} - 其他指标", ""]

    # --- 标量指标 section ---
    if other_metrics or "标量指标" in existing_sections:
        lines.extend(["## 标量指标", ""])
        new_scalar: Dict[str, Dict[str, str]] = {}
        for metric in other_metrics:
            new_scalar[metric] = {}
            for model in models_to_merge:
                if model in metrics_data.get(dataset_name, {}):
                    value = metrics_data[dataset_name][model].get("avg_metrics", {}).get(metric, "")
                    new_scalar[metric][model] = format_value(value) if value != "" else "-"
                else:
                    new_scalar[metric][model] = "-"

        if "标量指标" in existing_sections:
            merged, all_models = merge_table_data(existing_sections["标量指标"], models_to_merge, new_scalar)
            lines.extend(build_table_lines("指标 \\ 模型", all_models, merged, list(merged.keys())))
        else:
            lines.extend([
                "| 指标 \\ 模型 | " + " | ".join(models) + " |",
                "|" + "|".join(["---"] * (len(models) + 1)) + "|",
            ])
            for metric in other_metrics:
                row = [metric] + [new_scalar.get(metric, {}).get(m, "-") for m in models]
                lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # --- dict 指标 sections ---
    # 合并新数据有的 section 和已有文件有的 section
    all_dict_sections: List[str] = list(dict_metrics)
    for sec in existing_sections:
        if sec != "标量指标" and sec not in all_dict_sections:
            all_dict_sections.append(sec)

    for dict_metric in all_dict_sections:
        lines.extend([f"## {dict_metric}", ""])

        # 构建新数据的子键和行
        all_sub_keys: Set[str] = set()
        for model in models_to_merge:
            if model in metrics_data.get(dataset_name, {}):
                dv = metrics_data[dataset_name][model].get("avg_metrics", {}).get(dict_metric, {})
                if isinstance(dv, dict):
                    all_sub_keys.update(dv.keys())

        new_dict: Dict[str, Dict[str, str]] = {}
        for sub_key in sorted(all_sub_keys):
            new_dict[sub_key] = {}
            for model in models_to_merge:
                if model in metrics_data.get(dataset_name, {}):
                    dv = metrics_data[dataset_name][model].get("avg_metrics", {}).get(dict_metric, {})
                    new_dict[sub_key][model] = format_value(dv[sub_key]) if isinstance(dv, dict) and sub_key in dv else "-"
                else:
                    new_dict[sub_key][model] = "-"

        if dict_metric in existing_sections:
            merged, all_models = merge_table_data(existing_sections[dict_metric], models_to_merge, new_dict)
            lines.extend(build_table_lines("子指标 \\ 模型", all_models, merged, list(merged.keys())))
        elif new_dict:
            lines.extend([
                "| 子指标 \\ 模型 | " + " | ".join(models) + " |",
                "|" + "|".join(["---"] * (len(models) + 1)) + "|",
            ])
            for sub_key in sorted(all_sub_keys):
                row = [sub_key] + [new_dict.get(sub_key, {}).get(m, "-") for m in models]
                lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    if not other_metrics and not dict_metrics and not existing_sections:
        lines.append("该数据集没有其他指标。\n")

    return "\n".join(lines)


def generate_dataset_tables(
    results_dir: str = "results",
    output_dir: str = "tables",
    exp_suffix: str = None,
    dataset_filter: Optional[Union[str, List[str]]] = None,
    models_filter: Optional[List[str]] = None,
    model_display_names: Optional[Dict[str, str]] = None,
    model_runs: Optional[List[ModelRun]] = None,
) -> None:
    """生成所有数据集的指标表格

    Args:
        results_dir: results 目录路径
        output_dir: 输出目录路径
        exp_suffix: 实验时间后缀，用于筛选特定实验结果
        dataset_filter: 只处理该数据集（None 表示全部）
        models_filter: 只处理这些模型，使用目录名（None 表示全部）
        model_display_names: 模型显示名映射 {目录名: 显示名}，用于表格列名和 config 文件夹名
        model_runs: 显式模型运行列表，每个元素包含 name/display/exp_suffix
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    name_map = model_display_names or {}

    metrics_data = collect_metrics(results_dir, exp_suffix, dataset_filter, models_filter, model_runs=model_runs)

    if not metrics_data:
        print("没有找到任何评测结果。")
        return

    for dataset_name in sorted(metrics_data.keys()):
        if model_runs is not None:
            display_metrics_data = {dataset_name: metrics_data[dataset_name]}
            display_names = [
                (run.get("display") or run["name"])
                for run in model_runs
                if (run.get("display") or run["name"]) in metrics_data[dataset_name]
            ]
        else:
            dir_names = sorted(metrics_data[dataset_name].keys())

            # 将 metrics_data 中的模型 key 替换为显示名，供表格函数使用
            display_metrics_data = {
                dataset_name: {name_map.get(d, d): v for d, v in metrics_data[dataset_name].items()}
            }
            display_names = [name_map.get(d, d) for d in dir_names]

        dataset_report_dir = output_path / dataset_name
        dataset_report_dir.mkdir(parents=True, exist_ok=True)

        # 检测已有表格中是否存在同名模型，询问是否覆盖
        overwrite = True
        basic_md = dataset_report_dir / "基础指标.md"
        if basic_md.exists():
            existing_table = parse_markdown_table(basic_md.read_text(encoding="utf-8"))
            existing_models: Set[str] = {col for row_vals in existing_table.values() for col in row_vals}
            overlapping = [m for m in display_names if m in existing_models]
            if overlapping:
                _basic_keys = ["accuracy", "correct", "total"]
                for model in overlapping:
                    print(f"\n数据集 [{dataset_name}] 模型 [{model}] 已有结果：")
                    print(f"  {'指标':<12} {'旧值':>12} {'新值':>12}")
                    print(f"  {'-' * 38}")
                    for metric in _basic_keys:
                        old_val = existing_table.get(metric, {}).get(model, "-")
                        new_raw = display_metrics_data[dataset_name].get(model, {}).get("avg_metrics", {}).get(metric, "")
                        new_val = format_value(new_raw) if new_raw != "" else "-"
                        print(f"  {metric:<12} {old_val:>12} {new_val:>12}")
                try:
                    ans = input("\n是否覆盖以上模型的旧结果？(y/n，默认 n): ").strip().lower()
                except EOFError:
                    ans = "n"
                    print("n（非交互模式，默认不覆盖）")
                overwrite = (ans == "y")

        basic_metrics_content = generate_basic_metrics_table(dataset_name, display_names, display_metrics_data, existing_path=basic_md, overwrite=overwrite)
        basic_md.write_text(basic_metrics_content, encoding='utf-8')

        other_md = dataset_report_dir / "其他指标.md"
        other_metrics_content = generate_other_metrics_table(dataset_name, display_names, display_metrics_data, existing_path=other_md, overwrite=overwrite)
        other_md.write_text(other_metrics_content, encoding='utf-8')

        config_runs: List[ModelRun]
        if model_runs is not None:
            config_runs = [
                run for run in model_runs
                if (run.get("display") or run["name"]) in metrics_data[dataset_name]
            ]
        else:
            config_runs = [
                {"name": dir_name, "display": name_map.get(dir_name, dir_name), "exp_suffix": exp_suffix}
                for dir_name in dir_names
            ]

        for run in config_runs:
            dir_name = run["name"]
            display_name = run.get("display") or dir_name
            model_dir = results_path / dataset_name / dir_name
            run_exp_suffix = run.get("exp_suffix") or exp_suffix
            exp_dir = resolve_exp_dir(model_dir, run_exp_suffix, dataset_name, dir_name)
            config_file = exp_dir / "config.json" if exp_dir else model_dir / "config.json"

            if config_file.exists():
                output_model_dir = dataset_report_dir / display_name
                output_model_dir.mkdir(parents=True, exist_ok=True)
                with open(config_file, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                if exp_dir:
                    config_data["exp_dir"] = exp_dir.name
                config_data["model_dir"] = dir_name
                config_data["table_display"] = display_name
                with open(output_model_dir / "config.json", "w", encoding="utf-8") as f:
                    json.dump(config_data, f, ensure_ascii=False, indent=2)

        print(f"数据集 [{dataset_name}] 报告已保存到: {dataset_report_dir}/")


def main():
    """主函数，从 YAML 配置文件读取所有参数。

    用法：
        python generate_dataset_tables.py [config.yaml]

    配置文件格式（默认路径: tables_config.yaml）：
        results_dir: results
        output_dir: tables
        exp_suffix: 20240101_120000   # 可选，不填则自动选最新实验
        dataset: ToMQA                # 可选，单个数据集
        dataset:                      # 或列表，只处理其中的数据集
          - ToMQA
          - ToMi
        models:                       # 可选，只处理这些模型
          - Qwen3-8B                  # 字符串：目录名即显示名
          - name: some-long-name      # 字典：指定显示名
            display: ShortName
            exp_suffix: 20240101_120000  # 可选：该模型条目绑定的实验后缀
    """
    import sys
    import yaml

    config_path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "tables_config.yaml")
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    models_raw = cfg.get("models", None)
    models_filter = None
    model_display_names: Dict[str, str] = {}
    model_runs: Optional[List[ModelRun]] = None

    if models_raw is not None:
        models_filter = []
        model_runs = []
        for item in models_raw:
            if isinstance(item, str):
                models_filter.append(item)
                model_runs.append({"name": item, "display": item, "exp_suffix": None})
            elif isinstance(item, dict):
                dir_name = item["name"]
                display = item.get("display", dir_name)
                models_filter.append(dir_name)
                model_runs.append({
                    "name": dir_name,
                    "display": display,
                    "exp_suffix": item.get("exp_suffix"),
                })
                if display != dir_name:
                    model_display_names[dir_name] = display

    generate_dataset_tables(
        results_dir=cfg.get("results_dir", "results"),
        output_dir=cfg.get("output_dir", "tables"),
        exp_suffix=cfg.get("exp_suffix", None),
        dataset_filter=cfg.get("dataset", None),
        models_filter=models_filter,
        model_display_names=model_display_names or None,
        model_runs=model_runs,
    )


if __name__ == "__main__":
    main()
