#!/usr/bin/env python3
"""data_eval 入口脚本。

用法示例：
  python run_eval.py --eval format --dataset BigToM
  python run_eval.py --eval difficulty --dataset BigToM --max-rows 10
  python run_eval.py --eval all --dataset FanToM --iter 1 --model "*"
"""

import argparse
import sys


EVAL_CHOICES = ["format", "difficulty", "answerability", "representativeness", "all"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="合成数据质量评估")
    p.add_argument("--eval", required=True, choices=EVAL_CHOICES,
                   help="评估类型：format / difficulty / answerability / representativeness / all")
    p.add_argument("--dataset", required=True,
                   help="数据集名称，如 BigToM")
    p.add_argument("--iter", type=int, default=1,
                   help="迭代轮次，默认 1")
    p.add_argument("--model", default="*",
                   help="模型名（支持 glob），默认 *（自动匹配）")
    p.add_argument("--max-rows", type=int, default=None,
                   help="最多处理行数（用于快速测试）")
    p.add_argument("--root", default="feedback_data/synth_clean",
                   help="synth_clean 根目录，默认 feedback_data/synth_clean")
    p.add_argument("--output-root", default="data_eval_output",
                   help="评估报告输出根目录，默认 data_eval_output")
    return p.parse_args()


def run_format(args: argparse.Namespace) -> int:
    from data_eval.eval_format import run_format_eval
    try:
        result = run_format_eval(
            dataset=args.dataset,
            iter_n=args.iter,
            model=args.model,
            root=args.root,
            max_rows=args.max_rows,
            output_root=args.output_root,
        )
    except FileNotFoundError as e:
        print(f"[format] 找不到数据文件: {e}", file=sys.stderr)
        return 2
    status = "PASS" if result.pass_ else "FAIL"
    print(f"[format] {result.dataset} — {status} (total_rows={result.total_rows}, failures={len(result.records)})")
    return 0 if result.pass_ else 1


def run_difficulty(args: argparse.Namespace) -> int:
    from data_eval.eval_difficulty import run_difficulty_eval
    try:
        result = run_difficulty_eval(
            dataset=args.dataset,
            iter_n=args.iter,
            model=args.model,
            root=args.root,
            max_rows=args.max_rows,
            output_root=args.output_root,
        )
    except FileNotFoundError as e:
        print(f"[difficulty] 找不到数据文件: {e}", file=sys.stderr)
        return 2
    meta = result.meta
    print(
        f"[difficulty] {result.dataset} — total={result.total_rows} "
        f"simple_mean_pass_rate={meta.get('simple_mean_pass_rate', '-')} "
        f"strong_mean={meta.get('strong_difficulty_mean', '-')} "
        f"strong_failed={meta.get('strong_failed_count', 0)}"
    )
    return 0


def run_answerability(args: argparse.Namespace) -> int:
    from data_eval.eval_answerability import run_answerability_eval
    try:
        result = run_answerability_eval(
            dataset=args.dataset,
            iter_n=args.iter,
            model=args.model,
            root=args.root,
            max_rows=args.max_rows,
            output_root=args.output_root,
        )
    except FileNotFoundError as e:
        print(f"[answerability] 找不到数据文件: {e}", file=sys.stderr)
        return 2
    meta = result.meta
    a = meta.get("stage_a_distribution", {})
    b = meta.get("stage_b_distribution", {})
    c = meta.get("stage_c_label_distribution", {})
    print(
        f"[answerability] {result.dataset} — total={result.total_rows} "
        f"score={meta.get('answerability_score', '-')} "
        f"A={a} B={b} C={c}"
    )
    return 0


def run_representativeness(args: argparse.Namespace) -> int:
    from data_eval.eval_representativeness import run_representativeness_eval
    try:
        result = run_representativeness_eval(
            dataset=args.dataset,
            iter_n=args.iter,
            model=args.model,
            root=args.root,
            max_rows=args.max_rows,
            output_root=args.output_root,
        )
    except FileNotFoundError as e:
        print(f"[representativeness] 找不到数据文件: {e}", file=sys.stderr)
        return 2
    meta = result.meta
    print(
        f"[representativeness] {result.dataset} — total={result.total_rows} "
        f"mean_score={meta.get('mean_representativeness_score', '-')} "
        f"parse_error={meta.get('parse_error_count', 0)} "
        f"dimensions={len(meta.get('dimension_breakdown', {}))}"
    )
    return 0


DISPATCH = {
    "format": run_format,
    "difficulty": run_difficulty,
    "answerability": run_answerability,
    "representativeness": run_representativeness,
}


def main() -> int:
    args = parse_args()
    evals = list(DISPATCH.keys()) if args.eval == "all" else [args.eval]
    exit_code = 0
    for ev in evals:
        code = DISPATCH[ev](args)
        if code != 0:
            exit_code = code
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
