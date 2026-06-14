# 通用评测可视化模块 (`src/visualization`)

数据集无关的绘图模块：只消费 `results/.../metrics.json`，任何经标准流水线评测过的
数据集都能直接出图，无需为每个数据集写绘图代码。

## 用法

```bash
# 单模型：出该实验的全套分组图
python -m src.visualization --results results/SocialToM/<model>/<exp> --out figures/SocialToM

# 多模型对比：传多个实验目录，额外生成 comparison/ 对比图
python -m src.visualization \
    --results results/TactfulToM/modelA/<exp> results/TactfulToM/modelB/<exp> \
    --out figures/ttom_cmp
```

`--results` 可传实验目录或直接传 `metrics.json` 路径；`--out` 为图片输出目录。

## 自动产出（按 metrics.json 内容自适应）

- **分组准确率柱状图**：每个 `by_<x>` 分组一张（带 `n=样本数` 标注）。
  这些分组由各任务 `tasks/<DS>/metrics.py` 里的 `generic_group_metrics` 定义，
  例如 SocialToM 的 `by_qtype/by_variant/by_perspective/by_dim1/2/3`、
  TactfulToM 的 `by_category/by_question_type/by_lie_type/by_tom_type`。
- **热力图**：分组键形如 `"row|col"`（组合键）时自动透视成二维热力图。
  例：SocialToM 在 metrics 里输出 `by_dim3_qtype`（键 `"1.1.1|Q1"`）→ 维度×题型热力图。
- **分组平均分图**：键里含 `_by_`（分组）但不以 `by_` 开头的标量字典，视为
  "分组均分"（非 0-1 准确率），y 轴按数据自适应。例：SocialToM Q4 rubric 的
  `q4_mean_score_by_dim`（0-10 分）→ 各维度平均分柱状图，标题带 overall 总均分。
- **judge 一致性图**：`per_sample_results` 中含 `judge1_score/judge2_score` 时，
  出散点 + Bland-Altman（双 judge 评分一致性）；否则自动跳过。
- **多模型对比图**：传入多个实验时，出总体准确率对比 + 各共享分组的并排对比；
  其中类别数 ≥3 的分组额外出**雷达图**（如 SocialToM 二级维度 `by_dim2` 的多模型对比）。

## 让自己的数据集也能出图

不需要改可视化模块。只要 `tasks/<DS>/metrics.py` 的 `compute_metrics` 用
`src.evaluation.task_metrics.generic_group_metrics` 声明若干 `by_<x>` 分组即可；
想要热力图就额外加一个组合键分组（值形如 `f"{a}|{b}"`）。

依赖：`matplotlib`、`seaborn`（见 `requirements.txt`）。
