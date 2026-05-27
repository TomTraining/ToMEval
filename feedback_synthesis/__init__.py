"""
Data Processing Module — 合成数据流水线

主要入口：
- stage1_load_predictions: 从已有预测文件加载 bad case
- stage2_diagnosis:        维度批量诊断
- synthesis:               从诊断报告合成新样本
"""

from .stage1_load_predictions import load_bad_cases_from_predictions
from .stage3_synthesis import synthesize_from_reports, run_stage3_synthesis

__all__ = [
    "load_bad_cases_from_predictions",
    "synthesize_from_reports",
    "run_stage3_synthesis",
]
