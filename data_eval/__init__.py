"""
data_eval — 合成数据质量评价模块

四个评估维度：
- format:           格式校验（复用 F007 validate_dataframe）
- difficulty:       强 LLM 打 1-5 难度分
- answerability:    LLM 检查前提自洽/答案唯一/逻辑无误
- representativeness: LLM 评估维度代表性

入口：run_eval.py --eval {format,difficulty,answerability,representativeness,all}
"""
