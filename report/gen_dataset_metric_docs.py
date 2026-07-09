"""为每个数据集生成 docs/tasks/<DS>_table.md（EN 版）与 <DS>_table_zh.md（全中文版）。

内容 = 各级指标（一级 + 二/三/四级维度）的**定义**与**逐值释义**，以及汇总型维度的
精确计算口径。维度层级、定义、翻译全部来自本文件内的显式规格 `_DIMS`（单一事实源），
**不再依赖 results/.../metrics.json**，因此无实验产物也能离线重跑。

`_DIMS` 与各 tasks/<DS>/metrics.py 的维度声明保持一致；逐值取值来自各数据集 parquet 的
真实 meta 字段。改了 metrics.py 的维度，请同步改这里。

用法:
    python report/gen_dataset_metric_docs.py
产物写入 docs/tasks/，并刷新 README.md / README_zh.md 索引。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "tasks"

_KIND_ZH = {"split": "切分型", "summary": "汇总型"}


def D(
    code: str,
    name_zh: str,
    kind: str,
    desc: str,
    *,
    values: Optional[List[Tuple[str, str, str]]] = None,
    values_note: Optional[str] = None,
    value_list: Optional[List[str]] = None,
    compute: Optional[str] = None,
    unit: str = "ACC",
    children: Optional[List[dict]] = None,
) -> dict:
    """一个维度规格。values=[(code, 中文短名, 释义)]；高基数维度用 values_note+value_list。"""
    return {
        "code": code,
        "name_zh": name_zh,
        "kind": kind,
        "desc": desc,
        "values": values,
        "values_note": values_note,
        "value_list": value_list,
        "compute": compute,
        "unit": unit,
        "children": children or [],
    }


# --------------------------------------------------------------------------- #
# 一级指标（所有数据集固定五项）
# --------------------------------------------------------------------------- #
_PRIMARY: List[Tuple[str, str, str]] = [
    ("accuracy", "准确率", "一级指标。全部样本的总体准确率 = correct / total。open 题按各数据集判分模式（f1 / rubric）二值化后计入。"),
    ("correct", "答对数", "答对样本数。"),
    ("total", "样本总数", "参与评测的样本总数。"),
    ("extraction_failed", "抽取失败数", "答案抽取失败数（MCQ 严格模式下未输出 \\boxed{} 即记为抽取失败）。"),
    ("extraction_failed_rate", "抽取失败率", "抽取失败率 = extraction_failed / total，衡量模型对输出格式的遵循程度。"),
]

# 通用二级维度 type（所有数据集固定带）。
_TYPE = D(
    "type", "题型", "split",
    "题型维度（所有数据集固定带）。按 prompt_type 切分，每个 split 是该题型的准确率。",
    values=[
        ("mcq_single", "单选", "仅一个正确项 + 干扰项的选择题"),
        ("mcq_multi", "多选", "有 ≥2 个正确项的选择题"),
        ("mcq_grouped", "捆绑判分", "一条 prompt 内含多个子问题，全部答对才计为对（如 EmoBench EU 的情绪+原因）"),
        ("open", "开放题", "无干扰项、自由作答，按 f1 / rubric 判分后二值化"),
    ],
)


# --------------------------------------------------------------------------- #
# 各数据集维度规格（有序；type 由渲染器自动置于最前）。
# --------------------------------------------------------------------------- #
_DIMS: Dict[str, List[dict]] = {
    "Belief_R": [
        D("step", "推理步骤", "split", "按信念推理步骤切分（meta.step 归一）。", values=[
            ("belief_update", "信念修正", "需根据新信息更新既有信念（meta.step=time_t1）"),
            ("belief_matching", "信念匹配", "直接匹配已陈述的信念、无需更新（meta.step=time_t）"),
        ]),
        D("modus", "推理式", "split", "按逻辑推理式切分（meta.modus）。", values=[
            ("ponens", "肯定前件", "由「若 A 则 B」与 A，推出 B"),
            ("tollens", "否定后件", "由「若 A 则 B」与 ¬B，推出 ¬A"),
        ]),
        D("types_of_relation", "关系类型", "split", "按条件规则的关系类型切分（meta.types_of_relation）。", values=[
            ("If-Event-Then-Event", "事件→事件", "前件后件都是事件"),
            ("If-Event-Then-MentalState", "事件→心理状态", "后件是心理状态（信念/情绪等）"),
        ]),
        D("belief_reasoning", "信念推理", "summary",
          "Belief-R 官方头条指标，把信念修正/匹配两子集汇总。",
          compute="BREU = (BU-Acc + BM-Acc) / 2，是两子集准确率的**宏平均**，不是全样本 ACC；n 为两子集样本合计。",
          values=[
              ("BREU", "信念推理综合", "BU-Acc 与 BM-Acc 的宏平均（官方 BREU）"),
              ("BU-Acc", "信念修正准确率", "belief_update 子集准确率"),
              ("BM-Acc", "信念匹配准确率", "belief_matching 子集准确率"),
          ]),
    ],
    "BigToM": [
        D("condition", "生成条件", "split", "按因果模板生成条件切分（meta.condition_type）。", values=[
            ("forward_belief", "正向-信念", "由感知情境推断角色信念"),
            ("forward_action", "正向-行为", "由角色信念预测其行为"),
            ("backward_belief", "反向-信念", "由观察到的行为反推角色信念"),
            ("percept_to_belief", "感知→信念", "由是否感知到关键事件推断信念"),
        ]),
        D("belief_type", "信念类型", "split", "按信念类型切分（meta.dimension 归一化：小写、-/空格转 _）。", values=[
            ("true_belief", "真信念", "角色信念与现实一致"),
            ("false_belief", "假信念", "角色信念与现实不符（核心 ToM 考点）"),
            ("true_control", "真-对照", "去除 ToM 线索的对照项（真）"),
            ("false_control", "假-对照", "去除 ToM 线索的对照项（假）"),
        ]),
        D("tb_and_fb", "真假信念配对", "summary",
          "配对联合指标：同一故事的真信念题与假信念题都答对才算该 pair 通过。",
          compute="按故事分组（去掉 _true_belief/_false_belief 后缀得 pair 键），仅统计同时含 TB、FB 两问的 pair；两问全对的 pair 数 / 总 pair 数。无法由两个边际 ACC 反推。",
          values=[("overall", "总体", "全部配对故事上的联合通过率")]),
    ],
    "EmoBench": [
        D("subset", "子集", "split", "按官方子集切分（meta.subset）。", values=[
            ("emotional_understanding", "情绪理解(EU)", "判断当事人的情绪及其成因"),
            ("emotional_application", "情绪应用(EA)", "在情境中选择合适的行动/回应"),
        ]),
        D("language", "语种", "split", "按语种切分（meta.language）。", values=[
            ("en", "英文", "英文题"), ("zh", "中文", "中文题"),
        ]),
        D("question_subtype", "问题子类型", "split", "按问题子类型切分（meta.question_subtype）。", values=[
            ("emotion", "情绪", "判断当事人的情绪"),
            ("cause", "原因", "判断情绪的成因"),
            ("Action", "行动", "EA：应采取的行动"),
            ("Response", "回应", "EA：应作出的回应"),
        ]),
        D("coarse_category", "情绪粗类", "split",
          "情绪理解(EU)的粗类（meta.coarse_category），其下嵌套 finegrained_category（EA 子集无此标签）。",
          values=[
              ("complex_emotions", "复杂情绪", "混合/转变类复杂情绪"),
              ("emotional_cues", "情绪线索", "从视觉/言语线索推断情绪"),
              ("personal_beliefs_and_experiences", "个人信念与经历", "受个人信念、文化、经历影响的情绪"),
              ("perspective_taking", "观点采择", "站在他人视角推断情绪"),
              ("unknown", "未分类", "情绪应用(EA)子集样本无粗类标签，归为 unknown"),
          ],
          children=[
              D("finegrained_category", "情绪细类", "split",
                "情绪理解细类（meta.finegrained_category），挂在各 coarse_category split 下。", values=[
                    ("mixture_of_emotions", "混合情绪", "同时存在多种情绪"),
                    ("emotion_transition", "情绪转变", "情绪随情节变化"),
                    ("unexpected_outcome", "意外结局", "结局出人意料引发的情绪"),
                    ("visual_cues", "视觉线索", "由表情/动作等视觉线索判断"),
                    ("vocal_cues", "言语线索", "由语气/措辞等线索判断"),
                    ("cultural_value", "文化价值", "受文化价值观影响的情绪"),
                    ("sentimental_value", "情感价值", "受物品/回忆的情感价值影响"),
                    ("persona", "人物设定", "依据人物设定推断情绪"),
                    ("false_belief", "错误信念", "基于错误信念的情绪"),
                    ("faux_pas", "失礼", "失礼情境下的情绪"),
                    ("strange_story", "奇异故事", "Happé 奇异故事式情境"),
                    ("unknown", "未分类", "情绪应用(EA)子集样本无细类标签，归为 unknown"),
                ]),
          ]),
        D("dimension", "能力标签", "split",
          "细粒度能力标签（meta.dimension，可多值），把粗/细类与 EA 子任务平铺成一张总表。",
          values_note="取值为多段组合标签，形如 ['emotional_application','Personal-Others','Action']（EA）或 ['emotional_understanding', 粗类, 细类]（EU）。"),
        D("eu_subquestion", "EU 子问题", "summary",
          "EU 子问题诊断：从 mcq_grouped 记录的 sub_results 拆出情绪/原因两类子问题。",
          compute="遍历所有 mcq_grouped 记录的 sub_results，按 subtype(emotion/cause) 分别累计 correct/total，得两条子问题准确率。",
          values=[
              ("emotion", "情绪判断", "EU 情绪子问题准确率"),
              ("cause", "原因判断", "EU 原因子问题准确率"),
          ]),
    ],
    "ExploreToM": [
        D("dimension", "考察维度", "split", "按考察维度切分（meta.dimension）。", values=[
            ("belief", "信念", "一般信念追踪题"),
            ("false_belief", "错误信念", "错误信念题（核心 ToM）"),
        ]),
        D("answer_type", "答案类型", "split", "按答案形式切分（meta.answer_type）。", values=[
            ("binary_knows", "二元-知道", "判断「某角色是否知道」的是非题"),
            ("binary_yesno", "二元-是非", "一般是非题"),
            ("location", "位置", "回答物体所在位置（开放作答）"),
        ]),
        D("nth_order", "信念阶数", "split", "按信念阶数切分（meta.nth_order）。", values=[
            ("1", "一阶", "对他人信念的推理"),
            ("2", "二阶", "对「他人关于他人信念」的推理"),
            ("-1", "非信念", "非信念阶（事实/记忆类）"),
        ]),
        D("story_type", "故事类型", "split",
          "按故事生成模板切分（meta.story_type）。",
          values_note="由 ToMi/FANToM 等模板及其变体组合而成（tomi*/fantom-public/fantom-private/all* 等），后缀 +asymmetric 表示信息不对称加强。共 18 种。"),
    ],
    "FanToM": [
        D("question_type", "题型", "split", "按题型切分（meta.question_type）。", values=[
            ("beliefQAs", "信念题", "角色对事实的信念（标准化为多选形式）"),
            ("answerabilityQAs_binary", "可答性-二元", "该问题在给定信息下谁能回答（是非）"),
            ("answerabilityQA_list", "可答性-列举", "列举能回答该问题的角色"),
            ("infoAccessibilityQAs_binary", "信息可达-二元", "谁获知了该信息（是非）"),
            ("infoAccessibilityQA_list", "信息可达-列举", "列举获知该信息的角色"),
            ("factQA", "事实控制", "事实核对控制项（非 ToM 题）"),
        ]),
        D("order", "ToM 阶数", "split", "按 ToM 阶数切分（meta.order）。", values=[
            ("0", "事实/前置", "事实或一阶前置"),
            ("1", "一阶信念", "一阶信念推理"),
            ("2", "二阶信念", "二阶信念推理"),
        ]),
        D("set_all", "set 级 ALL", "summary",
          "FANToM 官方头条指标：同一 info-set 内指定 ToM 题型全部答对才算该 set 通过。",
          compute="按 snippet(info-set) 用 group_all_correct 分组，要求指定题型全部出现且全对；通过 set 数 / 总 set 数。对应官方 All(MC belief)。",
          values=[
              ("overall", "总体", "全部 ToM 题型都答对才通过"),
              ("answerability", "可答性子集", "仅要求 answerability 两题型全对"),
              ("infoaccess", "信息可达子集", "仅要求 infoAccessibility 两题型全对"),
          ]),
    ],
    "FictionalQA": [
        D("style", "文体", "split", "按虚构文体切分（meta.style）。", values=[
            ("news", "新闻", "新闻稿体"),
            ("corporate", "企业公文", "企业/公文体"),
            ("encyclopedia", "百科", "百科词条体"),
            ("blog", "博客", "博客文体"),
            ("social", "社媒", "社交媒体文体"),
        ]),
        D("grading", "有上下文 vs 盲评", "summary",
          "informed-vs-blind 对照，衡量「给了虚构上下文」相对「盲评」的增益。",
          compute="informed = 全样本 ACC；blind = 对每题的 meta.blind_grade_avg 求平均（无上下文盲评均分）；两者之差即官方关注的 gap。",
          values=[
              ("informed", "有上下文", "模型在给定虚构上下文下的准确率"),
              ("blind", "盲评", "无上下文时的盲评均分"),
          ]),
        D("macro_split_acc", "宏平均准确率", "summary",
          "三种分组口径下的宏平均，消除大组主导。",
          compute="每种口径先按组算组内 ACC，再对各组**等权平均**（n=组数），而非全样本 ACC。",
          values=[
              ("event", "按事件", "先按 event 分组算 ACC 再跨组平均"),
              ("document", "按文档", "先按 document 分组算 ACC 再跨组平均"),
              ("style", "按文体", "先按 style 分组算 ACC 再跨组平均"),
          ]),
    ],
    "HellaSwag": [
        D("split_type", "划分", "split", "按官方划分切分（meta.split_type）。", values=[
            ("indomain", "域内", "in-domain 划分"),
            ("zeroshot", "零样本", "zero-shot 划分（活动类别未在训练出现）"),
        ]),
    ],
    "HiToM": [
        D("order", "心智阶数", "split",
          "按心智推理阶数切分（meta.order，0–4）。阶数越高、嵌套信念越深，是 Hi-ToM 的核心难度轴。", values=[
            ("0", "零阶", "事实/现实层面"),
            ("1", "一阶", "对他人信念的推理"),
            ("2", "二阶", "对「他人关于他人信念」的推理"),
            ("3", "三阶", "三层嵌套信念"),
            ("4", "四阶", "四层嵌套信念"),
        ]),
    ],
    "PUB": [
        D("option_count", "选项个数", "split",
          "按候选选项个数切分（len(options)=meta.n_options），观察候选数对正确率的影响。",
          values_note="取值为 2 / 3 / 4 / 5。"),
    ],
    "SimpleToM": [
        D("dimension", "考察维度", "split", "按考察维度切分（meta.dimension）。", values=[
            ("information_access", "信息可达", "角色能否获知关键信息"),
            ("behavior_prediction", "行为预测", "预测角色在（可能错误的）信念下的行为"),
            ("social_judgment", "社会判断", "对角色行为的社会评价"),
        ]),
        D("qa_type", "题型", "split", "按问答类型切分（meta.qa_type）。", values=[
            ("mental_state", "心理状态", "关于角色心理状态的题"),
            ("behavior", "行为", "关于角色行为的题"),
            ("judgment", "判断", "社会判断题"),
        ]),
        D("scenario_name", "情景", "split", "按日常情景切分（meta.scenario_name），共 10 种。", values=[
            ("behind_the_scene_service_industry", "服务业幕后", "服务行业不为顾客所见的幕后环节"),
            ("food_item_in_grocery_store", "超市食品", "超市货架上的食品状况"),
            ("hidden_body_part_feature", "隐藏身体特征", "被遮挡而不可见的身体特征"),
            ("inside_containers_for_personal_belongings", "私人物品容器内", "私人物品容器内部的东西"),
            ("inside_reuse_labeled_containers", "重用标签容器内", "重复使用、标签与内容物不符的容器"),
            ("locked_devices_accounts", "上锁设备/账户", "上锁的设备或账户内部"),
            ("provider_info_healthcare", "医疗方信息", "医疗服务提供方的隐含信息"),
            ("seller_info_in_second_hand_market", "二手卖家信息", "二手市场卖家的隐含信息"),
            ("true_property_pretentious_labels", "名不副实标签", "标签夸大时的真实属性"),
            ("unobserved_unethical_actions", "无人目击不当行为", "无人目击时的不当行为"),
        ]),
    ],
    "SocialBench": [
        D("category", "类别", "split",
          "按官方类别码切分（meta.category），格式 <层级>-<能力>-<子任务>。", values=[
            ("Individual-SA-RoleKnowledge", "个体-自我认知-角色知识", "角色应知的背景知识"),
            ("Individual-SA-RoleStyle", "个体-自我认知-角色风格", "角色说话风格一致性"),
            ("Individual-EP-DialogueEmotionDetect", "个体-情绪感知-对话情绪识别", "识别对话中的情绪"),
            ("Individual-EP-HumorSarcasmDetect", "个体-情绪感知-幽默讽刺识别", "识别幽默/讽刺"),
            ("Individual-EP-SituationUnderstanding", "个体-情绪感知-情境理解", "理解情境含义"),
            ("Individual-MEM-Long", "个体-记忆-长程", "长程对话记忆"),
            ("Individual-MEM-Short", "个体-记忆-短程", "短程对话记忆"),
            ("Group-SAP-Positive", "群体-社会偏好-正向", "群体互动中的正向社会偏好"),
            ("Group-SAP-Neutral", "群体-社会偏好-中性", "中性社会偏好"),
            ("Group-SAP-Negative", "群体-社会偏好-负向", "负向社会偏好"),
        ]),
        D("dimension", "能力维度", "split", "按能力维度切分（meta.dimension，可多值）。", values=[
            ("conversation_memory", "对话记忆", "记住并利用对话历史"),
            ("self_awareness", "自我认知", "对自身角色设定的认知"),
            ("social_preference", "社会偏好", "群体互动中的社会偏好"),
            ("emotional_perception", "情绪感知", "感知与识别情绪"),
        ]),
        D("lang", "语种", "split", "按语种切分（meta.lang）。", values=[
            ("en", "英文", "英文题"), ("zh", "中文", "中文题"),
        ]),
        D("num_choices", "候选数", "split",
          "按候选个数切分（len(options)）。",
          values_note="0 表示开放题（走 f1 判分），其余为该题选项个数。"),
    ],
    "SocialIQA": [
        D("dimension", "推理维度", "split",
          "按 ATOMIC 推理维度切分（meta.dimension）。x* 关于事件主角 PersonX，o* 关于其他人。", values=[
            ("xIntent", "X的意图", "事件前 X 做该行为的意图/动机"),
            ("xNeed", "X的需求", "事件前 X 需要满足的前置条件"),
            ("xAttr", "X的属性", "事件反映出的 X 的性格/属性"),
            ("xEffect", "对X的影响", "事件后发生在 X 身上的影响"),
            ("xReact", "X的反应", "事件后 X 的情绪反应"),
            ("xWant", "X的意愿", "事件后 X 想做的事"),
            ("oEffect", "对他人的影响", "事件后其他人受到的影响"),
            ("oReact", "他人的反应", "事件后其他人的情绪反应"),
            ("oWant", "他人的意愿", "事件后其他人想做的事"),
        ]),
    ],
    "SoMBench": [
        D("dim1", "能力大类", "split",
          "维度体系顶层能力大类（meta.dim1），其下嵌套 dim2 → dim3。",
          values_note="取值 1 / 2 / 3，为三个顶层能力大类（编号制）。",
          children=[
              D("dim2", "二级标签", "split",
                "维度体系二级标签（meta.dim2），挂在各 dim1 split 下。",
                values_note="形如 1.1 / 2.4 / 3.2，共 17 个二级标签。",
                children=[
                    D("dim3", "三级标签", "split",
                      "维度体系三级最细标签（meta.dim），挂在各 dim2 split 下，是 SocMind 最细粒度考察点。",
                      values_note="形如 1.1.2 / 2.4.7，共 71 个三级标签。"),
                ]),
          ]),
        D("qtype", "题型编号", "split", "按题型编号切分（meta.qtype）。", values=[
            ("Q1", "客观题1", "客观题（单/多选）"),
            ("Q2", "客观题2", "客观题（单/多选）"),
            ("Q3", "客观题3", "客观题（单/多选）"),
            ("Q4", "开放分析题", "开放分析题，走 rubric LLM 判分（0–10 分过阈值）"),
        ]),
        D("perspective", "叙事视角", "split", "按叙事视角切分（meta.perspective）。", values=[
            ("first_person", "第一人称", "以第一人称叙述"),
            ("third_person", "第三人称", "以第三人称叙述"),
        ]),
        D("variant", "难度变体", "split", "按难度变体切分（meta.variant）。", values=[
            ("base", "基础全量", "全量基础题"),
            ("hardest", "加难版", "dim-3 加难版"),
            ("varA", "改写变体A", "其他改写变体 A"),
            ("varB", "改写变体B", "其他改写变体 B"),
        ]),
        D("length", "文本长短", "split", "按情景文本长短切分（meta.length_mode）。", values=[
            ("long", "长文", "长情景文本"),
            ("short", "短文", "短情景文本"),
        ]),
        D("q4_score", "Q4 评分", "summary",
          "Q4 rubric 平均分（0–10 分，非 0–1 准确率）。", unit="0-10 均分",
          compute="仅取 open 且有 judge_score 的样本，按 0–10 rubric 求均分。overall=全部 Q4 均分，其余 split 按 meta.dim（三级维度）分组。",
          values=[("overall", "总体", "全部 Q4 的 rubric 平均分")]),
    ],
    "TactfulToM": [
        D("category", "问题大类", "split", "按问题大类切分（meta.category）。", values=[
            ("belief", "信念", "对角色信念的判断"),
            ("answerability", "可答性", "问题在给定信息下是否可答"),
            ("info_accessibility", "信息可达性", "角色是否能获知某信息"),
            ("lieability", "善意谎言点", "识别谁会/该说善意谎言"),
            ("liedetectability", "谎言可识别性", "谎言是否可被识破"),
            ("justification", "辩护", "对善意谎言动机的解释判断（与 comprehension 配对做联合）"),
            ("comprehension", "对话理解", "对话事实理解题（与 justification 配对）"),
            ("fact", "事实控制", "事实核对控制项"),
        ]),
        D("question_type", "细粒度题型", "split",
          "按细粒度题型完整路径切分（meta.question_type）。",
          values_note="路径格式 <层级>:<能力>:<形式/可达性>:<truth 真话 / real_reason 真因 / reason 理由>，如 tom:belief:accessible:truth。共 23 种。"),
        D("lie_type", "白谎类型", "split", "按白谎类型切分（meta.lie_type）。", values=[
            ("altruistic_white_lies", "利他型善意谎言", "纯为他人利益的善意谎言"),
            ("pareto_white_lies", "帕累托型善意谎言", "利他且不损己的双赢善意谎言"),
        ]),
        D("tom_type", "ToM 阶数×角色", "split",
          "按 ToM 阶数与角色组合切分（meta.tom_type）。",
          values_note="形如 first-order:A / second-order:AB，字母为角色代号；空值为非 ToM 题。共 13 种。"),
        D("joint_comp_just", "理解∧辩护联合", "summary",
          "Comp∧Just 联合指标：同一对话的对话理解与辩护两题都答对，才算真正理解了善意谎言（Happé 双题判定）。",
          compute="按 set_id 用 group_all_correct 分组，要求同时含 comprehension、justification；两者皆对的对话数 / 总对话数。无法由两个边际 ACC 反推。",
          values=[("overall", "总体", "全部对话上的联合通过率")]),
    ],
    "ToMBench": [
        D("task", "任务", "split",
          "按任务切分（meta.filename 去 .jsonl 后缀），对应官方 Task-oriented 结果表。",
          values_note="共 20 个任务，如 False-Belief-Task / Faux-pas-Recognition-Test / Strange-Story-Task / Hinting-Task-Test 等。"),
        D("ability", "能力", "split",
          "按能力切分（meta.ability），对应官方 Ability-oriented。",
          values_note="共 33 项，格式 <大类>: <细项>，大类含 Belief / Desire / Emotion / Intention / Knowledge / Non-Literal Communication。"),
        D("lang", "语种", "split", "按语种切分（meta.lang）。", values=[
            ("en", "英文", "英文题"), ("zh", "中文", "中文题"),
        ]),
    ],
    "ToMChallenges": [
        D("question_type", "题型", "split", "按题型切分（meta.question_type）。", values=[
            ("1stA", "一阶-A", "一阶信念题（问法 A）"),
            ("1stB", "一阶-B", "一阶信念题（问法 B）"),
            ("2ndA", "二阶-A", "二阶信念题（问法 A）"),
            ("2ndB", "二阶-B", "二阶信念题（问法 B）"),
            ("assumption", "前提假设", "对前提假设的判断"),
            ("memory", "记忆", "记忆核对题"),
            ("reality", "现实", "现实核对题"),
        ]),
        D("task_format", "作答形式", "split", "按作答形式切分（meta.task_format）。", values=[
            ("mc", "选择题", "多选一形式"),
            ("qa", "问答", "开放问答形式"),
        ]),
        D("test_type", "测试范式", "split", "按经典 ToM 测试范式切分（meta.test_type）。", values=[
            ("sally-anne", "Sally-Anne", "意外转移范式"),
            ("smarties", "Smarties", "意外内容范式"),
        ]),
    ],
    "ToMQA": [
        D("dimension", "考察维度", "split", "按考察维度切分（meta.dimension）。", values=[
            ("belief", "信念", "信念推理题"),
            ("memory", "记忆", "记忆核对题"),
            ("reality", "现实", "现实核对题"),
            ("search", "搜索", "搜索/位置题"),
        ]),
        D("task", "bAbI 任务型", "split", "按 bAbI 风格任务型切分（meta.task）。", values=[
            ("fb", "一阶错误信念", "first-order false belief"),
            ("tb", "真信念", "true belief"),
            ("sofb", "二阶错误信念", "second-order false belief"),
        ]),
    ],
    "ToMato": [
        D("mental_state", "心智状态", "split", "按心智状态大类切分（meta.mental_state）。", values=[
            ("belief", "信念", "关于信念的推理"),
            ("desire", "欲望", "关于欲望的推理"),
            ("emotion", "情绪", "关于情绪的推理"),
            ("intention", "意图", "关于意图的推理"),
            ("knowledge", "知识", "关于知识的推理"),
        ]),
        D("order", "推理阶数", "split", "按心智推理阶数切分（meta.order）。", values=[
            ("1", "一阶", "对他人心智状态的推理"),
            ("2", "二阶", "对「他人关于他人心智状态」的推理"),
        ]),
        D("false_belief", "是否错误信念", "split", "按是否涉及错误信念切分（meta.false_belief）。", values=[
            ("True", "是", "涉及错误信念"),
            ("False", "否", "不涉及错误信念"),
        ]),
    ],
    "ToMi": [
        D("story_type", "故事型", "split", "按故事类型切分（meta.story_type）。", values=[
            ("true_belief", "真信念", "真信念故事"),
            ("false_belief", "错误信念", "一阶错误信念故事"),
            ("second_order_false_belief", "二阶错误信念", "二阶错误信念故事"),
            ("unknown", "未标注", "meta.story_type 为空的样本"),
        ]),
        D("question_type", "题型", "split", "按题型切分（meta.question_type）。", values=[
            ("first_order_0_tom", "一阶-需ToM(0)", "一阶、需要 ToM（变体 0）"),
            ("first_order_1_tom", "一阶-需ToM(1)", "一阶、需要 ToM（变体 1）"),
            ("first_order_0_no_tom", "一阶-无需ToM(0)", "一阶、无需 ToM（变体 0）"),
            ("first_order_1_no_tom", "一阶-无需ToM(1)", "一阶、无需 ToM（变体 1）"),
            ("second_order_0_tom", "二阶-需ToM(0)", "二阶、需要 ToM（变体 0）"),
            ("second_order_1_tom", "二阶-需ToM(1)", "二阶、需要 ToM（变体 1）"),
            ("second_order_0_no_tom", "二阶-无需ToM(0)", "二阶、无需 ToM（变体 0）"),
            ("second_order_1_no_tom", "二阶-无需ToM(1)", "二阶、无需 ToM（变体 1）"),
            ("memory", "记忆", "记忆核对题"),
            ("reality", "现实", "现实核对题"),
            ("unknown", "未标注", "meta.question_type 为空的样本"),
        ]),
    ],
}

# 数据集级说明/告警（可选），渲染进页头。
_DATASET_NOTES: Dict[str, str] = {
    "PUB": "⚠️ PUB 标准化后 meta 很稀薄：原始的 source / difficulty / ethics_category / task_type 及 14 个语用子任务信息在转换时已丢失，`dimension` 恒为单值 `pragmatics`。故本页只保留有区分度的 `option_count`（候选个数）与固定的 `type` 维度。",
    "ToMato": "说明：标准化数据里 `meta.dimension` 实为单槽（仅心智状态一项），旧版按 slot1→slot2→slot3 展开的三/四级维度全是占位空桶，已删除，改用 `mental_state` / `order` / `false_belief` 真实字段。",
    "ExploreToM": "说明：旧版声明的 `difficulty` / `task_type` / `order` 在标准化 meta 中并不存在（恒为 unknown），已替换为真实字段 `answer_type` / `nth_order` / `story_type`。",
    "ToMQA": "说明：旧版声明的 `difficulty` / `task_type` / `order` 在标准化 meta 中并不存在（恒为 unknown），已替换为真实字段 `task`。",
}


# --------------------------------------------------------------------------- #
# 渲染
# --------------------------------------------------------------------------- #
def _flatten_rows(dims: List[dict]) -> List[Tuple[str, str, str]]:
    """把维度树展平成 (二级, 三级, 四级) 行；无对应层级用 — 占位。"""
    rows: List[Tuple[str, str, str]] = []
    for d2 in dims:
        if not d2["children"]:
            rows.append((d2["code"], "—", "—"))
            continue
        for d3 in d2["children"]:
            if not d3["children"]:
                rows.append((d2["code"], d3["code"], "—"))
                continue
            for d4 in d3["children"]:
                rows.append((d2["code"], d3["code"], d4["code"]))
    return rows


def _render_hierarchy(dims: List[dict], lines: List[str]) -> None:
    lines.append("## 指标层级总表")
    lines.append("")
    lines.append("> 横轴为指标层级；同一行表示嵌套归属（如 `dim1 → dim2 → dim3` 表示 dim2 是 dim1 的子维度、dim3 又是 dim2 的子维度）。`—` 表示该维度没有更深层级。")
    lines.append("")
    lines.append("| 一级指标 | 二级指标 | 三级指标 | 四级指标 |")
    lines.append("|---|---|---|---|")
    first = True
    for l2, l3, l4 in _flatten_rows(dims):
        c1 = "`accuracy`" if first else ""
        first = False
        c3 = f"`{l3}`" if l3 != "—" else "—"
        c4 = f"`{l4}`" if l4 != "—" else "—"
        lines.append(f"| {c1} | `{l2}` | {c3} | {c4} |")
    lines.append("")


_LEVEL_NAME = {2: "二级", 3: "三级", 4: "四级"}


def _render_dim(dim: dict, lines: List[str], level: int, parent_chain: str, zh: bool) -> None:
    code = dim["code"]
    kind_zh = _KIND_ZH.get(dim["kind"], dim["kind"])
    level_name = _LEVEL_NAME.get(level, f"{level}级")
    chain = f"{parent_chain}{code}"
    if zh:
        title_id = f"{dim['name_zh']}（{chain}）"
    else:
        title_id = f"`{chain}`"
    lines.append(f"### {level_name}指标 · {title_id}（{kind_zh}，单位 {dim['unit']}）")
    lines.append("")
    lines.append(dim["desc"])
    lines.append("")
    if dim["compute"]:
        lines.append(f"**计算方式**：{dim['compute']}")
        lines.append("")
    if dim["values"]:
        for vcode, vname, vgloss in dim["values"]:
            if zh:
                lines.append(f"- {vname}（`{vcode}`）：{vgloss}")
            else:
                lines.append(f"- `{vcode}`：{vgloss}")
        lines.append("")
    elif dim["values_note"]:
        lines.append(dim["values_note"])
        lines.append("")
    for child in dim["children"]:
        _render_dim(child, lines, level + 1, f"{chain} → ", zh)


def render_dataset(ds: str, zh: bool) -> str:
    dims = [_TYPE] + _DIMS.get(ds, [])
    lines: List[str] = []
    lines.append(f"# {ds} — 指标定义{'（全中文版）' if zh else ''}")
    lines.append("")
    lines.append("- 指标层级与 `metrics.json` 的 `avg_metrics.dimensions` 树一一对应：")
    lines.append("  **一级**=总体 accuracy；**二级**=各维度；**三级 / 四级**=维度内嵌套子维度。")
    lines.append("- `切分型`：把数据集切成多个 split，各 split 一条准确率；`汇总型`：任务特有口径（如配对联合、宏平均、set 级 ALL），单值无法从边际准确率反推。")
    lines.append("- 本页只列指标定义与逐值释义，不含具体数值。")
    if zh:
        lines.append("- 本页为全中文版：指标名 / 字段值一律中文（英文标识符括注在后）。英文技术版见 " + f"[{ds}_table.md]({ds}_table.md)。")
    else:
        lines.append("- 全中文版见 " + f"[{ds}_table_zh.md]({ds}_table_zh.md)。")
    note = _DATASET_NOTES.get(ds)
    if note:
        lines.append("")
        lines.append(f"> {note}")
    lines.append("")

    # 一级指标
    lines.append("## 一级指标")
    lines.append("")
    if zh:
        lines.append("| 指标 | 定义 |")
        lines.append("|---|---|")
        for code, name_zh, definition in _PRIMARY:
            lines.append(f"| {name_zh}（{code}） | {definition} |")
    else:
        lines.append("| 指标 | 定义 |")
        lines.append("|---|---|")
        for code, _name_zh, definition in _PRIMARY:
            lines.append(f"| {code} | {definition} |")
    lines.append("")

    _render_hierarchy(dims, lines)

    lines.append("## 各维度定义")
    lines.append("")
    for dim in dims:
        _render_dim(dim, lines, 2, "", zh)

    if ds == "SoMBench":
        _render_sombench_extra(lines, zh)

    return "\n".join(lines).rstrip() + "\n"


def _render_sombench_extra(lines: List[str], zh: bool) -> None:
    lines.append("## 人工审核与 qualified 镜像")
    lines.append("")
    lines.append("| 指标 | 定义 |")
    lines.append("|---|---|")
    if zh:
        lines.append("| 审核合格数（review_pass_count） | 人工审核合格（meta.review_pass=True）样本数。 |")
        lines.append("| 审核不合格数（review_fail_count） | 审核不合格样本数。 |")
        lines.append("| 合格率（review_pass_rate） | = review_pass_count / total。 |")
    else:
        lines.append("| review_pass_count | 人工审核合格（meta.review_pass=True）样本数。 |")
        lines.append("| review_fail_count | 审核不合格样本数。 |")
        lines.append("| review_pass_rate | 合格率 = review_pass_count / total。 |")
    lines.append("")
    lines.append("> `qualified` 是一份镜像：仅在审核合格样本上**重算同一套**一级/二级/三级/四级指标，"
                 "结构与上文完全一致（accuracy + dimensions 树）。v5.3 默认全部合格时，qualified 与全量一致。")
    lines.append("")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = sorted(_DIMS.keys())

    for ds in datasets:
        (OUT_DIR / f"{ds}_table.md").write_text(render_dataset(ds, zh=False), encoding="utf-8")
        (OUT_DIR / f"{ds}_table_zh.md").write_text(render_dataset(ds, zh=True), encoding="utf-8")

    # 英文技术版索引
    idx = ["# 数据集指标定义表（索引）", "",
           "每个数据集一页，含一级 / 二级 / 三级 / 四级指标定义与逐值释义。每页均有全中文副本 `*_table_zh.md`。", ""]
    for ds in datasets:
        idx.append(f"- [{ds}]({ds}_table.md) · [中文]({ds}_table_zh.md)")
    (OUT_DIR / "README.md").write_text("\n".join(idx) + "\n", encoding="utf-8")

    # 全中文索引
    idx_zh = ["# 数据集指标定义表 · 全中文索引", "",
              "每个数据集一页，指标名 / 字段值一律中文（英文标识符括注在后）。", ""]
    for ds in datasets:
        idx_zh.append(f"- [{ds}]({ds}_table_zh.md)")
    (OUT_DIR / "README_zh.md").write_text("\n".join(idx_zh) + "\n", encoding="utf-8")

    print(f"已生成 {len(datasets)} 个数据集 × 2（EN+zh）= {len(datasets) * 2} 个表格 → {OUT_DIR}")


if __name__ == "__main__":
    main()
