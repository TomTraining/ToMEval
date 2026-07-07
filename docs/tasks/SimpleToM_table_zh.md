# SimpleToM — 指标定义（全中文版）

- 指标层级与 `metrics.json` 的 `avg_metrics.dimensions` 树一一对应：
  **一级**=总体 accuracy；**二级**=各维度；**三级 / 四级**=维度内嵌套子维度。
- `切分型`：把数据集切成多个 split，各 split 一条准确率；`汇总型`：任务特有口径（如配对联合、宏平均、set 级 ALL），单值无法从边际准确率反推。
- 本页只列指标定义与逐值释义，不含具体数值。
- 本页为全中文版：指标名 / 字段值一律中文（英文标识符括注在后）。英文技术版见 [SimpleToM_table.md](SimpleToM_table.md)。

## 一级指标

| 指标 | 定义 |
|---|---|
| 准确率（accuracy） | 一级指标。全部样本的总体准确率 = correct / total。open 题按各数据集判分模式（f1 / rubric）二值化后计入。 |
| 答对数（correct） | 答对样本数。 |
| 样本总数（total） | 参与评测的样本总数。 |
| 抽取失败数（extraction_failed） | 答案抽取失败数（MCQ 严格模式下未输出 \boxed{} 即记为抽取失败）。 |
| 抽取失败率（extraction_failed_rate） | 抽取失败率 = extraction_failed / total，衡量模型对输出格式的遵循程度。 |

## 指标层级总表

> 横轴为指标层级；同一行表示嵌套归属（如 `dim1 → dim2 → dim3` 表示 dim2 是 dim1 的子维度、dim3 又是 dim2 的子维度）。`—` 表示该维度没有更深层级。

| 一级指标 | 二级指标 | 三级指标 | 四级指标 |
|---|---|---|---|
| `accuracy` | `type` | — | — |
|  | `dimension` | — | — |
|  | `qa_type` | — | — |
|  | `scenario_name` | — | — |

## 各维度定义

### 二级指标 · 题型（type）（切分型，单位 ACC）

题型维度（所有数据集固定带）。按 prompt_type 切分，每个 split 是该题型的准确率。

- 单选（`mcq_single`）：仅一个正确项 + 干扰项的选择题
- 多选（`mcq_multi`）：有 ≥2 个正确项的选择题
- 捆绑判分（`mcq_grouped`）：一条 prompt 内含多个子问题，全部答对才计为对（如 EmoBench EU 的情绪+原因）
- 开放题（`open`）：无干扰项、自由作答，按 f1 / rubric 判分后二值化

### 二级指标 · 考察维度（dimension）（切分型，单位 ACC）

按考察维度切分（meta.dimension）。

- 信息可达（`information_access`）：角色能否获知关键信息
- 行为预测（`behavior_prediction`）：预测角色在（可能错误的）信念下的行为
- 社会判断（`social_judgment`）：对角色行为的社会评价

### 二级指标 · 题型（qa_type）（切分型，单位 ACC）

按问答类型切分（meta.qa_type）。

- 心理状态（`mental_state`）：关于角色心理状态的题
- 行为（`behavior`）：关于角色行为的题
- 判断（`judgment`）：社会判断题

### 二级指标 · 情景（scenario_name）（切分型，单位 ACC）

按日常情景切分（meta.scenario_name），共 10 种。

- 服务业幕后（`behind_the_scene_service_industry`）：服务行业不为顾客所见的幕后环节
- 超市食品（`food_item_in_grocery_store`）：超市货架上的食品状况
- 隐藏身体特征（`hidden_body_part_feature`）：被遮挡而不可见的身体特征
- 私人物品容器内（`inside_containers_for_personal_belongings`）：私人物品容器内部的东西
- 重用标签容器内（`inside_reuse_labeled_containers`）：重复使用、标签与内容物不符的容器
- 上锁设备/账户（`locked_devices_accounts`）：上锁的设备或账户内部
- 医疗方信息（`provider_info_healthcare`）：医疗服务提供方的隐含信息
- 二手卖家信息（`seller_info_in_second_hand_market`）：二手市场卖家的隐含信息
- 名不副实标签（`true_property_pretentious_labels`）：标签夸大时的真实属性
- 无人目击不当行为（`unobserved_unethical_actions`）：无人目击时的不当行为
