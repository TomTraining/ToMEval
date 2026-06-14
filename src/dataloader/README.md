# DataLoader

从 `datasets/` 目录加载数据集（该目录通过 `experiment_config.yaml` 中的 `normalized_datasets_path` 配置）。

## 使用方法

```python
from src.dataloader import load_dataset, list_subsets

# 列出所有可用子集
subsets = list_subsets()
# ['ExploreToM/SIP/raw', 'ExploreToM/SIP/all_wrong', 'SocialIQA/SIP/all_wrong', ...]

# 加载特定子集
data = load_dataset("ExploreToM/SIP/raw")
print(len(data))  # 数据条数
print(data[0].keys())  # 字段名
```

## 数据集目录结构

```
datasets/
├── ExploreToM/
│   └── SIP/
│       ├── raw/
│       │   ├── dataset_info.json
│       │   └── data-00000-of-00001.arrow
│       └── all_wrong/
│           ├── dataset_info.json
│           └── data-00000-of-00001.arrow
├── SocialIQA/
│   └── SIP/
│       └── all_wrong/
│           ├── dataset_info.json
│           └── data-00000-of-00001.arrow
├── ToMBench/
│   ├── test/
│   │   ├── dataset_info.json
│   │   └── data-00000-of-00001.arrow
│   ├── train/
│   └── SIP/
├── ToMato/
│   ├── test/
│   ├── train/
│   └── SIP/
└── ToMQA/
    ├── test/
    ├── train/
    └── validation/
```

## API

### `load_dataset(subset, datasets_root=None)`

加载数据集。

- `subset` (str): 子集路径，如 "ToMBench/test"、"ToMato/test"
- `datasets_root` (可选): 自定义根目录，默认为 "datasets"

返回 `List[Dict]`

### `list_subsets(datasets_root=None)`

列出所有可用子集。

返回 `List[str]`

## 数据集路径说明

**重要**：数据集配置文件中的 `path` 字段应该使用相对路径，格式为 `{dataset_name}/{split}`，例如：

```yaml
# tasks/ToMBench/config.yaml
dataset: ToMBench
path: ToMBench/test  # 注意：没有 tasks/ 前缀
```

这样 `load_dataset` 会正确加载 `datasets/ToMBench/test/` 目录下的数据。
