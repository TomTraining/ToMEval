# DataLoader

从 `datasets/` 目录加载数据集（该目录通过 `experiment_config.yaml` 中的 `normalized_datasets_path` 配置）。同时支持 HuggingFace `save_to_disk` 的 arrow 目录与 parquet 文件两种格式，当前仓库中的数据集均为 parquet。

## 使用方法

```python
from src.dataloader import load_dataset

# 加载特定子集：subset 形如 "{数据集名}/{split}"
data = load_dataset("ToMBench/test")
print(len(data))        # 数据条数
print(data[0].keys())   # 字段名
```

## 数据集目录结构

每个数据集是 `datasets/` 下的一个目录，split 以 parquet 文件形式存放，文件名格式为 `{split}-{shard}-of-{total}.parquet`：

```
datasets/
├── ToMBench/
│   └── test-00000-of-00001.parquet
├── ToMQA/
│   └── test-00000-of-00001.parquet
├── ToMato/
│   └── test-00000-of-00001.parquet
├── SocialIQA/
│   └── test-00000-of-00001.parquet
└── ExploreToM/
    └── test-00000-of-00001.parquet
```

`load_dataset("ToMBench/test")` 会匹配 `datasets/ToMBench/test-*.parquet` 并合并加载。若数据集改以 HuggingFace arrow 目录（含 `dataset_info.json` + `data-*.arrow`）存放，同名 `subset` 也能被自动识别。

## API

### `load_dataset(subset, datasets_root=None)`

加载数据集。

- `subset` (str): 子集路径，形如 `"ToMBench/test"`、`"ToMato/test"`。
- `datasets_root` (可选): 自定义根目录，默认为仓库根的 `datasets/`。

返回 `List[Dict]`。

### `list_subsets(datasets_root=None)`

列出以 arrow 目录形式存放的可用子集（parquet 数据集不会被枚举）。

返回 `List[str]`。

## 数据集路径说明

**重要**：数据集配置文件中的 `path` 字段应使用相对路径，格式为 `{dataset_name}/{split}`，例如：

```yaml
# tasks/ToMBench/config.yaml
dataset: ToMBench
path: ToMBench/test  # 注意：没有 tasks/ 前缀
```

这样 `load_dataset` 会正确加载 `datasets/ToMBench/test-*.parquet`。
