"""
Dataset Loader

根据名称加载 datasets 目录下的数据集。
支持 arrow 和 parquet 格式。
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class DataLoader:
    """数据集加载器"""

    def __init__(self, datasets_root: Union[str, Path] = None):
        self.datasets_root = Path(datasets_root) if datasets_root else Path(__file__).parent.parent.parent / "datasets"

    def load(self, subset: str) -> List[Dict[str, Any]]:
        """加载数据集。

        支持从本地加载 arrow 或 parquet 格式。

        Args:
            subset: 子集路径，如 "ToMBench/test"

        Returns:
            数据列表
        """
        from datasets import load_from_disk, load_dataset as hf_load_dataset

        path = self.datasets_root / subset

        # 尝试从本地 arrow 目录加载
        if path.exists():
            # 如果路径下没有 arrow 文件但有子目录，则合并加载所有子目录
            if not list(path.glob("*.arrow")):
                arrow_paths = self._find_arrow_paths(path)
                if arrow_paths:
                    result = []
                    for arrow_path in arrow_paths:
                        dataset = load_from_disk(str(arrow_path))
                        result.extend(dataset.to_list())
                    return result

            dataset = load_from_disk(str(path))
            return dataset.to_list()

        # Arrow目录不存在，尝试加载parquet文件
        # 解析subset: "DatasetName/split" -> 查找 datasets/DatasetName/{split}-*.parquet
        parts = subset.split("/")
        if len(parts) >= 2:
            dataset_name = parts[0]
            split_name = parts[1]
            dataset_dir = self.datasets_root / dataset_name

            if dataset_dir.exists():
                # 查找匹配的parquet文件
                parquet_files = list(dataset_dir.glob(f"{split_name}-*.parquet"))
                if parquet_files:
                    logger.info(f"Loading from parquet: {parquet_files}")
                    dataset = hf_load_dataset(
                        'parquet',
                        data_files=[str(f) for f in parquet_files],
                        split='train'  # parquet加载后默认是train split
                    )
                    return dataset.to_list()

        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Searched for:\n"
            f"  - Arrow directory: {path}\n"
            f"  - Parquet files: {self.datasets_root / parts[0] / (parts[1] + '-*.parquet') if len(parts) >= 2 else 'N/A'}"
        )

    def load_all(self, subset_prefix: str = "") -> List[Dict[str, Any]]:
        """加载指定路径下所有的 arrow 数据集并合并。

        Args:
            subset_prefix: 子集路径前缀，如 "ExploreToM/SIP"。
                          如果为空，则加载 datasets_root 下所有数据集。

        Returns:
            合并后的数据列表

        Examples:
            # 加载所有数据集
            loader = DataLoader()
            all_data = loader.load_all()

            # 加载特定路径下的所有数据集
            data = loader.load_all("ExploreToM/SIP")
        """
        from datasets import load_from_disk

        if subset_prefix:
            search_path = self.datasets_root / subset_prefix
        else:
            search_path = self.datasets_root

        result = []
        for subset_path in self._find_arrow_paths(search_path):
            dataset = load_from_disk(str(subset_path))
            result.extend(dataset.to_list())

        return result

    def list_subsets(self) -> List[str]:
        """列出所有可用的子集路径。"""
        subsets = []
        for ds_dir in self.datasets_root.iterdir():
            if ds_dir.is_dir() and not ds_dir.name.startswith("."):
                subsets.extend(self._find_arrow_subsets(ds_dir, ds_dir.name))
        return subsets

    def _find_arrow_subsets(self, path: Path, prefix: str) -> List[str]:
        """递归查找包含 arrow 文件的目录。"""
        subsets = []
        arrow_files = list(path.glob("*.arrow"))
        if arrow_files:
            subsets.append(prefix)
        for sub_dir in path.iterdir():
            if sub_dir.is_dir():
                subsets.extend(self._find_arrow_subsets(sub_dir, f"{prefix}/{sub_dir.name}"))
        return subsets

    def _find_arrow_paths(self, path: Path) -> List[Path]:
        """递归查找包含 arrow 文件的目录，返回 Path 列表。"""
        arrow_paths = []
        if list(path.glob("*.arrow")):
            arrow_paths.append(path)
        for sub_dir in path.iterdir():
            if sub_dir.is_dir():
                arrow_paths.extend(self._find_arrow_paths(sub_dir))
        return arrow_paths


def load_dataset(subset: str, datasets_root: Union[str, Path] = None) -> List[Dict[str, Any]]:
    """便捷函数：加载数据集。

    Args:
        subset: 子集路径，如 "ExploreToM/SIP/raw"
        datasets_root: 数据集根目录

    Returns:
        数据列表

    Examples:
        data = load_dataset("ExploreToM/SIP/raw")
    """
    loader = DataLoader(datasets_root)
    return loader.load(subset)


def list_subsets(datasets_root: Union[str, Path] = None) -> List[str]:
    """便捷函数：列出所有子集。"""
    loader = DataLoader(datasets_root)
    return loader.list_subsets()