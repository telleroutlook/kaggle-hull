"""数据加载和预处理工具，包含高效CSV读取优化"""

from __future__ import annotations

import os
import numpy as np
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .env import DataPaths, detect_run_environment, get_data_paths

TRAIN_REQUIRED_COLS = [
    "date_id",
    "forward_returns",
    "risk_free_rate",
    "market_forward_excess_returns",
]
TEST_REQUIRED_COLS = ["date_id", "is_scored"]
FLOAT_PREFIXES = ("M", "E", "I", "P", "V", "S", "MOM", "D")
LAG_FEATURE_SOURCES = {
    "lagged_forward_returns": "forward_returns",
    "lagged_risk_free_rate": "risk_free_rate",
    "lagged_market_forward_excess_returns": "market_forward_excess_returns",
}
TRUE_STRINGS = {"1", "true", "yes", "on"}


def _get_env_usecols() -> Optional[List[str]]:
    raw = os.getenv("HULL_DATA_USECOLS")
    if not raw:
        return None
    values = [col.strip() for col in raw.split(",") if col.strip()]
    return values or None


def _resolve_usecols(required: Iterable[str], requested: Optional[Iterable[str]]) -> Optional[List[str]]:
    base = list(requested) if requested is not None else _get_env_usecols()
    if not base:
        return None
    ordered: List[str] = []
    seen = set()
    for col in list(required) + list(base):
        if col and col not in seen:
            ordered.append(col)
            seen.add(col)
    return ordered


def _get_chunk_size() -> int:
    raw = os.getenv("HULL_CSV_CHUNKSIZE")
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _should_use_polars() -> bool:
    flag = os.getenv("HULL_USE_POLARS", "0").lower()
    return flag in {"1", "true", "yes"}


def _env_flag(var_name: str, default: bool = False) -> bool:
    raw = os.getenv(var_name)
    if raw is None:
        return default
    return raw.lower() in TRUE_STRINGS


def _env_float(var_name: str, default: float) -> float:
    raw = os.getenv(var_name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@lru_cache(maxsize=4)
def _read_header(csv_path: Path) -> List[str]:
    return pd.read_csv(csv_path, nrows=0).columns.tolist()


def _guess_dtype(column: str) -> str:
    lowered = column.lower()
    if column == "date_id" or lowered.endswith("_id"):
        return "int32"
    if lowered.startswith("is_") or lowered.endswith("_flag"):
        return "int8"
    if lowered.startswith("lagged_") or lowered.endswith("_returns"):
        return "float32"
    if any(column.startswith(prefix) for prefix in FLOAT_PREFIXES):
        return "float32"
    if column in {"risk_free_rate", "market_forward_excess_returns"}:
        return "float32"
    return "float32"


def _build_dtype_map(csv_path: Path, usecols: Optional[Iterable[str]]) -> Dict[str, str]:
    columns = list(usecols) if usecols is not None else _read_header(csv_path)
    return {col: _guess_dtype(col) for col in columns}


def _cast_dtypes(df: pd.DataFrame, dtype_map: Dict[str, str]) -> pd.DataFrame:
    for col, dtype in dtype_map.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype, copy=False)
            except ValueError:
                # 兼容极端情况下的类型转换失败
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _read_with_polars(csv_path: Path, usecols: Optional[List[str]]) -> Optional[pd.DataFrame]:
    if not _should_use_polars():
        return None
    try:
        import polars as pl
    except ImportError:
        print("⚠️ HULL_USE_POLARS=1 但未安装polars，退回pandas读取")
        return None

    df = pl.read_csv(csv_path, columns=usecols)
    return df.to_pandas(use_pyarrow_extension=True)


def _read_csv_fast(csv_path: Path, *, usecols: Optional[List[str]]) -> pd.DataFrame:
    dtype_map = _build_dtype_map(csv_path, usecols)
    polars_df = _read_with_polars(csv_path, usecols)
    if polars_df is not None:
        return _cast_dtypes(polars_df, dtype_map)

    read_kwargs = {
        "dtype": dtype_map,
        "usecols": usecols,
        "low_memory": False,
    }
    chunk_size = _get_chunk_size()
    if chunk_size > 0:
        iterator = pd.read_csv(csv_path, chunksize=chunk_size, **read_kwargs)
        df = pd.concat(iterator, ignore_index=True)
    else:
        df = pd.read_csv(csv_path, **read_kwargs)
    return df


def _log_df_stats(label: str, df: pd.DataFrame) -> None:
    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"{label}数据形状: {df.shape}, 约 {memory_mb:.2f} MB")


def _ensure_data_paths(data_paths: Optional[DataPaths]) -> DataPaths:
    if data_paths is not None:
        return data_paths
    env = detect_run_environment()
    return get_data_paths(env)


def ensure_lagged_feature_parity(
    df: pd.DataFrame,
    *,
    sort_column: str = "date_id",
) -> pd.DataFrame:
    """Ensure train data carries the lagged columns present in the test feed."""

    if df.empty:
        for new_col in LAG_FEATURE_SOURCES:
            if new_col not in df.columns:
                df[new_col] = pd.Series(dtype="float32")
        return df

    result = df.copy()
    if sort_column in result.columns:
        order = result[sort_column].sort_values(kind="mergesort").index
    else:
        order = result.index

    for new_col, source_col in LAG_FEATURE_SOURCES.items():
        if new_col in result.columns or source_col not in result.columns:
            continue
        lagged_series = (
            pd.to_numeric(result.loc[order, source_col], errors="coerce")
            .shift(1)
            .astype("float32")
        )
        result.loc[order, new_col] = lagged_series.values

    return result


def load_train_data(
    data_paths: Optional[DataPaths] = None,
    *,
    columns: Optional[Iterable[str]] = None,
    augment_data: bool = False,
    augmentation_factor: float = 0.1,
) -> pd.DataFrame:
    """加载训练数据，自动应用dtype/usecols优化"""

    paths = _ensure_data_paths(data_paths)
    requested_cols = list(columns) if columns is not None else None
    usecols = _resolve_usecols(TRAIN_REQUIRED_COLS, requested_cols)
    print(f"加载训练数据: {paths.train_data}")
    df = _read_csv_fast(paths.train_data, usecols=usecols)
    df = ensure_lagged_feature_parity(df)
    
    # 数据增强
    if augment_data:
        df = _augment_data(df, augmentation_factor)
    
    _log_df_stats("训练", df)
    return df


def load_training_frame(
    data_paths: Optional[DataPaths] = None,
    *,
    columns: Optional[Iterable[str]] = None,
    augment: Optional[bool] = None,
    augmentation_factor: Optional[float] = None,
    return_metadata: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, float | bool]]:
    """Shared helper for training/inference to load and annotate the train frame."""

    resolved_augment = bool(augment) if augment is not None else _env_flag("HULL_AUGMENT_DATA", False)
    resolved_factor = augmentation_factor
    if resolved_factor is None:
        resolved_factor = _env_float("HULL_AUGMENTATION_FACTOR", 0.05)
    resolved_factor = max(0.0, resolved_factor)
    df = load_train_data(
        data_paths,
        columns=columns,
        augment_data=resolved_augment,
        augmentation_factor=resolved_factor,
    )
    metadata: Dict[str, float | bool] = {
        "augment_data": resolved_augment,
        "augmentation_factor": resolved_factor if resolved_augment else 0.0,
    }
    df.attrs["hull_training_metadata"] = metadata
    if return_metadata:
        return df, metadata
    return df


def _augment_data(df: pd.DataFrame, factor: float = 0.1) -> pd.DataFrame:
    """数据增强：添加少量噪声以提高模型泛化能力"""
    
    if factor <= 0:
        return df
    
    df_aug = df.copy()
    
    # 获取数值列
    numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
    
    # 移除不需要增强的列
    exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    for col in feature_cols:
        if col in df_aug.columns:
            # 添加基于列标准差的小幅噪声
            col_std = df_aug[col].std()
            if pd.notna(col_std) and col_std > 0:
                noise = pd.Series(
                    np.random.normal(0, col_std * factor, size=len(df_aug)),
                    index=df_aug.index
                )
                df_aug[col] = df_aug[col] + noise
    
    return df_aug


def load_test_data(
    data_paths: Optional[DataPaths] = None,
    *,
    columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """加载测试数据"""

    paths = _ensure_data_paths(data_paths)
    requested_cols = list(columns) if columns is not None else None
    usecols = _resolve_usecols(TEST_REQUIRED_COLS, requested_cols)
    print(f"加载测试数据: {paths.test_data}")
    df = _read_csv_fast(paths.test_data, usecols=usecols)
    _log_df_stats("测试", df)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """获取特征列名"""
    
    # 排除目标变量和其他非特征列
    exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 
                   'market_forward_excess_returns', 'is_scored']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def get_target_columns() -> list:
    """获取目标变量列名"""
    
    return ['forward_returns', 'risk_free_rate', 'market_forward_excess_returns']


def validate_data(df: pd.DataFrame, data_type: str = "train") -> bool:
    """验证数据完整性"""
    
    required_cols = ['date_id']
    
    if data_type == "train":
        required_cols.extend(['forward_returns', 'risk_free_rate', 'market_forward_excess_returns'])
    elif data_type == "test":
        required_cols.extend(['is_scored'])
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"警告: 缺少必要的列: {missing_cols}")
        return False
    
    print(f"✅ {data_type}数据验证通过")
    return True


__all__ = [
    "load_train_data",
    "load_training_frame",
    "load_test_data", 
    "get_feature_columns",
    "get_target_columns",
    "validate_data",
    "ensure_lagged_feature_parity",
]
