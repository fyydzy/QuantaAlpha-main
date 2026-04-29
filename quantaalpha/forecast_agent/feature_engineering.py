"""月度燃气预测的特征工程工具函数。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantaalpha.forecast_agent.data import MONTH_COL, TARGET_COL


DEFAULT_HDD_BASE_TEMP = 15.0
WEATHER_INPUT_COLS = ("avg_temp", "max_temp", "min_temp")
WEATHER_FEATURE_COLS = ("avg_temp", "max_temp", "min_temp", "HDD", "temp_range")
SARIMAX_EXOG_COLS = WEATHER_FEATURE_COLS


def add_weather_features(
    df: pd.DataFrame,
    *,
    avg_temp_col: str = "avg_temp",
    max_temp_col: str = "max_temp",
    min_temp_col: str = "min_temp",
    hdd_base_temp: float = DEFAULT_HDD_BASE_TEMP,
    inplace: bool = False,
) -> pd.DataFrame:
    """添加 SARIMAX 外生变量会用到的天气特征。"""
    required = {avg_temp_col, max_temp_col, min_temp_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少温度列，无法进行特征工程: {sorted(missing)}")

    out = df if inplace else df.copy()
    avg_temp = pd.to_numeric(out[avg_temp_col], errors="coerce")
    max_temp = pd.to_numeric(out[max_temp_col], errors="coerce")
    min_temp = pd.to_numeric(out[min_temp_col], errors="coerce")

    out["HDD"] = (hdd_base_temp - avg_temp).clip(lower=0.0)
    out["temp_range"] = max_temp - min_temp
    return out


def add_time_features(
    df: pd.DataFrame,
    *,
    month_col: str = MONTH_COL,
    inplace: bool = False,
) -> pd.DataFrame:
    """添加趋势、月份周期编码与供暖季标记。"""
    if month_col not in df.columns:
        raise ValueError(f"缺少时间列: {month_col}")

    out = df if inplace else df.copy()
    dates = pd.to_datetime(out[month_col].astype(str))
    months = dates.dt.month

    out["time_index"] = np.arange(1, len(out) + 1)
    out["month_sin"] = np.sin(2 * np.pi * months / 12)
    out["month_cos"] = np.cos(2 * np.pi * months / 12)
    out["is_heating_season"] = months.isin([11, 12, 1, 2, 3]).astype(int)
    return out


def add_lag_features(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    month_col: str = MONTH_COL,
    lags: tuple[int, ...] = (12,),
    inplace: bool = False,
) -> pd.DataFrame:
    """添加目标列滞后项。"""
    required = {target_col, month_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少目标列或时间列: {sorted(missing)}")

    out = df if inplace else df.copy()
    out = out.sort_values(month_col)
    for lag in lags:
        out[f"Lag_{lag}"] = out[target_col].shift(lag)
    return out


def add_interaction_features(
    df: pd.DataFrame,
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """添加基于 HDD、供暖季与 12 月滞后的交互特征。"""
    required = {"HDD", "Lag_12", "is_heating_season"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少前置依赖特征: {sorted(missing)}")

    out = df if inplace else df.copy()
    out["HDD_squared"] = out["HDD"] ** 2
    out["HDD_cross_Lag_12"] = out["HDD"] * out["Lag_12"]
    out["HDD_cross_HeatingSeason"] = out["HDD"] * out["is_heating_season"]
    return out


def build_features_pipeline(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    month_col: str = MONTH_COL,
    dropna: bool = True,
) -> pd.DataFrame:
    """完整特征工程流水线，适合需要时间/滞后/交互特征的模型使用。"""
    out = df.sort_values(month_col).reset_index(drop=True)
    out = add_weather_features(out)
    out = add_time_features(out, month_col=month_col)
    out = add_lag_features(out, target_col=target_col, month_col=month_col)
    out = add_interaction_features(out)
    if dropna:
        out = out.dropna().reset_index(drop=True)
    return out


__all__ = [
    "DEFAULT_HDD_BASE_TEMP",
    "WEATHER_INPUT_COLS",
    "WEATHER_FEATURE_COLS",
    "SARIMAX_EXOG_COLS",
    "add_weather_features",
    "add_time_features",
    "add_lag_features",
    "add_interaction_features",
    "build_features_pipeline",
]