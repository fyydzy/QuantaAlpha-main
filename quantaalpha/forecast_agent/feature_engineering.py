"""旬度燃气预测的特征工程工具函数。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantaalpha.forecast_agent.data import DATE_COL, MONTH_COL, PERIODS_PER_MONTH, TARGET_COL, parse_period


WEATHER_INPUT_COLS = ("avg_temp", "max_temp", "min_temp", "HDD", "extreme_cold_days")
WEATHER_FEATURE_COLS = (
    "avg_temp",
    "max_temp",
    "min_temp",
    "HDD",
    "extreme_cold_days",
    "temp_range",
)
SARIMAX_EXOG_COLS = WEATHER_FEATURE_COLS

# 12 个月 × 3 旬/月，同比滞后
DEFAULT_TARGET_LAGS = (12 * PERIODS_PER_MONTH,)


def add_weather_features(
    df: pd.DataFrame,
    *,
    avg_temp_col: str = "avg_temp",
    hdd_col: str = "HDD",
    cold_days_col: str = "extreme_cold_days",
    max_temp_col: str = "max_temp",
    min_temp_col: str = "min_temp",
    inplace: bool = False,
) -> pd.DataFrame:
    required = {avg_temp_col, hdd_col, cold_days_col, max_temp_col, min_temp_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"底层数据缺失必要的气象列: {sorted(missing)}")

    out = df if inplace else df.copy()
    out[avg_temp_col] = pd.to_numeric(out[avg_temp_col], errors="coerce")
    out[hdd_col] = pd.to_numeric(out[hdd_col], errors="coerce")
    out[cold_days_col] = pd.to_numeric(out[cold_days_col], errors="coerce")
    max_temp = pd.to_numeric(out[max_temp_col], errors="coerce")
    min_temp = pd.to_numeric(out[min_temp_col], errors="coerce")
    out["temp_range"] = max_temp - min_temp
    return out


def add_time_features(
    df: pd.DataFrame,
    *,
    date_col: str = DATE_COL,
    month_col: str = MONTH_COL,
    inplace: bool = False,
) -> pd.DataFrame:
    if date_col in df.columns:
        calendar_months = pd.to_datetime(df[date_col]).dt.month
        ten_vals = pd.to_datetime(df[date_col]).dt.day.map(lambda d: 1 if d <= 10 else (2 if d <= 20 else 3))
    elif month_col in df.columns:
        calendar_months = df[month_col].astype(str).map(lambda p: parse_period(p)[1])
        ten_vals = df[month_col].astype(str).map(lambda p: parse_period(p)[2])
    else:
        raise ValueError(f"缺少时间列: {date_col} 或 {month_col}")

    out = df if inplace else df.copy()
    out["time_index"] = np.arange(1, len(out) + 1)
    out["month_sin"] = np.sin(2 * np.pi * calendar_months / 12)
    out["month_cos"] = np.cos(2 * np.pi * calendar_months / 12)
    out["ten_sin"] = np.sin(2 * np.pi * ten_vals / 3)
    out["ten_cos"] = np.cos(2 * np.pi * ten_vals / 3)
    out["is_heating_season"] = calendar_months.isin([11, 12, 1, 2, 3]).astype(int)
    return out


def add_lag_features(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    date_col: str = DATE_COL,
    month_col: str = MONTH_COL,
    lags: tuple[int, ...] = DEFAULT_TARGET_LAGS,
    inplace: bool = False,
) -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError(f"缺少目标列: {target_col}")
    sort_col = date_col if date_col in df.columns else month_col
    if sort_col not in df.columns:
        raise ValueError(f"缺少时间列: {date_col} 或 {month_col}")

    out = df if inplace else df.copy()
    out = out.sort_values(sort_col)
    for lag in lags:
        out[f"Lag_{lag}"] = out[target_col].shift(lag)
    return out


def add_lag_rolling_features(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    month_col: str = MONTH_COL,
    lags: tuple[int, ...] = DEFAULT_TARGET_LAGS,
    inplace: bool = False,
) -> pd.DataFrame:
    return add_lag_features(
        df,
        target_col=target_col,
        month_col=month_col,
        lags=lags,
        inplace=inplace,
    )


def add_interaction_features(
    df: pd.DataFrame,
    *,
    cold_days_col: str = "extreme_cold_days",
    lag_periods: int = 12 * PERIODS_PER_MONTH,
    inplace: bool = False,
) -> pd.DataFrame:
    lag_col = f"Lag_{lag_periods}"
    required = {"HDD", cold_days_col, lag_col, "is_heating_season"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少前置依赖特征: {sorted(missing)}")

    out = df if inplace else df.copy()
    out["HDD_squared"] = out["HDD"] ** 2
    out[f"HDD_cross_{lag_col}"] = out["HDD"] * out[lag_col]
    out["HDD_cross_HeatingSeason"] = out["HDD"] * out["is_heating_season"]
    out[f"ColdDays_cross_{lag_col}"] = out[cold_days_col] * out[lag_col]
    return out


def build_features_pipeline(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    date_col: str = DATE_COL,
    month_col: str = MONTH_COL,
    dropna: bool = True,
) -> pd.DataFrame:
    sort_col = date_col if date_col in df.columns else month_col
    out = df.sort_values(sort_col).reset_index(drop=True)
    out = add_weather_features(out)
    out = add_time_features(out, date_col=date_col, month_col=month_col)
    out = add_lag_features(out, target_col=target_col, date_col=date_col, month_col=month_col)
    out = add_interaction_features(out)
    if dropna:
        out = out.dropna().reset_index(drop=True)
    return out


__all__ = [
    "DATE_COL",
    "WEATHER_INPUT_COLS",
    "WEATHER_FEATURE_COLS",
    "SARIMAX_EXOG_COLS",
    "DEFAULT_TARGET_LAGS",
    "add_weather_features",
    "add_time_features",
    "add_lag_features",
    "add_lag_rolling_features",
    "add_interaction_features",
    "build_features_pipeline",
]
