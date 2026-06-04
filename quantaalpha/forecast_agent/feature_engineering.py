"""旬度燃气预测的特征工程工具函数。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantaalpha.forecast_agent.data import (
    DATE_COL,
    MONTH_COL,
    PERIODS_PER_MONTH,
    TARGET_COL,
    date_to_period,
    normalize_period,
    parse_period,
    period_after,
    period_le,
)


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

# 农历正月初一（公历），用于 spring_rework_peak
SPRING_FESTIVAL_DAY: dict[int, tuple[int, int]] = {
    2010: (2, 14),
    2011: (2, 3),
    2012: (1, 23),
    2013: (2, 10),
    2014: (1, 31),
    2015: (2, 19),
    2016: (2, 8),
    2017: (1, 28),
    2018: (2, 16),
    2019: (2, 5),
    2020: (1, 25),
    2021: (2, 12),
    2022: (2, 1),
    2023: (1, 22),
    2024: (2, 10),
    2025: (1, 29),
    2026: (2, 17),
    2027: (2, 6),
    2028: (1, 26),
    2029: (2, 13),
    2030: (2, 3),
    2031: (1, 23),
    2032: (2, 11),
    2033: (1, 31),
    2034: (2, 19),
    2035: (2, 8),
}

SPRING_REWORK_PEAK_AFTER = (1.0, 0.6, 0.3)


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


def spring_festival_period(year: int) -> str:
    """返回指定公历年份春节初一所在旬标签。"""
    if year not in SPRING_FESTIVAL_DAY:
        raise ValueError(
            f"未配置 {year} 年春节日期，请在 feature_engineering.SPRING_FESTIVAL_DAY 中补充"
        )
    month, day = SPRING_FESTIVAL_DAY[year]
    return date_to_period(pd.Timestamp(year=year, month=month, day=day))


def spring_rework_peak_for_period(period: str) -> float:
    """春节复工峰值特征：春节所在旬及之前=0；其后三旬=1.0/0.6/0.3；其余=0。"""
    year, _, _ = parse_period(period)
    p = normalize_period(period)
    cny_period = spring_festival_period(year)
    if period_le(p, cny_period):
        return 0.0
    post_periods = [normalize_period(period_after(cny_period, i)) for i in (1, 2, 3)]
    for peak, post_p in zip(SPRING_REWORK_PEAK_AFTER, post_periods, strict=True):
        if p == post_p:
            return peak
    return 0.0


def add_spring_rework_features(
    df: pd.DataFrame,
    *,
    month_col: str = MONTH_COL,
    inplace: bool = False,
) -> pd.DataFrame:
    if month_col not in df.columns:
        raise ValueError(f"缺少旬标签列: {month_col}")

    out = df if inplace else df.copy()
    out["spring_rework_peak"] = (
        out[month_col].astype(str).map(lambda x: spring_rework_peak_for_period(x)).astype(float)
    )
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
    required = {"HDD", cold_days_col, lag_col, "is_heating_season", "spring_rework_peak"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少前置依赖特征: {sorted(missing)}")

    out = df if inplace else df.copy()
    out["HDD_squared"] = out["HDD"] ** 2
    out[f"HDD_cross_{lag_col}"] = out["HDD"] * out[lag_col]
    out["HDD_cross_HeatingSeason"] = out["HDD"] * out["is_heating_season"]
    out["HDD_cross_spring_rework_peak"] = out["HDD"] * out["spring_rework_peak"]
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
    out = add_spring_rework_features(out, month_col=month_col)
    out = add_lag_features(out, target_col=target_col, date_col=date_col, month_col=month_col)
    out = add_interaction_features(out)
    if dropna:
        out = out.dropna().reset_index(drop=True)
    return out


def select_feature_columns(
    df: pd.DataFrame,
    feature_set: list[str] | None,
    *,
    allowlist: set[str] | None = None,
) -> list[str]:
    """从特征工程后的表中选择训练列（用于 forecast_search）。

    - feature_set=None/[]：返回默认可用的全部数值特征（由 data.tabular_feature_columns 决定）
    - feature_set 非空：校验其必须存在于可用特征中（且可选进一步受 allowlist 限制）
    """
    from quantaalpha.forecast_agent.data import tabular_feature_columns

    all_features = tabular_feature_columns(df)
    if not feature_set:
        return all_features

    candidate = [str(x) for x in feature_set]
    if allowlist is not None:
        unknown_in_allowlist = sorted(set(candidate) - set(allowlist))
        if unknown_in_allowlist:
            raise ValueError(f"feature_set 包含不在特征池白名单中的特征: {unknown_in_allowlist}")

    missing = sorted(set(candidate) - set(all_features))
    if missing:
        raise ValueError(f"feature_set 包含不存在或不可用的特征: {missing}")
    return candidate


def resolve_feature_columns(
    feature_df: pd.DataFrame,
    feature_set: list[str] | None,
    *,
    allowlist: set[str] | None = None,
) -> list[str]:
    """统一入口：forecast / forecast_search 按 feature_set 选训练列。"""
    return select_feature_columns(feature_df, feature_set or None, allowlist=allowlist)


__all__ = [
    "DATE_COL",
    "WEATHER_INPUT_COLS",
    "WEATHER_FEATURE_COLS",
    "SARIMAX_EXOG_COLS",
    "DEFAULT_TARGET_LAGS",
    "add_weather_features",
    "add_time_features",
    "add_spring_rework_features",
    "spring_rework_peak_for_period",
    "spring_festival_period",
    "SPRING_FESTIVAL_DAY",
    "add_lag_features",
    "add_lag_rolling_features",
    "add_interaction_features",
    "select_feature_columns",
    "resolve_feature_columns",
    "build_features_pipeline",
]
