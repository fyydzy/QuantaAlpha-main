"""Forecast Agent 数据层：旬度 Excel（``date`` 为旬开始日）读取、切分与评估指标。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantaalpha.forecast_agent.framework import ForecastTask

# Excel 主时间列为旬开始日期；``month`` 为派生的内部旬标签 YYYY-MM-{1|2|3}
PERIODS_PER_MONTH = 3
MIN_HISTORY_PERIODS = 24 * PERIODS_PER_MONTH
DEFAULT_VAL_PERIODS = 12 * PERIODS_PER_MONTH

AS_OF_DATE = "2025-06-21"
TEST_START = "2025-11-01"
TEST_END = "2026-03-21"
AS_OF_PERIOD = AS_OF_DATE

AS_OF_MONTH = AS_OF_PERIOD
TEST_START_MONTH = TEST_START
TEST_END_MONTH = TEST_END

TARGET_COL = "gas_sales"
DATE_COL = "date"
MONTH_COL = "month"
PERIOD_COL = MONTH_COL

PROCESSED_PROVINCE = "河北"

_DECADE_CHAR_MAP = {
    "上": 1,
    "上旬": 1,
    "中": 2,
    "中旬": 2,
    "下": 3,
    "下旬": 3,
}


def date_to_period(ts: pd.Timestamp) -> str:
    """由旬开始日推导旬标签（1–10 日上旬，11–20 日中旬，21 日起下旬）。"""
    t = pd.Timestamp(ts).normalize()
    day = int(t.day)
    if day <= 10:
        ten = 1
    elif day <= 20:
        ten = 2
    else:
        ten = 3
    return f"{t.year:04d}-{t.month:02d}-{ten}"


def cli_date_to_period(value: Any, field_name: str = "date") -> str:
    """CLI / ForecastTask 边界只接受真实旬开始日，不接受 YYYY-MM-1/2/3 标签。"""
    raw = str(value).strip()
    if not raw:
        raise ValueError(f"{field_name} 不能为空")
    if re.fullmatch(r"\d{4}-\d{1,2}-[123]", raw):
        raise ValueError(
            f"{field_name} 请使用旬开始日 YYYY-MM-DD（如 2025-06-21），"
            f"不要使用旬标签 {raw!r}"
        )

    ts = pd.to_datetime(raw, errors="raise")
    ts = pd.Timestamp(ts).normalize()
    if int(ts.day) not in (1, 11, 21):
        raise ValueError(
            f"{field_name} 必须是旬开始日（每月 1/11/21 日），当前为 {ts:%Y-%m-%d}"
        )
    return date_to_period(ts)


def period_to_start_date(period: str) -> pd.Timestamp:
    """旬标签 -> 旬开始日（与 date_to_period 默认规则一致）。"""
    year, month, ten = parse_period(period)
    day = {1: 1, 2: 11, 3: 21}[ten]
    return pd.Timestamp(year=year, month=month, day=day)


def normalize_period(value: Any) -> str:
    """内部旬标签规范化；CLI 入参请使用 ``cli_date_to_period``。"""
    if isinstance(value, pd.Timestamp):
        return date_to_period(value)

    raw = str(value).strip()
    if not raw:
        raise ValueError("空的时间标签")

    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d)", raw)
    if m:
        year, month, ten = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if ten not in (1, 2, 3):
            raise ValueError(f"旬编号须为 1/2/3: {value}")
        return f"{year:04d}-{month:02d}-{ten}"

    m = re.fullmatch(r"(\d{4})-(\d{1,2})([上中下])", raw)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        ten = _DECADE_CHAR_MAP[m.group(3)]
        return f"{year:04d}-{month:02d}-{ten}"

    m = re.fullmatch(r"(\d{4})-(\d{1,2})(上旬|中旬|下旬)", raw)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        ten = _DECADE_CHAR_MAP[m.group(3)]
        return f"{year:04d}-{month:02d}-{ten}"

    m = re.fullmatch(r"(\d{4})(\d{2})([123])", raw)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3))}"

    m = re.fullmatch(r"(\d{4})-(\d{1,2})", raw)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-3"

    try:
        ts = pd.to_datetime(raw, errors="raise")
        if re.search(r"\d{4}", raw):
            return date_to_period(ts)
    except (ValueError, TypeError):
        pass

    raise ValueError(f"无法解析内部旬标签或旬开始日: {value!r}")


def parse_period(period: str) -> tuple[int, int, int]:
    norm = normalize_period(period)
    year, month, ten = norm.split("-")
    return int(year), int(month), int(ten)


def _period_key(period: str) -> tuple[int, int, int]:
    return parse_period(period)


def period_le(a: str, b: str) -> bool:
    return _period_key(normalize_period(a)) <= _period_key(normalize_period(b))


def period_between(value: str, start: str, end: str) -> bool:
    key = _period_key(normalize_period(value))
    return _period_key(normalize_period(start)) <= key <= _period_key(normalize_period(end))


def _format_period(year: int, month: int, ten: int) -> str:
    if ten not in (1, 2, 3):
        raise ValueError(f"旬编号须为 1/2/3: {ten}")
    return f"{year:04d}-{month:02d}-{ten}"


def _advance_period(year: int, month: int, ten: int) -> tuple[int, int, int]:
    if ten < 3:
        return year, month, ten + 1
    if month < 12:
        return year, month + 1, 1
    return year + 1, 1, 1


def _retreat_period(year: int, month: int, ten: int) -> tuple[int, int, int]:
    if ten > 1:
        return year, month, ten - 1
    if month > 1:
        return year, month - 1, 3
    return year - 1, 12, 3


def period_after(period: str, steps: int = 1) -> str:
    y, m, t = parse_period(period)
    for _ in range(steps):
        y, m, t = _advance_period(y, m, t)
    return _format_period(y, m, t)


def period_before(period: str, steps: int = 1) -> str:
    y, m, t = parse_period(period)
    for _ in range(steps):
        y, m, t = _retreat_period(y, m, t)
    return _format_period(y, m, t)


def period_range(start: str, end: str) -> list[str]:
    s = normalize_period(start)
    e = normalize_period(end)
    if _period_key(s) > _period_key(e):
        return []

    out: list[str] = []
    y, m, t = parse_period(s)
    ey, em, et = parse_period(e)
    while (y, m, t) <= (ey, em, et):
        out.append(_format_period(y, m, t))
        y, m, t = _advance_period(y, m, t)
    return out


def month_range(start: str, end: str) -> list[str]:
    return period_range(start, end)


def _attach_period_from_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    if out[DATE_COL].isna().any():
        raise ValueError(f"{DATE_COL} 列存在无法解析的日期")
    out[MONTH_COL] = out[DATE_COL].map(date_to_period)
    return out.sort_values(DATE_COL).reset_index(drop=True)


def _read_excel_base(path: str, columns: list[str]) -> pd.DataFrame:
    raw = pd.read_excel(path)
    if DATE_COL in raw.columns:
        missing = set(columns) - set(raw.columns)
        if DATE_COL not in columns:
            missing.discard(DATE_COL)
        if missing:
            raise ValueError(f"文件缺少列: {sorted(missing)}")
        out = raw[[c for c in columns if c in raw.columns]].copy()
        if DATE_COL not in out.columns:
            out[DATE_COL] = raw[DATE_COL]
        return _attach_period_from_date(out)

    if MONTH_COL in raw.columns:
        missing = set(columns) - set(raw.columns) - {DATE_COL}
        if missing:
            raise ValueError(f"文件缺少列: {sorted(missing)}")
        out = raw[[c for c in columns if c in raw.columns]].copy()
        out[MONTH_COL] = out[MONTH_COL].astype(str).map(normalize_period)
        out[DATE_COL] = out[MONTH_COL].map(period_to_start_date)
        return out.sort_values(DATE_COL).reset_index(drop=True)

    raise ValueError(f"文件须包含 {DATE_COL}（旬开始日）或 {MONTH_COL}（旬标签）列")


def normalize_period_column(df: pd.DataFrame, col: str = MONTH_COL) -> pd.DataFrame:
    if col == DATE_COL or DATE_COL in df.columns:
        return _attach_period_from_date(df)
    out = df.copy()
    out[col] = out[col].map(normalize_period)
    if DATE_COL not in out.columns:
        out[DATE_COL] = out[col].map(period_to_start_date)
    return out.sort_values(DATE_COL).reset_index(drop=True)


def find_processed_excel(province: str | None = None) -> str:
    prov = PROCESSED_PROVINCE if province is None else province
    candidates = [
        os.path.join("data", "processed_data", f"{prov}.xlsx"),
        os.path.join("processed_data", f"{prov}.xlsx"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"未找到 {prov} 的 processed 文件，请确认 data/processed_data/{prov}.xlsx 存在。"
    )


def load_gas_series(path: str) -> pd.DataFrame:
    out = _read_excel_base(path, [DATE_COL, TARGET_COL])
    out[TARGET_COL] = pd.to_numeric(out[TARGET_COL], errors="coerce")
    out = out.dropna(subset=[TARGET_COL, DATE_COL])
    return out[[DATE_COL, MONTH_COL, TARGET_COL]]


def load_tabular_feature_frame(path: str) -> pd.DataFrame:
    from quantaalpha.forecast_agent.feature_engineering import (
        WEATHER_INPUT_COLS,
        build_features_pipeline,
    )

    cols = [DATE_COL, TARGET_COL, *WEATHER_INPUT_COLS]
    out = _read_excel_base(path, cols)
    numeric_cols = [TARGET_COL, *WEATHER_INPUT_COLS]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=[DATE_COL, TARGET_COL, *WEATHER_INPUT_COLS])
    return build_features_pipeline(
        out,
        target_col=TARGET_COL,
        date_col=DATE_COL,
        month_col=MONTH_COL,
        dropna=True,
    )


def split_asof_test(
    df: pd.DataFrame,
    as_of_period: str = AS_OF_PERIOD,
    test_start: str = TEST_START,
    test_end: str = TEST_END,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    as_of_period = normalize_period(as_of_period)
    test_start = normalize_period(test_start)
    test_end = normalize_period(test_end)

    train = df[df[MONTH_COL].map(lambda x: period_le(str(x), as_of_period))].copy()
    test = df[df[MONTH_COL].map(lambda x: period_between(str(x), test_start, test_end))].copy()

    bridge_start = period_after(as_of_period)
    bridge_end = period_before(test_start)
    bridge_periods = (
        period_range(bridge_start, bridge_end)
        if period_le(bridge_start, bridge_end)
        else []
    )

    if train.empty:
        raise ValueError("训练集为空，请检查 as_of 或数据范围。")
    if test.empty:
        raise ValueError("测试集为空，请检查 test_start / test_end 或数据范围。")
    return train, test, bridge_periods


def split_asof_forecast_periods(
    df: pd.DataFrame,
    as_of_period: str = AS_OF_PERIOD,
    test_start: str = TEST_START,
    test_end: str = TEST_END,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train, test, bridge_periods = split_asof_test(df, as_of_period, test_start, test_end)
    forecast_periods = bridge_periods + [normalize_period(x) for x in test[MONTH_COL].astype(str)]
    return train, test, forecast_periods


def split_asof_forecast_months(
    df: pd.DataFrame,
    as_of_month: str = AS_OF_PERIOD,
    test_start: str = TEST_START,
    test_end: str = TEST_END,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    return split_asof_forecast_periods(df, as_of_month, test_start, test_end)


def filter_through_period(df: pd.DataFrame, through: str) -> pd.DataFrame:
    """保留旬标签 <= through 的行，按 ``date`` 排序。"""
    sort_col = DATE_COL if DATE_COL in df.columns else MONTH_COL
    return (
        df[df[MONTH_COL].astype(str).map(lambda x: period_le(str(x), through))]
        .sort_values(sort_col)
        .copy()
    )


def filter_periods_in(df: pd.DataFrame, periods: list[str]) -> pd.DataFrame:
    allowed = {normalize_period(p) for p in periods}
    sort_col = DATE_COL if DATE_COL in df.columns else MONTH_COL
    return (
        df[df[MONTH_COL].astype(str).map(normalize_period).isin(allowed)]
        .sort_values(sort_col)
        .copy()
    )


def tabular_feature_columns(df: pd.DataFrame) -> list[str]:
    """返回可直接喂给 sklearn/GBDT 的数值特征列。"""
    excluded = {DATE_COL, MONTH_COL, TARGET_COL}
    return [
        c
        for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]


def period_start_dates(df: pd.DataFrame, periods: list[str]) -> pd.DatetimeIndex:
    """按旬标签从表中取真实旬开始日（用于 TimesFM / 作图）。"""
    lookup = (
        df.drop_duplicates(MONTH_COL)
        .set_index(MONTH_COL)[DATE_COL]
        .astype("datetime64[ns]")
    )
    dates: list[pd.Timestamp] = []
    for p in periods:
        norm = normalize_period(p)
        if norm in lookup.index:
            dates.append(pd.Timestamp(lookup.loc[norm]))
        else:
            dates.append(period_to_start_date(norm))
    return pd.DatetimeIndex(dates)


def compute_score_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    err = y_true_arr - y_pred_arr

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    denom = np.abs(y_true_arr) + np.abs(y_pred_arr) + 1e-6
    smape = float(np.mean(2.0 * np.abs(err) / denom) * 100.0)

    non_zero = np.where(y_true_arr != 0, y_true_arr, np.nan)
    mape = float(np.nanmean(np.abs(err / non_zero)) * 100.0)
    if not np.isfinite(mape):
        mape = float("inf")

    bias = float(np.mean(y_pred_arr - y_true_arr))
    scale = float(np.mean(np.abs(y_true_arr)) + 1e-6)
    score = 0.7 * smape + 0.3 * (rmse / scale * 100.0)
    return {
        "score": score,
        "smape": smape,
        "mape": mape,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
    }


def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics = compute_score_metrics(y_true, y_pred)
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    err = y_true_arr - y_pred_arr
    mse = float(np.mean(np.square(err)))
    ss_res = float(np.sum(np.square(err)))
    ss_tot = float(np.sum(np.square(y_true_arr - float(np.mean(y_true_arr))))) if len(y_true_arr) else 0.0
    r2 = float("nan")
    if ss_tot > 1e-12:
        r2 = 1.0 - ss_res / ss_tot
    return {
        "MAE": metrics["mae"],
        "MSE": mse,
        "MAPE": metrics["mape"],
        "RMSE": metrics["rmse"],
        "R2": r2,
    }


def build_test_table(
    forecast_periods: list[str],
    predictions: np.ndarray,
    test_df: pd.DataFrame,
    test_start: str = TEST_START,
    test_end: str = TEST_END,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    test_start = normalize_period(test_start)
    test_end = normalize_period(test_end)
    forecast_df = pd.DataFrame(
        {
            MONTH_COL: [normalize_period(p) for p in forecast_periods],
            "predicted_gas_sales": predictions[: len(forecast_periods)],
        }
    )
    test_norm = test_df.copy()
    test_norm[MONTH_COL] = test_norm[MONTH_COL].astype(str).map(normalize_period)
    merge_cols = [MONTH_COL, TARGET_COL]
    if DATE_COL in test_norm.columns:
        merge_cols = [DATE_COL, MONTH_COL, TARGET_COL]
    result = forecast_df.merge(
        test_norm[merge_cols],
        on=MONTH_COL,
        how="left",
    ).rename(columns={TARGET_COL: "actual_gas_sales"})

    is_test = result[MONTH_COL].map(lambda x: period_between(str(x), test_start, test_end))
    result["phase"] = np.where(is_test, "test", "bridge")
    result["error"] = result["predicted_gas_sales"] - result["actual_gas_sales"]
    result["abs_error"] = np.abs(result["error"])

    y_true = result.loc[is_test, "actual_gas_sales"].to_numpy(dtype=float)
    y_pred = result.loc[is_test, "predicted_gas_sales"].to_numpy(dtype=float)
    return result, y_true, y_pred


@dataclass
class TaskContext:
    series: pd.Series
    future_index: pd.DatetimeIndex
    test_df: pd.DataFrame
    forecast_months: list[str]
    test_start: str
    test_end: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        return len(self.forecast_months)

    @property
    def forecast_periods(self) -> list[str]:
        return self.forecast_months


def load_task_context(task: ForecastTask) -> TaskContext:
    path = str(task.excel_path) if task.excel_path is not None else find_processed_excel(task.province)
    raw_df = load_gas_series(path)

    as_of = cli_date_to_period(task.as_of_month or AS_OF_DATE, "--as-of-month")
    test_start = cli_date_to_period(task.test_start or TEST_START, "--test-start")
    test_end = cli_date_to_period(task.test_end or TEST_END, "--test-end")
    train_df, test_df, forecast_periods = split_asof_forecast_periods(
        raw_df,
        as_of_period=as_of,
        test_start=test_start,
        test_end=test_end,
    )

    train_index = pd.DatetimeIndex(train_df[DATE_COL])
    series = pd.Series(
        train_df[TARGET_COL].to_numpy(dtype=float),
        index=train_index,
        name="y",
    ).sort_index()
    series.index.name = "ds"

    return TaskContext(
        series=series,
        future_index=period_start_dates(raw_df, forecast_periods),
        test_df=test_df,
        forecast_months=list(forecast_periods),
        test_start=test_start,
        test_end=test_end,
        metadata={
            "source": "gas_decadal",
            "excel_path": path,
            "as_of_period": as_of,
            "as_of_month": as_of,
            "test_start": test_start,
            "test_end": test_end,
            "province": task.province,
            "freq": "decadal",
            "time_col": DATE_COL,
        },
    )


def build_result_table(
    ctx: TaskContext,
    forecast_values: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    return build_test_table(
        ctx.forecast_months,
        np.asarray(forecast_values, dtype=float),
        ctx.test_df,
        test_start=ctx.test_start,
        test_end=ctx.test_end,
    )


def format_period_ds_for_display(ds: pd.Series) -> pd.Series:
    return pd.to_datetime(ds).dt.strftime("%Y-%m-%d")


def format_month_ds_for_display(ds: pd.Series) -> pd.Series:
    return format_period_ds_for_display(ds)


def period_to_lstm_slot(period: str) -> int:
    _, month, ten = parse_period(period)
    return (month - 1) * PERIODS_PER_MONTH + (ten - 1)


__all__ = [
    "PERIODS_PER_MONTH",
    "MIN_HISTORY_PERIODS",
    "DEFAULT_VAL_PERIODS",
    "AS_OF_PERIOD",
    "AS_OF_DATE",
    "TEST_START",
    "TEST_END",
    "AS_OF_MONTH",
    "TARGET_COL",
    "DATE_COL",
    "MONTH_COL",
    "PERIOD_COL",
    "PROCESSED_PROVINCE",
    "date_to_period",
    "cli_date_to_period",
    "period_to_start_date",
    "normalize_period",
    "parse_period",
    "period_le",
    "period_between",
    "filter_through_period",
    "filter_periods_in",
    "tabular_feature_columns",
    "period_range",
    "month_range",
    "period_after",
    "period_before",
    "normalize_period_column",
    "period_start_dates",
    "find_processed_excel",
    "load_gas_series",
    "load_tabular_feature_frame",
    "split_asof_test",
    "split_asof_forecast_periods",
    "split_asof_forecast_months",
    "compute_score_metrics",
    "forecast_metrics",
    "build_test_table",
    "TaskContext",
    "load_task_context",
    "build_result_table",
    "format_period_ds_for_display",
    "format_month_ds_for_display",
    "period_to_lstm_slot",
]
