"""Forecast Agent 数据层：月度 Excel 燃气序列读取、切分与评估指标。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantaalpha.forecast_agent.framework import ForecastTask


AS_OF_MONTH = "2025-06"
TEST_START = "2025-11"
TEST_END = "2026-03"
TARGET_COL = "gas_sales"
MONTH_COL = "month"

PROCESSED_PROVINCE = "河北"


def month_range(start: str, end: str) -> list[str]:
    return list(pd.period_range(start=start, end=end, freq="M").astype(str))


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
    df = pd.read_excel(path)
    required = {MONTH_COL, TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"文件缺少必要列: {missing}")

    out = df[[MONTH_COL, TARGET_COL]].copy()
    out[MONTH_COL] = out[MONTH_COL].astype(str).str.slice(0, 7)
    out[TARGET_COL] = pd.to_numeric(out[TARGET_COL], errors="coerce")
    out = out.dropna(subset=[TARGET_COL])
    return out.sort_values(MONTH_COL).reset_index(drop=True)


def split_asof_test(
    df: pd.DataFrame,
    as_of_month: str = AS_OF_MONTH,
    test_start: str = TEST_START,
    test_end: str = TEST_END,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """按 as-of 切训练集，并返回测试集与桥接月份。"""
    train = df[df[MONTH_COL] <= as_of_month].copy()
    test = df[(df[MONTH_COL] >= test_start) & (df[MONTH_COL] <= test_end)].copy()

    bridge_start = str((pd.Period(as_of_month, freq="M") + 1).strftime("%Y-%m"))
    bridge_end = str((pd.Period(test_start, freq="M") - 1).strftime("%Y-%m"))
    bridge_months = month_range(bridge_start, bridge_end)

    if train.empty:
        raise ValueError("训练集为空，请检查 as_of_month 或数据范围。")
    if test.empty:
        raise ValueError("测试集为空，请检查 test_start / test_end 或数据范围。")
    return train, test, bridge_months


def split_asof_forecast_months(
    df: pd.DataFrame,
    as_of_month: str = AS_OF_MONTH,
    test_start: str = TEST_START,
    test_end: str = TEST_END,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """返回训练集、测试集、最终要一次性预测的月份（bridge + test）。"""
    train, test, bridge_months = split_asof_test(df, as_of_month, test_start, test_end)
    forecast_months = bridge_months + list(test[MONTH_COL].astype(str).values)
    return train, test, forecast_months


def compute_score_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    """计算选参指标：score / smape / mape / rmse / mae / bias。"""
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
    forecast_months: list[str],
    predictions: np.ndarray,
    test_df: pd.DataFrame,
    test_start: str = TEST_START,
    test_end: str = TEST_END,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """合并预测与测试真实值，并返回测试段真值 / 预测值。"""
    forecast_df = pd.DataFrame(
        {
            MONTH_COL: forecast_months,
            "predicted_gas_sales": predictions[: len(forecast_months)],
        }
    )
    result = forecast_df.merge(
        test_df[[MONTH_COL, TARGET_COL]],
        on=MONTH_COL,
        how="left",
    ).rename(columns={TARGET_COL: "actual_gas_sales"})

    is_test = (result[MONTH_COL] >= test_start) & (result[MONTH_COL] <= test_end)
    result["phase"] = np.where(is_test, "test", "bridge")
    result["error"] = result["predicted_gas_sales"] - result["actual_gas_sales"]
    result["abs_error"] = np.abs(result["error"])

    y_true = result.loc[is_test, "actual_gas_sales"].to_numpy(dtype=float)
    y_pred = result.loc[is_test, "predicted_gas_sales"].to_numpy(dtype=float)
    return result, y_true, y_pred


@dataclass
class TaskContext:
    """TimesFM 训练、搜索与输出所需上下文。"""

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


def _months_to_timestamp_index(months: list[str]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        pd.PeriodIndex(pd.Index(months).astype(str), freq="M").to_timestamp()
    )


def load_task_context(task: ForecastTask) -> TaskContext:
    path = str(task.excel_path) if task.excel_path is not None else find_processed_excel(task.province)
    raw_df = load_gas_series(path)

    as_of = task.as_of_month or AS_OF_MONTH
    test_start = task.test_start or TEST_START
    test_end = task.test_end or TEST_END
    train_df, test_df, forecast_months = split_asof_forecast_months(
        raw_df,
        as_of_month=as_of,
        test_start=test_start,
        test_end=test_end,
    )

    train_index = _months_to_timestamp_index(list(train_df[MONTH_COL].astype(str)))
    series = pd.Series(
        train_df[TARGET_COL].to_numpy(dtype=float),
        index=train_index,
        name="y",
    ).sort_index()
    series.index.name = "ds"

    return TaskContext(
        series=series,
        future_index=_months_to_timestamp_index(list(forecast_months)),
        test_df=test_df,
        forecast_months=list(forecast_months),
        test_start=test_start,
        test_end=test_end,
        metadata={
            "source": "gas_monthly",
            "excel_path": path,
            "as_of_month": as_of,
            "test_start": test_start,
            "test_end": test_end,
            "province": task.province,
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


def format_month_ds_for_display(ds: pd.Series) -> pd.Series:
    """月度序列常用月初 Timestamp；转为 ``YYYY-MM`` 字符串便于 CLI 预览。"""
    return pd.to_datetime(ds).dt.to_period("M").astype(str)


__all__ = [
    "AS_OF_MONTH",
    "TEST_START",
    "TEST_END",
    "TARGET_COL",
    "MONTH_COL",
    "PROCESSED_PROVINCE",
    "month_range",
    "find_processed_excel",
    "load_gas_series",
    "split_asof_test",
    "split_asof_forecast_months",
    "compute_score_metrics",
    "forecast_metrics",
    "build_test_table",
    "TaskContext",
    "load_task_context",
    "build_result_table",
    "format_month_ds_for_display",
]
