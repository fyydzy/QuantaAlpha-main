"""SARIMAX 预测后端。

与当前 Forecast Agent 流程保持一致：

1. 使用 ``as_of_month`` 指定的旬开始日及以前的数据训练；
2. 一次性预测 bridge + test 各旬；
3. 使用外部传入的固定 context 窗口，不再自动搜索最优 context_len。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantaalpha.forecast_agent.data import (
    DATE_COL,
    MONTH_COL,
    PERIODS_PER_MONTH,
    TARGET_COL,
    MIN_HISTORY_PERIODS,
    TaskContext,
    build_result_table,
    compute_score_metrics,
    filter_periods_in,
    filter_through_period,
    forecast_metrics,
    format_month_ds_for_display,
    feature_set_from_task,
    load_task_context,
    normalize_period_column,
)
from quantaalpha.forecast_agent.feature_engineering import (
    SARIMAX_EXOG_COLS,
    WEATHER_INPUT_COLS,
    add_weather_features,
)
from quantaalpha.forecast_agent.framework import (
    ForecastAgent,
    ForecastEvaluator,
    ForecastFeedback,
    ForecastStep,
    ForecastStrategy,
    ForecastSubjects,
    ForecastTask,
)


@dataclass(frozen=True)
class SarimaxHyperParams:
    """SARIMAX 搜索参数。

    当前自动搜索只枚举 ``context_len``；ARIMA 阶数由 ``pmdarima.auto_arima`` 自动选择。
    """

    context_len: int = 111

    def signature(self) -> tuple[int]:
        return (self.context_len,)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _failed_feedback(message: str) -> ForecastFeedback:
    return ForecastFeedback(
        score=float("inf"),
        smape=float("inf"),
        mape=float("inf"),
        rmse=float("inf"),
        mae=float("inf"),
        bias=float("inf"),
        aic=None,
        success=False,
        message=message,
    )


def _build_feedback_message(metrics: dict[str, float], params: SarimaxHyperParams) -> str:
    notes: list[str] = []
    if metrics["smape"] > 35:
        notes.append("测试集误差较高")
    if abs(metrics["bias"]) > max(metrics["mae"] * 0.2, 1.0):
        notes.append("预测存在系统性偏差")
    if params.context_len < MIN_HISTORY_PERIODS:
        notes.append("上下文长度偏短")
    return "；".join(notes) if notes else "SARIMAX 当前配置较稳定"


def _load_series_with_weather(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    required = {DATE_COL, TARGET_COL, *WEATHER_INPUT_COLS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"文件缺少 SARIMAX 所需列: {sorted(missing)}")

    out = df[list(required)].copy()
    out = normalize_period_column(out, DATE_COL)
    numeric_cols = [TARGET_COL, *WEATHER_INPUT_COLS]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=numeric_cols).sort_values(DATE_COL).reset_index(drop=True)
    return add_weather_features(out, inplace=False)


def _apply_context_window(df: pd.DataFrame, context_len: int) -> pd.DataFrame:
    if context_len <= 0 or len(df) <= context_len:
        return df.copy()
    return df.iloc[-context_len:].copy()


def _fit_auto_sarimax(y_train: np.ndarray, x_train: np.ndarray) -> Any:
    import pmdarima as pm  # type: ignore[import-not-found]  # 可选依赖，按需导入

    return pm.auto_arima(
        y=y_train,
        X=x_train,
        seasonal=True,
        m=12 * PERIODS_PER_MONTH,
        start_p=0,
        max_p=3,
        start_q=0,
        max_q=3,
        start_P=0,
        max_P=2,
        start_Q=0,
        max_Q=2,
        d=None,
        D=None,
        information_criterion="aic",
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )


def _prepare_train_and_future(
    ctx: TaskContext,
    params: SarimaxHyperParams,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = str(ctx.metadata["excel_path"])
    as_of = str(ctx.metadata["as_of_month"])
    df = _load_series_with_weather(path)

    train_df = filter_through_period(df, as_of)
    train_df = _apply_context_window(train_df, params.context_len)
    if len(train_df) < MIN_HISTORY_PERIODS:
        raise ValueError(
            f"context={params.context_len} 样本仅 {len(train_df)} 旬，"
            f"少于 {MIN_HISTORY_PERIODS} 旬（约 24 个月）"
        )

    future_df = filter_periods_in(df, ctx.forecast_months)
    if len(future_df) != len(ctx.forecast_months):
        missing = sorted(set(ctx.forecast_months) - set(future_df[MONTH_COL].astype(str)))
        raise ValueError(f"SARIMAX 外生特征旬不完整，缺少: {missing}")
    return train_df, future_df


def _sarimax_exog_cols(feature_set: list[str] | None) -> list[str]:
    if not feature_set:
        return list(SARIMAX_EXOG_COLS)
    return list(feature_set)


class SarimaxEvaluator(ForecastEvaluator):
    def evaluate(self, task: ForecastTask, subjects: ForecastSubjects) -> ForecastFeedback:
        ctx = load_task_context(task)
        params: SarimaxHyperParams = subjects.params  # type: ignore[assignment]
        try:
            train_df, future_df = _prepare_train_and_future(ctx, params)
            exog_cols = _sarimax_exog_cols(feature_set_from_task(task))
            y_train = train_df[TARGET_COL].to_numpy(dtype=float)
            x_train = train_df[exog_cols].to_numpy(dtype=float)
            x_future = future_df[exog_cols].to_numpy(dtype=float)

            model = _fit_auto_sarimax(y_train, x_train)
            yhat = np.asarray(model.predict(n_periods=ctx.horizon, X=x_future), dtype=float)
            _, y_true, y_pred = build_result_table(ctx, yhat)
            metrics = compute_score_metrics(y_true, y_pred)
            return ForecastFeedback(
                score=metrics["score"],
                smape=metrics["smape"],
                mape=metrics["mape"],
                rmse=metrics["rmse"],
                mae=metrics["mae"],
                bias=metrics["bias"],
                aic=float(model.aic()) if hasattr(model, "aic") else None,
                success=True,
                message=_build_feedback_message(metrics, params),
                metrics=metrics,
            )
        except Exception as exc:  # noqa: BLE001 - 搜索阶段统一转成失败反馈
            return _failed_feedback(f"SARIMAX 推理失败: {exc}")


class FixedSarimaxStrategy(ForecastStrategy):
    def __init__(self, context_len: int = 111) -> None:
        self.context_len = int(context_len)

    def seed_subjects(self, task: ForecastTask) -> list[ForecastSubjects]:
        params = SarimaxHyperParams(context_len=self.context_len)
        return [
            ForecastSubjects(
                params=params,
                metadata={"reason": f"固定 context_len={params.context_len}"},
            )
        ]


class AutoSarimaxForecastAgent(ForecastAgent):
    def __init__(
        self,
        context_len: int = 111,
        strategy: ForecastStrategy | None = None,
        evaluator: ForecastEvaluator | None = None,
        selection_metric: str = "mape",
    ) -> None:
        super().__init__(
            strategy=strategy or FixedSarimaxStrategy(context_len=context_len),
            evaluator=evaluator or SarimaxEvaluator(),
        )
        metric = (selection_metric or "mape").lower()
        if metric not in {"mape", "score"}:
            raise ValueError(f"Unsupported selection_metric: {selection_metric}")
        self.selection_metric = metric

    def _selection_key(self, feedback: ForecastFeedback) -> tuple[float, float]:
        primary = getattr(feedback, self.selection_metric, float("inf"))
        secondary_name = "score" if self.selection_metric == "mape" else "mape"
        secondary = getattr(feedback, secondary_name, float("inf"))
        return (float(primary), float(secondary))

    def run(self, task: ForecastTask) -> dict[str, Any]:
        self.trace = []
        best_step: ForecastStep | None = None

        for idx, subject in enumerate(self.strategy.seed_subjects(task), start=1):
            feedback = self.evaluator.evaluate(task, subject)
            step = ForecastStep(
                evolvable_subjects=subject,
                feedback=feedback,
                proposal_reason=subject.metadata.get("reason", f"固定候选 #{idx}"),
            )
            self.trace.append(step)

            if not feedback.success:
                continue
            if best_step is None or self._selection_key(feedback) < self._selection_key(best_step.feedback):  # type: ignore[arg-type]
                best_step = step

        if best_step is None or best_step.feedback is None:
            raise RuntimeError(self._format_failure())

        final_result = self.refit_and_forecast(task, best_step.evolvable_subjects.params)  # type: ignore[arg-type]
        output_paths = self.save_outputs(task, best_step, final_result)

        forecast_head = final_result["forecast_df"].head(10).copy()
        forecast_head["ds"] = format_month_ds_for_display(forecast_head["ds"])
        return {
            "backend": "sarimax",
            "best_params": best_step.evolvable_subjects.params.to_dict(),
            "best_feedback": asdict(best_step.feedback),
            "trace_size": len(self.trace),
            "forecast_head": forecast_head.to_dict(orient="records"),
            "outputs": output_paths,
        }

    def _format_failure(self) -> str:
        messages: list[str] = []
        for step in self.trace:
            if step.feedback is not None and not step.feedback.success:
                if step.feedback.message not in messages:
                    messages.append(step.feedback.message)
                if len(messages) >= 3:
                    break
        suffix = f" Sample errors: {' | '.join(messages)}" if messages else ""
        return f"SARIMAX forecast agent failed to find a valid configuration.{suffix}"

    def refit_and_forecast(self, task: ForecastTask, params: SarimaxHyperParams) -> dict[str, Any]:
        ctx = load_task_context(task)
        train_df, future_df = _prepare_train_and_future(ctx, params)
        fs = feature_set_from_task(task)
        exog_cols = _sarimax_exog_cols(fs)
        y_train = train_df[TARGET_COL].to_numpy(dtype=float)
        x_train = train_df[exog_cols].to_numpy(dtype=float)
        x_future = future_df[exog_cols].to_numpy(dtype=float)

        model = _fit_auto_sarimax(y_train, x_train)
        yhat = np.asarray(model.predict(n_periods=ctx.horizon, X=x_future), dtype=float)
        forecast_df = pd.DataFrame({"ds": ctx.future_index, "yhat": yhat})
        return {
            "series": ctx.series,
            "forecast_df": forecast_df,
            "ctx": ctx,
            "feature_set": fs,
            "feature_cols_used": exog_cols,
        }

    def save_outputs(
        self,
        task: ForecastTask,
        best_step: ForecastStep,
        final_result: dict[str, Any],
    ) -> dict[str, str]:
        output_dir = Path(task.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        trace_path = output_dir / "sarimax_search_trace.csv"
        forecast_path = output_dir / "sarimax_forecast.csv"
        summary_path = output_dir / "sarimax_best_summary.json"
        plot_path = output_dir / "sarimax_forecast_plot.png"

        pd.DataFrame(self._trace_rows()).to_csv(trace_path, index=False, encoding="utf-8-sig")
        final_result["forecast_df"].to_csv(forecast_path, index=False, encoding="utf-8-sig")

        ctx: TaskContext = final_result["ctx"]
        result_df, y_true, y_pred = build_result_table(
            ctx,
            final_result["forecast_df"]["yhat"].to_numpy(),
        )

        test_xlsx = output_dir / "sarimax_test.xlsx"
        try:
            result_df.to_excel(test_xlsx, index=False, sheet_name="test")
            result_path_key = "test_xlsx"
            result_path = test_xlsx
        except Exception:
            test_csv = output_dir / "sarimax_test.csv"
            result_df.to_csv(test_csv, index=False, encoding="utf-8-sig")
            result_path_key = "test_csv"
            result_path = test_csv

        summary: dict[str, Any] = {
            "backend": "sarimax",
            "task": {
                **ctx.metadata,
                "horizon": ctx.horizon,
                "selection_metric": self.selection_metric,
                "exog_cols": final_result.get("feature_cols_used", list(SARIMAX_EXOG_COLS)),
            },
            "best_params": best_step.evolvable_subjects.params.to_dict(),
            "best_feedback": asdict(best_step.feedback) if best_step.feedback else {},
            "trace_size": len(self.trace),
            "test_metrics": forecast_metrics(y_true, y_pred),
            "feature_set": final_result.get("feature_set", list(getattr(task, "feature_set", []) or [])),
            "feature_cols_used": final_result.get("feature_cols_used", list(SARIMAX_EXOG_COLS)),
            result_path_key: str(result_path),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        self._save_plot(
            series=final_result["series"],
            forecast_df=final_result["forecast_df"],
            plot_path=plot_path,
        )

        return {
            "trace_csv": str(trace_path),
            "forecast_csv": str(forecast_path),
            "summary_json": str(summary_path),
            "plot_png": str(plot_path),
            result_path_key: str(result_path),
        }

    def _trace_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx, step in enumerate(self.trace, start=1):
            feedback = step.feedback
            rows.append(
                {
                    "trial": idx,
                    "proposal_reason": step.proposal_reason,
                    **step.evolvable_subjects.params.to_dict(),
                    "success": feedback.success if feedback else False,
                    "score": feedback.score if feedback else None,
                    "smape": feedback.smape if feedback else None,
                    "mape": feedback.mape if feedback else None,
                    "rmse": feedback.rmse if feedback else None,
                    "mae": feedback.mae if feedback else None,
                    "bias": feedback.bias if feedback else None,
                    "aic": feedback.aic if feedback else None,
                    "message": feedback.message if feedback else "",
                }
            )
        return rows

    @staticmethod
    def _save_plot(series: pd.Series, forecast_df: pd.DataFrame, plot_path: Path) -> None:
        plt.figure(figsize=(14, 6))
        plt.plot(series.index, series.values, label="history", color="tab:blue")
        plt.plot(pd.to_datetime(forecast_df["ds"]), forecast_df["yhat"], label="forecast", color="tab:orange")
        plt.title("Forecast Agent - Auto SARIMAX")
        plt.xlabel("Date")
        plt.ylabel("y")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()


__all__ = [
    "SarimaxHyperParams",
    "SarimaxEvaluator",
    "FixedSarimaxStrategy",
    "AutoSarimaxForecastAgent",
]