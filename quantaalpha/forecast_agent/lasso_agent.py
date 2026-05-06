"""Lasso 预测后端。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from quantaalpha.forecast_agent.data import (
    MONTH_COL,
    TARGET_COL,
    TaskContext,
    build_result_table,
    compute_score_metrics,
    forecast_metrics,
    format_month_ds_for_display,
    load_task_context,
)
from quantaalpha.forecast_agent.feature_engineering import WEATHER_INPUT_COLS, build_features_pipeline
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
class LassoHyperParams:
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


def _build_feedback_message(metrics: dict[str, float], params: LassoHyperParams) -> str:
    notes: list[str] = []
    if metrics["smape"] > 35:
        notes.append("测试集误差较高")
    if abs(metrics["bias"]) > max(metrics["mae"] * 0.2, 1.0):
        notes.append("预测存在系统性偏差")
    if params.context_len < 24:
        notes.append("上下文长度偏短")
    return "；".join(notes) if notes else "Lasso 当前配置较稳定"


def _load_feature_frame(path: str) -> pd.DataFrame:
    raw = pd.read_excel(path)
    required = {MONTH_COL, TARGET_COL, *WEATHER_INPUT_COLS}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"文件缺少 Lasso 所需列: {sorted(missing)}")

    out = raw[list(required)].copy()
    out[MONTH_COL] = out[MONTH_COL].astype(str).str.slice(0, 7)
    numeric_cols = [TARGET_COL, *WEATHER_INPUT_COLS]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=numeric_cols).sort_values(MONTH_COL).reset_index(drop=True)
    return build_features_pipeline(out, target_col=TARGET_COL, month_col=MONTH_COL, dropna=True)


def _apply_context_window(df: pd.DataFrame, context_len: int) -> pd.DataFrame:
    if context_len <= 0 or len(df) <= context_len:
        return df.copy()
    return df.iloc[-context_len:].copy()


def _prepare_train_and_future(
    ctx: TaskContext,
    params: LassoHyperParams,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    path = str(ctx.metadata["excel_path"])
    as_of = str(ctx.metadata["as_of_month"])
    feature_df = _load_feature_frame(path)

    train_df = feature_df[feature_df[MONTH_COL] <= as_of].sort_values(MONTH_COL)
    train_df = _apply_context_window(train_df, params.context_len)
    if len(train_df) < 24:
        raise ValueError(f"context={params.context_len} 样本仅 {len(train_df)} 月，少于 24 月")

    future_df = feature_df[feature_df[MONTH_COL].isin(ctx.forecast_months)].sort_values(MONTH_COL)
    if len(future_df) != len(ctx.forecast_months):
        missing = sorted(set(ctx.forecast_months) - set(future_df[MONTH_COL].astype(str)))
        raise ValueError(f"Lasso 特征月份不完整，缺少: {missing}")

    feature_cols = [c for c in feature_df.columns if c not in {MONTH_COL, TARGET_COL}]
    return train_df, future_df, feature_cols


def _fit_lasso_cv(X_train: np.ndarray, y_train: np.ndarray) -> tuple[LassoCV, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    cv = max(2, min(5, len(X_scaled)))
    model = LassoCV(cv=cv, random_state=42, max_iter=10000, n_alphas=200).fit(X_scaled, y_train)
    return model, scaler


class LassoEvaluator(ForecastEvaluator):
    def evaluate(self, task: ForecastTask, subjects: ForecastSubjects) -> ForecastFeedback:
        ctx = load_task_context(task)
        params: LassoHyperParams = subjects.params  # type: ignore[assignment]
        try:
            train_df, future_df, feature_cols = _prepare_train_and_future(ctx, params)
            X_train = train_df[feature_cols].to_numpy(dtype=float)
            y_train = train_df[TARGET_COL].to_numpy(dtype=float)
            X_future = future_df[feature_cols].to_numpy(dtype=float)

            model, scaler = _fit_lasso_cv(X_train, y_train)
            yhat = np.asarray(model.predict(scaler.transform(X_future)), dtype=float)
            _, y_true, y_pred = build_result_table(ctx, yhat)
            metrics = compute_score_metrics(y_true, y_pred)
            return ForecastFeedback(
                score=metrics["score"],
                smape=metrics["smape"],
                mape=metrics["mape"],
                rmse=metrics["rmse"],
                mae=metrics["mae"],
                bias=metrics["bias"],
                aic=None,
                success=True,
                message=_build_feedback_message(metrics, params),
                metrics={**metrics, "alpha": float(model.alpha_)},
            )
        except Exception as exc:  # noqa: BLE001
            return _failed_feedback(f"Lasso 推理失败: {exc}")


class FixedLassoStrategy(ForecastStrategy):
    def __init__(self, context_len: int = 111) -> None:
        self.context_len = int(context_len)

    def seed_subjects(self, task: ForecastTask) -> list[ForecastSubjects]:
        params = LassoHyperParams(context_len=self.context_len)
        return [
            ForecastSubjects(
                params=params,
                metadata={"reason": f"固定 context_len={params.context_len}"},
            )
        ]


class AutoLassoForecastAgent(ForecastAgent):
    def __init__(
        self,
        context_len: int = 111,
        strategy: ForecastStrategy | None = None,
        evaluator: ForecastEvaluator | None = None,
        selection_metric: str = "mape",
    ) -> None:
        super().__init__(
            strategy=strategy or FixedLassoStrategy(context_len=context_len),
            evaluator=evaluator or LassoEvaluator(),
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
            "backend": "lasso",
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
        return f"Lasso forecast agent failed to find a valid configuration.{suffix}"

    def refit_and_forecast(self, task: ForecastTask, params: LassoHyperParams) -> dict[str, Any]:
        ctx = load_task_context(task)
        train_df, future_df, feature_cols = _prepare_train_and_future(ctx, params)
        X_train = train_df[feature_cols].to_numpy(dtype=float)
        y_train = train_df[TARGET_COL].to_numpy(dtype=float)
        X_future = future_df[feature_cols].to_numpy(dtype=float)

        model, scaler = _fit_lasso_cv(X_train, y_train)
        yhat = np.asarray(model.predict(scaler.transform(X_future)), dtype=float)
        coef = pd.Series(model.coef_, index=feature_cols)
        selected = coef[coef.abs() > 1e-8].sort_values(key=np.abs, ascending=False)
        forecast_df = pd.DataFrame({"ds": ctx.future_index, "yhat": yhat})
        return {
            "series": ctx.series,
            "forecast_df": forecast_df,
            "ctx": ctx,
            "alpha": float(model.alpha_),
            "selected_features": selected.to_dict(),
        }

    def save_outputs(
        self,
        task: ForecastTask,
        best_step: ForecastStep,
        final_result: dict[str, Any],
    ) -> dict[str, str]:
        output_dir = Path(task.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        trace_path = output_dir / "lasso_search_trace.csv"
        forecast_path = output_dir / "lasso_forecast.csv"
        summary_path = output_dir / "lasso_best_summary.json"
        plot_path = output_dir / "lasso_forecast_plot.png"

        pd.DataFrame(self._trace_rows()).to_csv(trace_path, index=False, encoding="utf-8-sig")
        final_result["forecast_df"].to_csv(forecast_path, index=False, encoding="utf-8-sig")

        ctx: TaskContext = final_result["ctx"]
        result_df, y_true, y_pred = build_result_table(
            ctx,
            final_result["forecast_df"]["yhat"].to_numpy(),
        )
        test_xlsx = output_dir / "lasso_test.xlsx"
        try:
            result_df.to_excel(test_xlsx, index=False, sheet_name="test")
            result_path_key = "test_xlsx"
            result_path = test_xlsx
        except Exception:
            test_csv = output_dir / "lasso_test.csv"
            result_df.to_csv(test_csv, index=False, encoding="utf-8-sig")
            result_path_key = "test_csv"
            result_path = test_csv

        summary: dict[str, Any] = {
            "backend": "lasso",
            "task": {
                **ctx.metadata,
                "horizon": ctx.horizon,
                "selection_metric": self.selection_metric,
            },
            "best_params": best_step.evolvable_subjects.params.to_dict(),
            "best_feedback": asdict(best_step.feedback) if best_step.feedback else {},
            "trace_size": len(self.trace),
            "test_metrics": forecast_metrics(y_true, y_pred),
            "alpha": final_result.get("alpha"),
            "selected_features": final_result.get("selected_features", {}),
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
            row = {
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
                "alpha": (feedback.metrics.get("alpha") if feedback and feedback.metrics else None),
                "message": feedback.message if feedback else "",
            }
            rows.append(row)
        return rows

    @staticmethod
    def _save_plot(series: pd.Series, forecast_df: pd.DataFrame, plot_path: Path) -> None:
        plt.figure(figsize=(14, 6))
        plt.plot(series.index, series.values, label="history", color="tab:blue")
        plt.plot(pd.to_datetime(forecast_df["ds"]), forecast_df["yhat"], label="forecast", color="tab:orange")
        plt.title("Forecast Agent - Auto Lasso")
        plt.xlabel("Date")
        plt.ylabel("y")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()


__all__ = [
    "LassoHyperParams",
    "LassoEvaluator",
    "FixedLassoStrategy",
    "AutoLassoForecastAgent",
]