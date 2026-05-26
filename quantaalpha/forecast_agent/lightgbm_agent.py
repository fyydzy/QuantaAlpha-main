"""LightGBM 预测后端（保留随机网格搜索，仅固定 context_len）。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantaalpha.forecast_agent.data import (
    DEFAULT_VAL_PERIODS,
    MONTH_COL,
    PERIODS_PER_MONTH,
    TARGET_COL,
    MIN_HISTORY_PERIODS,
    TaskContext,
    build_result_table,
    compute_score_metrics,
    forecast_metrics,
    format_month_ds_for_display,
    filter_periods_in,
    filter_through_period,
    load_tabular_feature_frame,
    load_task_context,
    tabular_feature_columns,
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

EARLY_STOPPING_ROUNDS = 50
RANDOM_SEARCH_N_TRIALS = 200
RANDOM_SEARCH_SEED = 42


@dataclass(frozen=True)
class LightgbmHyperParams:
    context_len: int = 111
    max_depth: int = 4
    learning_rate: float = 0.03
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    min_child_samples: int = 10
    subsample: float = 0.8
    colsample_bytree: float = 0.8

    def signature(self) -> tuple[Any, ...]:
        return (
            self.context_len,
            self.max_depth,
            self.learning_rate,
            self.reg_alpha,
            self.reg_lambda,
            self.min_child_samples,
            self.subsample,
            self.colsample_bytree,
        )

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


def _build_feedback_message(metrics: dict[str, float], params: LightgbmHyperParams) -> str:
    notes: list[str] = []
    if metrics["smape"] > 35:
        notes.append("测试集误差较高")
    if abs(metrics["bias"]) > max(metrics["mae"] * 0.2, 1.0):
        notes.append("预测存在系统性偏差")
    if params.context_len < MIN_HISTORY_PERIODS:
        notes.append("上下文长度偏短")
    return "；".join(notes) if notes else "LightGBM 当前配置较稳定"


def _load_feature_frame(path: str) -> pd.DataFrame:
    return load_tabular_feature_frame(path)


def _apply_context_window(df: pd.DataFrame, context_len: int) -> pd.DataFrame:
    if context_len <= 0 or len(df) <= context_len:
        return df.copy()
    return df.iloc[-context_len:].copy()


def _prepare_train_and_future(
    ctx: TaskContext,
    params: LightgbmHyperParams,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    path = str(ctx.metadata["excel_path"])
    as_of = str(ctx.metadata["as_of_month"])
    feature_df = _load_feature_frame(path)

    train_df_full = filter_through_period(feature_df, as_of)
    train_df = _apply_context_window(train_df_full, params.context_len)
    if len(train_df) < MIN_HISTORY_PERIODS:
        raise ValueError(
            f"context={params.context_len} 样本仅 {len(train_df)} 旬，"
            f"少于 {MIN_HISTORY_PERIODS} 旬（约 24 个月）"
        )

    future_df = filter_periods_in(feature_df, ctx.forecast_months)
    if len(future_df) != len(ctx.forecast_months):
        missing = sorted(set(ctx.forecast_months) - set(future_df[MONTH_COL].astype(str)))
        raise ValueError(f"LightGBM 特征旬不完整，缺少: {missing}")

    feature_cols = tabular_feature_columns(feature_df)
    data_debug = {
        "feature_rows_total": int(len(feature_df)),
        "train_rows_before_context": int(len(train_df_full)),
        "train_rows_after_context": int(len(train_df)),
        "context_len": int(params.context_len),
        "context_applied": bool(len(train_df_full) != len(train_df)),
    }
    return train_df, future_df, feature_cols, data_debug


def _split_train_val(
    X_full: pd.DataFrame,
    y_full: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    n = len(X_full)
    val_size = min(DEFAULT_VAL_PERIODS, max(PERIODS_PER_MONTH * 2, n // 5))
    split_idx = n - val_size
    if split_idx <= 0:
        split_idx = max(1, n - 1)
    return X_full[:split_idx], y_full[:split_idx], X_full[split_idx:], y_full[split_idx:]


def _fit_with_early_stopping(
    X_train_full: pd.DataFrame,
    y_train_full: np.ndarray,
    params: LightgbmHyperParams,
) -> tuple[Any, int, dict[str, float], dict[str, int]]:
    import lightgbm as lgb  # type: ignore[import-not-found]
    from lightgbm import LGBMRegressor  # type: ignore[import-not-found]

    X_train, y_train, X_val, y_val = _split_train_val(X_train_full, y_train_full)
    model = LGBMRegressor(
        n_estimators=800,
        max_depth=int(params.max_depth),
        learning_rate=float(params.learning_rate),
        reg_alpha=float(params.reg_alpha),
        reg_lambda=float(params.reg_lambda),
        min_child_samples=int(params.min_child_samples),
        subsample=float(params.subsample),
        colsample_bytree=float(params.colsample_bytree),
        objective="regression",
        random_state=42,
        verbosity=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    val_pred = np.asarray(model.predict(X_val), dtype=float)
    val_metrics = compute_score_metrics(y_val.astype(float), val_pred)
    best_iteration = int(getattr(model, "best_iteration_", 799))
    if best_iteration <= 0:
        best_iteration = 799
    split_debug = {
        "fit_train_rows": int(len(X_train)),
        "fit_val_rows": int(len(X_val)),
    }
    return model, best_iteration, val_metrics, split_debug


def _refit_full_train(
    X_train_full: pd.DataFrame,
    y_train_full: np.ndarray,
    params: LightgbmHyperParams,
    best_iteration: int,
) -> Any:
    from lightgbm import LGBMRegressor  # type: ignore[import-not-found]

    model = LGBMRegressor(
        n_estimators=max(int(best_iteration) + 1, 50),
        max_depth=int(params.max_depth),
        learning_rate=float(params.learning_rate),
        reg_alpha=float(params.reg_alpha),
        reg_lambda=float(params.reg_lambda),
        min_child_samples=int(params.min_child_samples),
        subsample=float(params.subsample),
        colsample_bytree=float(params.colsample_bytree),
        objective="regression",
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train_full, y_train_full)
    return model


class LightgbmEvaluator(ForecastEvaluator):
    def evaluate(self, task: ForecastTask, subjects: ForecastSubjects) -> ForecastFeedback:
        ctx = load_task_context(task)
        params: LightgbmHyperParams = subjects.params  # type: ignore[assignment]
        try:
            train_df, _, feature_cols, data_debug = _prepare_train_and_future(ctx, params)
            X_train = train_df[feature_cols].astype(float)
            y_train = train_df[TARGET_COL].to_numpy(dtype=float)

            _, best_iteration, val_metrics, split_debug = _fit_with_early_stopping(X_train, y_train, params)
            return ForecastFeedback(
                score=val_metrics["score"],
                smape=val_metrics["smape"],
                mape=val_metrics["mape"],
                rmse=val_metrics["rmse"],
                mae=val_metrics["mae"],
                bias=val_metrics["bias"],
                aic=None,
                success=True,
                message=_build_feedback_message(val_metrics, params),
                metrics={
                    **val_metrics,
                    "best_iteration": int(best_iteration),
                    **data_debug,
                    **split_debug,
                },
            )
        except Exception as exc:  # noqa: BLE001
            return _failed_feedback(f"LightGBM 推理失败: {exc}")


class RandomGridLightgbmStrategy(ForecastStrategy):
    def __init__(
        self,
        context_len: int = 111,
        n_trials: int = RANDOM_SEARCH_N_TRIALS,
        random_seed: int = RANDOM_SEARCH_SEED,
    ) -> None:
        self.context_len = int(context_len)
        self.n_trials = int(n_trials)
        self.random_seed = int(random_seed)

    def seed_subjects(self, task: ForecastTask) -> list[ForecastSubjects]:
        grid_max_depth = (3, 4, 5)
        grid_learning_rate = (0.02, 0.03, 0.05)
        grid_reg_alpha = (0.0, 0.1, 0.2)
        grid_reg_lambda = (0.5, 1.0, 2.0)
        grid_min_child_samples = (5, 10, 20)
        grid_subsample = (0.7, 0.8)
        grid_colsample_bytree = (0.7, 0.8)

        candidates: list[LightgbmHyperParams] = []
        for max_depth in grid_max_depth:
            for learning_rate in grid_learning_rate:
                for reg_alpha in grid_reg_alpha:
                    for reg_lambda in grid_reg_lambda:
                        for min_child_samples in grid_min_child_samples:
                            for subsample in grid_subsample:
                                for colsample_bytree in grid_colsample_bytree:
                                    candidates.append(
                                        LightgbmHyperParams(
                                            context_len=self.context_len,
                                            max_depth=int(max_depth),
                                            learning_rate=float(learning_rate),
                                            reg_alpha=float(reg_alpha),
                                            reg_lambda=float(reg_lambda),
                                            min_child_samples=int(min_child_samples),
                                            subsample=float(subsample),
                                            colsample_bytree=float(colsample_bytree),
                                        )
                                    )

        rng = np.random.default_rng(self.random_seed)
        total = len(candidates)
        n_trials = min(max(1, self.n_trials), total)
        chosen_idx = rng.choice(total, size=n_trials, replace=False)
        sampled = [candidates[int(i)] for i in chosen_idx]

        return [
            ForecastSubjects(
                params=params,
                metadata={
                    "reason": (
                        "随机网格候选 "
                        f"max_depth={params.max_depth}, lr={params.learning_rate}, "
                        f"alpha={params.reg_alpha}, lambda={params.reg_lambda}, "
                        f"min_child_samples={params.min_child_samples}, "
                        f"subsample={params.subsample}, colsample_bytree={params.colsample_bytree}"
                    )
                },
            )
            for params in sampled
        ]


class AutoLightgbmForecastAgent(ForecastAgent):
    def __init__(
        self,
        context_len: int = 111,
        strategy: ForecastStrategy | None = None,
        evaluator: ForecastEvaluator | None = None,
        selection_metric: str = "mape",
    ) -> None:
        super().__init__(
            strategy=strategy or RandomGridLightgbmStrategy(context_len=context_len),
            evaluator=evaluator or LightgbmEvaluator(),
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
            "backend": "lightgbm",
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
        return f"LightGBM forecast agent failed to find a valid configuration.{suffix}"

    def refit_and_forecast(self, task: ForecastTask, params: LightgbmHyperParams) -> dict[str, Any]:
        ctx = load_task_context(task)
        train_df, future_df, feature_cols, data_debug = _prepare_train_and_future(ctx, params)
        X_train = train_df[feature_cols].astype(float)
        y_train = train_df[TARGET_COL].to_numpy(dtype=float)
        X_future = future_df[feature_cols].astype(float)

        _, best_iteration, _, split_debug = _fit_with_early_stopping(X_train, y_train, params)
        model = _refit_full_train(X_train, y_train, params, best_iteration=best_iteration)
        yhat = np.asarray(model.predict(X_future), dtype=float)
        forecast_df = pd.DataFrame({"ds": ctx.future_index, "yhat": yhat})

        importance: dict[str, float] = {}
        fi = getattr(model, "feature_importances_", None)
        if fi is not None:
            arr = np.asarray(fi, dtype=float)
            if len(arr) == len(feature_cols):
                importance = pd.Series(arr, index=feature_cols).sort_values(ascending=False).to_dict()

        return {
            "series": ctx.series,
            "forecast_df": forecast_df,
            "ctx": ctx,
            "feature_importance": importance,
            "best_iteration": int(best_iteration),
            "data_debug": {
                **data_debug,
                **split_debug,
            },
        }

    def save_outputs(
        self,
        task: ForecastTask,
        best_step: ForecastStep,
        final_result: dict[str, Any],
    ) -> dict[str, str]:
        output_dir = Path(task.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        trace_path = output_dir / "lightgbm_search_trace.csv"
        forecast_path = output_dir / "lightgbm_forecast.csv"
        summary_path = output_dir / "lightgbm_best_summary.json"
        plot_path = output_dir / "lightgbm_forecast_plot.png"

        pd.DataFrame(self._trace_rows()).to_csv(trace_path, index=False, encoding="utf-8-sig")
        final_result["forecast_df"].to_csv(forecast_path, index=False, encoding="utf-8-sig")

        ctx: TaskContext = final_result["ctx"]
        result_df, y_true, y_pred = build_result_table(
            ctx,
            final_result["forecast_df"]["yhat"].to_numpy(),
        )

        test_xlsx = output_dir / "lightgbm_test.xlsx"
        try:
            result_df.to_excel(test_xlsx, index=False, sheet_name="test")
            result_path_key = "test_xlsx"
            result_path = test_xlsx
        except Exception:
            test_csv = output_dir / "lightgbm_test.csv"
            result_df.to_csv(test_csv, index=False, encoding="utf-8-sig")
            result_path_key = "test_csv"
            result_path = test_csv

        summary: dict[str, Any] = {
            "backend": "lightgbm",
            "task": {
                **ctx.metadata,
                "horizon": ctx.horizon,
                "selection_metric": self.selection_metric,
            },
            "best_params": best_step.evolvable_subjects.params.to_dict(),
            "best_feedback": asdict(best_step.feedback) if best_step.feedback else {},
            "trace_size": len(self.trace),
            "test_metrics": forecast_metrics(y_true, y_pred),
            "best_iteration": final_result.get("best_iteration"),
            "data_debug": final_result.get("data_debug", {}),
            "feature_importance": final_result.get("feature_importance", {}),
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
                    "best_iteration": (feedback.metrics.get("best_iteration") if feedback and feedback.metrics else None),
                    "train_rows_before_context": (
                        feedback.metrics.get("train_rows_before_context") if feedback and feedback.metrics else None
                    ),
                    "train_rows_after_context": (
                        feedback.metrics.get("train_rows_after_context") if feedback and feedback.metrics else None
                    ),
                    "fit_train_rows": (feedback.metrics.get("fit_train_rows") if feedback and feedback.metrics else None),
                    "fit_val_rows": (feedback.metrics.get("fit_val_rows") if feedback and feedback.metrics else None),
                    "message": feedback.message if feedback else "",
                }
            )
        return rows

    @staticmethod
    def _save_plot(series: pd.Series, forecast_df: pd.DataFrame, plot_path: Path) -> None:
        plt.figure(figsize=(14, 6))
        plt.plot(series.index, series.values, label="history", color="tab:blue")
        plt.plot(pd.to_datetime(forecast_df["ds"]), forecast_df["yhat"], label="forecast", color="tab:orange")
        plt.title("Forecast Agent - Auto LightGBM")
        plt.xlabel("Date")
        plt.ylabel("y")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()


__all__ = [
    "LightgbmHyperParams",
    "LightgbmEvaluator",
    "RandomGridLightgbmStrategy",
    "AutoLightgbmForecastAgent",
]
