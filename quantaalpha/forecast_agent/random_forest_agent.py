"""随机森林预测后端（与 Lasso/ElasticNet 同一套 Forecast Agent 流程；保留原脚本的随机网格与 RF 训练逻辑）。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

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

# --- 与原 random_forest 脚本一致的随机网格 ---
GRID_N_ESTIMATORS = [300, 500, 800]
GRID_MAX_DEPTH = [3, 4, 5, None]
GRID_MIN_SAMPLES_LEAF = [1, 2, 4]
GRID_MIN_SAMPLES_SPLIT = [2, 5, 10]
GRID_MAX_FEATURES = ["sqrt", "log2", 0.6]
GRID_MAX_SAMPLES = [0.7, 0.8, 1.0]
RANDOM_SEARCH_N_TRIALS = 200
RANDOM_SEARCH_SEED = 42

BASE_RF_PARAMS: dict[str, Any] = {
    "random_state": 42,
    "n_jobs": -1,
    "bootstrap": True,
}


@dataclass(frozen=True)
class RandomForestHyperParams:
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


def _build_feedback_message(metrics: dict[str, float], params: RandomForestHyperParams) -> str:
    notes: list[str] = []
    if metrics["smape"] > 35:
        notes.append("测试集误差较高")
    if abs(metrics["bias"]) > max(metrics["mae"] * 0.2, 1.0):
        notes.append("预测存在系统性偏差")
    if params.context_len < 24:
        notes.append("上下文长度偏短")
    return "；".join(notes) if notes else "随机森林当前配置较稳定"


def _load_feature_frame(path: str) -> pd.DataFrame:
    raw = pd.read_excel(path)
    required = {MONTH_COL, TARGET_COL, *WEATHER_INPUT_COLS}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"文件缺少随机森林所需列: {sorted(missing)}")

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
    params: RandomForestHyperParams,
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
        raise ValueError(f"随机森林特征月份不完整，缺少: {missing}")

    feature_cols = [c for c in feature_df.columns if c not in {MONTH_COL, TARGET_COL}]
    return train_df, future_df, feature_cols


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if np.any(mask):
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    return float("nan")


def _grid_search_rf(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    best_params: dict[str, Any] | None = None
    best_mape = float("inf")

    all_combos = list(
        product(
            GRID_N_ESTIMATORS,
            GRID_MAX_DEPTH,
            GRID_MIN_SAMPLES_LEAF,
            GRID_MIN_SAMPLES_SPLIT,
            GRID_MAX_FEATURES,
            GRID_MAX_SAMPLES,
        )
    )
    total_combos = len(all_combos)
    n_trials = min(RANDOM_SEARCH_N_TRIALS, total_combos)
    rng = np.random.default_rng(RANDOM_SEARCH_SEED)
    chosen_idx = rng.choice(total_combos, size=n_trials, replace=False)
    sampled_combos = [all_combos[i] for i in chosen_idx]

    for (
        n_estimators,
        max_depth,
        min_samples_leaf,
        min_samples_split,
        max_features,
        max_samples,
    ) in sampled_combos:
        if min_samples_split <= min_samples_leaf:
            continue
        params = dict(BASE_RF_PARAMS)
        params.update(
            {
                "n_estimators": int(n_estimators),
                "max_depth": max_depth,
                "min_samples_leaf": int(min_samples_leaf),
                "min_samples_split": int(min_samples_split),
                "max_features": max_features,
                "max_samples": float(max_samples),
            }
        )
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val).astype(float)
        val_mape = _mape(y_val.astype(float), val_pred)
        rows.append(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth if max_depth is not None else "None",
                "min_samples_leaf": min_samples_leaf,
                "min_samples_split": min_samples_split,
                "max_features": max_features,
                "max_samples": max_samples,
                "val_mape_pct": val_mape,
            }
        )
        if val_mape < best_mape:
            best_mape = val_mape
            best_params = params

    if best_params is None:
        raise ValueError("网格搜索失败，未找到有效参数（可能需放宽 min_samples_split / leaf 约束）。")
    grid_df = pd.DataFrame(rows).sort_values("val_mape_pct").reset_index(drop=True)
    return best_params, grid_df


def _split_train_val(train_df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    X_train_full = train_df[feature_cols]
    y_train_full = train_df[TARGET_COL].to_numpy(dtype=float)
    n = len(X_train_full)
    val_size = min(12, max(6, n // 5))
    split_idx = n - val_size
    if split_idx <= 0:
        split_idx = max(1, n - 1)
    X_train = X_train_full.iloc[:split_idx]
    y_train = y_train_full[:split_idx]
    X_val = X_train_full.iloc[split_idx:]
    y_val = y_train_full[split_idx:]
    return X_train, y_train, X_val, y_val


def _fit_rf_full_train(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    best_params: dict[str, Any],
) -> RandomForestRegressor:
    X_full = train_df[feature_cols]
    y_full = train_df[TARGET_COL].to_numpy(dtype=float)
    model = RandomForestRegressor(**best_params)
    model.fit(X_full, y_full)
    return model


def _fit_rf_with_search(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[RandomForestRegressor, dict[str, Any], pd.DataFrame]:
    X_train, y_train, X_val, y_val = _split_train_val(train_df, feature_cols)
    best_params, grid_df = _grid_search_rf(X_train, y_train, X_val, y_val)
    model = _fit_rf_full_train(train_df, feature_cols, best_params)
    return model, best_params, grid_df


class RandomForestEvaluator(ForecastEvaluator):
    """evaluate 结束后将 (best_params, grid_df) 缓存在实例上，供同一 agent 的 refit 复用，避免重复随机搜索。"""

    def __init__(self) -> None:
        self._last_fit_bundle: tuple[dict[str, Any], pd.DataFrame] | None = None

    def evaluate(self, task: ForecastTask, subjects: ForecastSubjects) -> ForecastFeedback:
        ctx = load_task_context(task)
        params: RandomForestHyperParams = subjects.params  # type: ignore[assignment]
        self._last_fit_bundle = None
        try:
            train_df, future_df, feature_cols = _prepare_train_and_future(ctx, params)
            model, best_params, grid_df = _fit_rf_with_search(train_df, feature_cols)
            self._last_fit_bundle = (best_params, grid_df)
            yhat = np.asarray(model.predict(future_df[feature_cols]), dtype=float)
            _, y_true, y_pred = build_result_table(ctx, yhat)
            metrics = compute_score_metrics(y_true, y_pred)
            rf_metrics = {
                "n_estimators": int(best_params["n_estimators"]),
                "max_depth": best_params["max_depth"],
                "min_samples_leaf": int(best_params["min_samples_leaf"]),
                "min_samples_split": int(best_params["min_samples_split"]),
                "max_features": best_params["max_features"],
                "max_samples": float(best_params["max_samples"]),
            }
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
                metrics={**metrics, **rf_metrics},
            )
        except Exception as exc:  # noqa: BLE001
            self._last_fit_bundle = None
            return _failed_feedback(f"随机森林推理失败: {exc}")


class FixedRandomForestStrategy(ForecastStrategy):
    def __init__(self, context_len: int = 111) -> None:
        self.context_len = int(context_len)

    def seed_subjects(self, task: ForecastTask) -> list[ForecastSubjects]:
        params = RandomForestHyperParams(context_len=self.context_len)
        return [
            ForecastSubjects(
                params=params,
                metadata={"reason": f"固定 context_len={params.context_len}"},
            )
        ]


class AutoRandomForestForecastAgent(ForecastAgent):
    def __init__(
        self,
        context_len: int = 111,
        strategy: ForecastStrategy | None = None,
        evaluator: ForecastEvaluator | None = None,
        selection_metric: str = "mape",
    ) -> None:
        super().__init__(
            strategy=strategy or FixedRandomForestStrategy(context_len=context_len),
            evaluator=evaluator or RandomForestEvaluator(),
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
            "backend": "random_forest",
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
        return f"Random forest forecast agent failed to find a valid configuration.{suffix}"

    def refit_and_forecast(self, task: ForecastTask, params: RandomForestHyperParams) -> dict[str, Any]:
        ctx = load_task_context(task)
        train_df, future_df, feature_cols = _prepare_train_and_future(ctx, params)
        ev = self.evaluator
        bundle = getattr(ev, "_last_fit_bundle", None) if isinstance(ev, RandomForestEvaluator) else None
        if bundle is not None:
            best_params, grid_df = bundle
            model = _fit_rf_full_train(train_df, feature_cols, best_params)
        else:
            model, best_params, grid_df = _fit_rf_with_search(train_df, feature_cols)
        yhat = np.asarray(model.predict(future_df[feature_cols]), dtype=float)
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        top_importance = importances.sort_values(ascending=False).head(30).to_dict()
        forecast_df = pd.DataFrame({"ds": ctx.future_index, "yhat": yhat})
        return {
            "series": ctx.series,
            "forecast_df": forecast_df,
            "ctx": ctx,
            "best_rf_params": best_params,
            "grid_search_df": grid_df,
            "top_feature_importance": top_importance,
        }

    def save_outputs(
        self,
        task: ForecastTask,
        best_step: ForecastStep,
        final_result: dict[str, Any],
    ) -> dict[str, str]:
        output_dir = Path(task.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        trace_path = output_dir / "random_forest_search_trace.csv"
        forecast_path = output_dir / "random_forest_forecast.csv"
        summary_path = output_dir / "random_forest_best_summary.json"
        plot_path = output_dir / "random_forest_forecast_plot.png"
        grid_path = output_dir / "random_forest_grid_search_results.csv"

        pd.DataFrame(self._trace_rows()).to_csv(trace_path, index=False, encoding="utf-8-sig")
        final_result["forecast_df"].to_csv(forecast_path, index=False, encoding="utf-8-sig")
        final_result["grid_search_df"].to_csv(grid_path, index=False, encoding="utf-8-sig")

        ctx: TaskContext = final_result["ctx"]
        result_df, y_true, y_pred = build_result_table(
            ctx,
            final_result["forecast_df"]["yhat"].to_numpy(),
        )

        test_xlsx = output_dir / "random_forest_test.xlsx"
        try:
            result_df.to_excel(test_xlsx, index=False, sheet_name="test")
            result_path_key = "test_xlsx"
            result_path = test_xlsx
        except Exception:
            test_csv = output_dir / "random_forest_test.csv"
            result_df.to_csv(test_csv, index=False, encoding="utf-8-sig")
            result_path_key = "test_csv"
            result_path = test_csv

        bp = final_result.get("best_rf_params") or {}
        summary: dict[str, Any] = {
            "backend": "random_forest",
            "task": {
                **ctx.metadata,
                "horizon": ctx.horizon,
                "selection_metric": self.selection_metric,
            },
            "best_params": best_step.evolvable_subjects.params.to_dict(),
            "best_feedback": asdict(best_step.feedback) if best_step.feedback else {},
            "trace_size": len(self.trace),
            "test_metrics": forecast_metrics(y_true, y_pred),
            "rf_params": dict(bp),
            "top_feature_importance": final_result.get("top_feature_importance", {}),
            result_path_key: str(result_path),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

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
            "grid_search_csv": str(grid_path),
            result_path_key: str(result_path),
        }

    def _trace_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx, step in enumerate(self.trace, start=1):
            feedback = step.feedback
            m = feedback.metrics if feedback and feedback.metrics else {}
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
                "n_estimators": m.get("n_estimators"),
                "max_depth": m.get("max_depth"),
                "min_samples_leaf": m.get("min_samples_leaf"),
                "min_samples_split": m.get("min_samples_split"),
                "max_features": m.get("max_features"),
                "max_samples": m.get("max_samples"),
                "message": feedback.message if feedback else "",
            }
            rows.append(row)
        return rows

    @staticmethod
    def _save_plot(series: pd.Series, forecast_df: pd.DataFrame, plot_path: Path) -> None:
        plt.figure(figsize=(14, 6))
        plt.plot(series.index, series.values, label="history", color="tab:blue")
        plt.plot(pd.to_datetime(forecast_df["ds"]), forecast_df["yhat"], label="forecast", color="tab:orange")
        plt.title("Forecast Agent - Auto Random Forest")
        plt.xlabel("Date")
        plt.ylabel("y")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()


__all__ = [
    "RandomForestHyperParams",
    "RandomForestEvaluator",
    "FixedRandomForestStrategy",
    "AutoRandomForestForecastAgent",
]
