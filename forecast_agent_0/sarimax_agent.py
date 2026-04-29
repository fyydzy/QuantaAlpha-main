from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from quantaalpha.forecast_agent.framework import (
    ForecastAgent,
    ForecastEvaluator,
    ForecastFeedback,
    ForecastStep,
    ForecastStrategy,
    ForecastSubjects,
    ForecastTask,
    SarimaxHyperParams,
)


def load_series(task: ForecastTask) -> pd.Series:
    df = pd.read_csv(task.csv_path, parse_dates=[task.ds_col])
    df = df[[task.ds_col, task.y_col]].copy()
    df = df.sort_values(task.ds_col).drop_duplicates(subset=[task.ds_col])
    series = df.set_index(task.ds_col)[task.y_col].asfreq("D").fillna(0.0).astype(float)
    series.index.name = "ds"
    series.name = "y"
    return series


def build_fourier_features(index: pd.Index, weekly_order: int, yearly_order: int) -> pd.DataFrame:
    t = np.arange(len(index), dtype=float)
    features: dict[str, np.ndarray] = {}

    for k in range(1, weekly_order + 1):
        angle = 2 * np.pi * k * t / 7.0
        features[f"weekly_sin_{k}"] = np.sin(angle)
        features[f"weekly_cos_{k}"] = np.cos(angle)

    for k in range(1, yearly_order + 1):
        angle = 2 * np.pi * k * t / 365.25
        features[f"yearly_sin_{k}"] = np.sin(angle)
        features[f"yearly_cos_{k}"] = np.cos(angle)

    return pd.DataFrame(features, index=index)


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-6
    smape = float(np.mean(2.0 * np.abs(err) / denom) * 100.0)
    bias = float(np.mean(y_pred - y_true))
    scale = float(np.mean(np.abs(y_true)) + 1e-6)
    score = 0.7 * smape + 0.3 * (rmse / scale * 100.0)
    return {
        "score": score,
        "smape": smape,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
    }


def build_feedback_message(metrics: dict[str, float], params: SarimaxHyperParams) -> str:
    notes: list[str] = []
    if params.yearly_order < 3:
        notes.append("年度周期阶数偏低")
    if abs(metrics["bias"]) > max(metrics["mae"] * 0.2, 1.0):
        notes.append("预测存在系统性偏差")
    if metrics["smape"] > 35:
        notes.append("验证集误差较高")
    if not notes:
        notes.append("当前参数较稳定")
    return "；".join(notes)


class SarimaxEvaluator(ForecastEvaluator):
    def evaluate(self, task: ForecastTask, subjects: ForecastSubjects) -> ForecastFeedback:
        series = load_series(task)
        if len(series) <= task.validation_days + 30:
            return ForecastFeedback(
                score=float("inf"),
                smape=float("inf"),
                rmse=float("inf"),
                mae=float("inf"),
                bias=float("inf"),
                aic=None,
                success=False,
                message="样本太短，无法做验证集评估",
            )

        params = subjects.params
        train = series.iloc[:-task.validation_days]
        valid = series.iloc[-task.validation_days:]
        full_index = train.index.append(valid.index)
        full_exog = build_fourier_features(full_index, params.weekly_order, params.yearly_order)
        train_exog = full_exog.loc[train.index]
        valid_exog = full_exog.loc[valid.index]

        try:
            model = SARIMAX(
                train,
                exog=train_exog,
                order=(params.p, params.d, params.q),
                seasonal_order=(0, 0, 0, 0),
                trend=params.trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False)
            pred = result.get_forecast(steps=len(valid), exog=valid_exog).predicted_mean
            pred.index = valid.index
            metrics = compute_metrics(valid, pred)
            message = build_feedback_message(metrics, params)
            return ForecastFeedback(
                score=metrics["score"],
                smape=metrics["smape"],
                rmse=metrics["rmse"],
                mae=metrics["mae"],
                bias=metrics["bias"],
                aic=float(result.aic) if np.isfinite(result.aic) else None,
                success=True,
                message=message,
                metrics=metrics,
            )
        except Exception as exc:
            return ForecastFeedback(
                score=float("inf"),
                smape=float("inf"),
                rmse=float("inf"),
                mae=float("inf"),
                bias=float("inf"),
                aic=None,
                success=False,
                message=f"拟合失败: {exc}",
            )


class HeuristicSarimaxStrategy(ForecastStrategy):
    def __init__(
        self,
        p_range: tuple[int, int] = (0, 3),
        d_range: tuple[int, int] = (0, 1),
        q_range: tuple[int, int] = (0, 3),
        weekly_range: tuple[int, int] = (0, 2),
        yearly_range: tuple[int, int] = (2, 6),
    ) -> None:
        self.p_range = p_range
        self.d_range = d_range
        self.q_range = q_range
        self.weekly_range = weekly_range
        self.yearly_range = yearly_range

    def seed_subjects(self, task: ForecastTask) -> list[ForecastSubjects]:
        seeds = [
            SarimaxHyperParams(1, 1, 1, weekly_order=1, yearly_order=3, trend="c"),
            SarimaxHyperParams(2, 1, 1, weekly_order=1, yearly_order=4, trend="c"),
            SarimaxHyperParams(1, 1, 2, weekly_order=1, yearly_order=5, trend="c"),
            SarimaxHyperParams(2, 0, 2, weekly_order=1, yearly_order=4, trend="c"),
        ]
        return [ForecastSubjects(params=s, metadata={"source": "seed"}) for s in seeds]

    def evolve(
        self,
        best_subjects: ForecastSubjects,
        evolving_trace: list[ForecastStep],
        top_k: int = 4,
    ) -> list[ForecastSubjects]:
        best = best_subjects.params
        candidates: dict[tuple[int, int, int, int, int, str], ForecastSubjects] = {}

        def add_candidate(
            p: int,
            d: int,
            q: int,
            weekly_order: int,
            yearly_order: int,
            trend: str,
            reason: str,
        ) -> None:
            if not (self.p_range[0] <= p <= self.p_range[1]):
                return
            if not (self.d_range[0] <= d <= self.d_range[1]):
                return
            if not (self.q_range[0] <= q <= self.q_range[1]):
                return
            if not (self.weekly_range[0] <= weekly_order <= self.weekly_range[1]):
                return
            if not (self.yearly_range[0] <= yearly_order <= self.yearly_range[1]):
                return
            params = SarimaxHyperParams(
                p=p,
                d=d,
                q=q,
                weekly_order=weekly_order,
                yearly_order=yearly_order,
                trend=trend,
            )
            candidates[params.signature()] = ForecastSubjects(
                params=params,
                metadata={"source": "evolve", "reason": reason},
            )

        base_reason = "围绕当前最优参数做局部搜索"
        for delta in (-1, 1):
            add_candidate(best.p + delta, best.d, best.q, best.weekly_order, best.yearly_order, best.trend, f"{base_reason}: 调整 p")
            add_candidate(best.p, best.d, best.q + delta, best.weekly_order, best.yearly_order, best.trend, f"{base_reason}: 调整 q")
            add_candidate(best.p, best.d, best.q, best.weekly_order + delta, best.yearly_order, best.trend, f"{base_reason}: 调整周周期")
            add_candidate(best.p, best.d, best.q, best.weekly_order, best.yearly_order + delta, best.trend, f"{base_reason}: 调整年周期")

        add_candidate(best.p, 1 - best.d, best.q, best.weekly_order, best.yearly_order, best.trend, "切换差分阶数")
        add_candidate(best.p, best.d, best.q, best.weekly_order, best.yearly_order, "n" if best.trend == "c" else "c", "切换趋势项")
        add_candidate(best.p + 1, best.d, best.q + 1, best.weekly_order, min(best.yearly_order + 1, self.yearly_range[1]), best.trend, "同时增强 AR/MA 与年周期")
        add_candidate(max(best.p - 1, 0), best.d, max(best.q - 1, 0), best.weekly_order, max(best.yearly_order - 1, self.yearly_range[0]), best.trend, "同时简化模型复杂度")

        tried = {step.evolvable_subjects.params.signature() for step in evolving_trace}
        fresh = [subject for sig, subject in candidates.items() if sig not in tried]
        return fresh[:top_k]


class AutoSarimaxForecastAgent(ForecastAgent):
    def __init__(
        self,
        max_loops: int = 6,
        beam_width: int = 4,
        evolving_strategy: ForecastStrategy | None = None,
        evaluator: ForecastEvaluator | None = None,
    ) -> None:
        super().__init__(
            max_loops=max_loops,
            evolving_strategy=evolving_strategy or HeuristicSarimaxStrategy(),
            evaluator=evaluator or SarimaxEvaluator(),
        )
        self.beam_width = beam_width

    def run(self, task: ForecastTask) -> dict[str, Any]:
        frontier = self.evolving_strategy.seed_subjects(task)
        best_step: ForecastStep | None = None

        for loop_idx in range(self.max_loops):
            scored_steps: list[ForecastStep] = []
            for subject in frontier:
                feedback = self.evaluator.evaluate(task, subject)
                reason = subject.metadata.get("reason", f"第 {loop_idx + 1} 轮候选")
                step = ForecastStep(
                    evolvable_subjects=subject,
                    feedback=feedback,
                    proposal_reason=reason,
                )
                self.evolving_trace.append(step)
                scored_steps.append(step)

            successful = [step for step in scored_steps if step.feedback and step.feedback.success]
            if successful:
                successful.sort(key=lambda x: x.feedback.score)  # type: ignore[arg-type]
                round_best = successful[0]
                if best_step is None or round_best.feedback.score < best_step.feedback.score:  # type: ignore[union-attr]
                    best_step = round_best

            if best_step is None:
                break

            frontier = self.evolving_strategy.evolve(
                best_subjects=best_step.evolvable_subjects,
                evolving_trace=self.evolving_trace,
                top_k=self.beam_width,
            )
            if not frontier:
                break

        if best_step is None or best_step.feedback is None:
            raise RuntimeError("Forecast agent failed to find a valid SARIMAX configuration.")

        final_result = self.refit_and_forecast(task, best_step.evolvable_subjects.params)
        output_paths = self.save_outputs(task, best_step, final_result)
        forecast_head_df = final_result["forecast_df"].head(10).copy()
        forecast_head_df["ds"] = forecast_head_df["ds"].astype(str)
        return {
            "best_params": best_step.evolvable_subjects.params.to_dict(),
            "best_feedback": asdict(best_step.feedback),
            "trace_size": len(self.evolving_trace),
            "forecast_head": forecast_head_df.to_dict(orient="records"),
            "outputs": output_paths,
        }

    def refit_and_forecast(self, task: ForecastTask, params: SarimaxHyperParams) -> dict[str, Any]:
        series = load_series(task)
        future_index = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=task.horizon, freq="D")
        full_index = series.index.append(future_index)
        full_exog = build_fourier_features(full_index, params.weekly_order, params.yearly_order)
        train_exog = full_exog.loc[series.index]
        future_exog = full_exog.loc[future_index]

        model = SARIMAX(
            series,
            exog=train_exog,
            order=(params.p, params.d, params.q),
            seasonal_order=(0, 0, 0, 0),
            trend=params.trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False)
        forecast_res = result.get_forecast(steps=task.horizon, exog=future_exog)
        forecast_mean = forecast_res.predicted_mean
        forecast_ci = forecast_res.conf_int()

        forecast_df = pd.DataFrame(
            {
                "ds": forecast_mean.index,
                "yhat": forecast_mean.values,
                "yhat_lower": forecast_ci.iloc[:, 0].values,
                "yhat_upper": forecast_ci.iloc[:, 1].values,
            }
        )
        return {
            "series": series,
            "forecast_df": forecast_df,
            "aic": float(result.aic) if np.isfinite(result.aic) else None,
        }

    def save_outputs(
        self,
        task: ForecastTask,
        best_step: ForecastStep,
        final_result: dict[str, Any],
    ) -> dict[str, str]:
        output_dir = Path(task.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        trace_rows: list[dict[str, Any]] = []
        for idx, step in enumerate(self.evolving_trace, start=1):
            feedback = step.feedback
            row = {
                "trial": idx,
                "proposal_reason": step.proposal_reason,
                **step.evolvable_subjects.params.to_dict(),
                "success": feedback.success if feedback else False,
                "score": feedback.score if feedback else None,
                "smape": feedback.smape if feedback else None,
                "rmse": feedback.rmse if feedback else None,
                "mae": feedback.mae if feedback else None,
                "bias": feedback.bias if feedback else None,
                "aic": feedback.aic if feedback else None,
                "message": feedback.message if feedback else "",
            }
            trace_rows.append(row)

        trace_path = output_dir / "search_trace.csv"
        forecast_path = output_dir / "forecast.csv"
        summary_path = output_dir / "best_summary.json"
        plot_path = output_dir / "forecast_plot.png"

        pd.DataFrame(trace_rows).to_csv(trace_path, index=False, encoding="utf-8-sig")
        final_result["forecast_df"].to_csv(forecast_path, index=False, encoding="utf-8-sig")

        summary = {
            "task": {
                "csv_path": str(task.csv_path),
                "horizon": task.horizon,
                "validation_days": task.validation_days,
            },
            "best_params": best_step.evolvable_subjects.params.to_dict(),
            "best_feedback": asdict(best_step.feedback) if best_step.feedback else {},
            "refit_aic": final_result["aic"],
            "trace_size": len(self.evolving_trace),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        series = final_result["series"]
        forecast_df = final_result["forecast_df"]
        x_forecast = mdates.date2num(pd.to_datetime(forecast_df["ds"]).to_numpy())
        yhat = np.asarray(forecast_df["yhat"], dtype=float)
        yhat_lower = np.asarray(forecast_df["yhat_lower"], dtype=float)
        yhat_upper = np.asarray(forecast_df["yhat_upper"], dtype=float)
        fill_mask = np.ones(len(x_forecast), dtype=bool)
        plt.figure(figsize=(14, 6))
        plt.plot(series.index, series.values, label="history", color="tab:blue")
        plt.plot(pd.to_datetime(forecast_df["ds"]), yhat, label="forecast", color="tab:orange")
        plt.fill_between(
            x_forecast,
            yhat_lower,
            yhat_upper,
            where=fill_mask,
            color="tab:orange",
            alpha=0.2,
            label="95% CI",
        )
        plt.gca().xaxis_date()
        plt.title("Forecast Agent - Auto SARIMAX on test.csv")
        plt.xlabel("Date")
        plt.ylabel("y")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

        return {
            "trace_csv": str(trace_path),
            "forecast_csv": str(forecast_path),
            "summary_json": str(summary_path),
            "plot_png": str(plot_path),
        }
