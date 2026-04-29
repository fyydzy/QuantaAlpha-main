"""TimesFM 预测后端。

当前流程固定为月度 Excel 数据：

1. 使用 ``as_of_month`` 及以前的数据作为训练序列；
2. 一次性预测 bridge + test 月份；
3. 仅用 test 月份真实值计算指标，并据此选择最优 ``context_len``。
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
    TaskContext,
    build_result_table,
    compute_score_metrics,
    forecast_metrics,
    load_task_context,
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


_PATCH_LEN = 3


@dataclass(frozen=True)
class TimesFmHyperParams:
    """TimesFM 搜索参数。

    自动搜索只枚举 ``context_len``；其余字段保留为显式实验入口。
    """

    context_len: int = 111
    freq: int = 2
    normalize: bool = False
    log_transform: bool = False

    def signature(self) -> tuple[int, int, bool, bool]:
        return (self.context_len, self.freq, self.normalize, self.log_transform)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _round_up(value: int, base: int) -> int:
    return ((value + base - 1) // base) * base


def _build_feedback_message(metrics: dict[str, float], params: TimesFmHyperParams) -> str:
    notes: list[str] = []
    if metrics["smape"] > 35:
        notes.append("测试集误差较高")
    if abs(metrics["bias"]) > max(metrics["mae"] * 0.2, 1.0):
        notes.append("预测存在系统性偏差")
    if params.context_len < 24:
        notes.append("上下文长度偏短")
    return "；".join(notes) if notes else "TimesFM 当前配置较稳定"


class _TimesFmModelCache:
    """按 (预测步长桶, 设备类型) 缓存 TimesFM 模型实例。"""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, str], Any] = {}

    def get(self, horizon_len: int, backend: str = "cpu") -> Any:
        horizon_bucket = max(_round_up(horizon_len, _PATCH_LEN), _PATCH_LEN)
        key = (horizon_bucket, backend)
        if key in self._cache:
            return self._cache[key]

        import timesfm  # type: ignore[import-not-found]  # 可选依赖，按需延迟导入

        model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=backend,
                per_core_batch_size=1,
                horizon_len=horizon_bucket,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-1.0-200m-pytorch",
            ),
        )
        self._cache[key] = model
        return model


_MODEL_CACHE = _TimesFmModelCache()


def _prepare_context(values: np.ndarray, context_len: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    tail = values[-context_len:] if context_len > 0 else values
    pad = (-len(tail)) % _PATCH_LEN
    if pad:
        tail = np.concatenate([np.full(pad, np.nan, dtype=np.float32), tail])
    return tail


def _forecast_once(
    train_values: np.ndarray,
    horizon_len: int,
    params: TimesFmHyperParams,
    backend: str = "cpu",
) -> np.ndarray:
    x = np.asarray(train_values, dtype=np.float64)

    if params.log_transform:
        x = np.log1p(np.clip(x, a_min=0.0, a_max=None))

    mu = 0.0
    sigma = 1.0
    if params.normalize:
        mu = float(np.nanmean(x))
        sigma = float(np.nanstd(x))
        if not np.isfinite(sigma) or sigma < 1e-9:
            sigma = 1.0
        x = (x - mu) / sigma

    context = _prepare_context(x.astype(np.float32), params.context_len)
    model = _MODEL_CACHE.get(horizon_len=horizon_len, backend=backend)
    pred = model.forecast([context], freq=[params.freq])
    pred_arr = np.asarray(pred[0] if isinstance(pred, tuple) else pred)
    yhat = (pred_arr[0] if pred_arr.ndim == 2 else pred_arr)[:horizon_len].astype(np.float64)

    if params.normalize:
        yhat = yhat * sigma + mu
    if params.log_transform:
        yhat = np.expm1(yhat)
    return yhat


class TimesFmEvaluator(ForecastEvaluator):
    def __init__(self, backend: str = "cpu") -> None:
        self.backend = backend

    def evaluate(self, task: ForecastTask, subjects: ForecastSubjects) -> ForecastFeedback:
        ctx = load_task_context(task)
        if len(ctx.series) < 12:
            return _failed_feedback("训练样本太短，无法评估 TimesFM 参数")

        params: TimesFmHyperParams = subjects.params  # type: ignore[assignment]
        try:
            yhat = _forecast_once(
                ctx.series.values,
                horizon_len=ctx.horizon,
                params=params,
                backend=self.backend,
            )
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
                metrics=metrics,
            )
        except Exception as exc:  # noqa: BLE001 - 搜索阶段统一转成失败反馈
            return _failed_feedback(f"TimesFM 推理失败: {exc}")


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


class GridTimesFmStrategy(ForecastStrategy):
    def __init__(
        self,
        context_choices: tuple[int, ...] = (
            60,
            63,
            66,
            69,
            72,
            75,
            78,
            81,
            84,
            87,
            90,
            93,
            96,
            99,
            102,
            105,
            108,
            111,
        ),
    ) -> None:
        self.context_choices = context_choices

    def seed_subjects(self, task: ForecastTask) -> list[ForecastSubjects]:
        params_pool: dict[tuple[int, int, bool, bool], TimesFmHyperParams] = {}
        params_pool[TimesFmHyperParams().signature()] = TimesFmHyperParams()
        for context_len in self.context_choices:
            params = TimesFmHyperParams(context_len=context_len)
            params_pool[params.signature()] = params

        return [
            ForecastSubjects(
                params=params,
                metadata={"reason": f"网格候选 context_len={params.context_len}"},
            )
            for params in params_pool.values()
        ]


class AutoTimesFmForecastAgent(ForecastAgent):
    def __init__(
        self,
        strategy: ForecastStrategy | None = None,
        evaluator: ForecastEvaluator | None = None,
        backend: str = "cpu",
        selection_metric: str = "mape",
    ) -> None:
        super().__init__(
            strategy=strategy or GridTimesFmStrategy(),
            evaluator=evaluator or TimesFmEvaluator(backend=backend),
        )
        metric = (selection_metric or "mape").lower()
        if metric not in {"mape", "score"}:
            raise ValueError(f"Unsupported selection_metric: {selection_metric}")
        self.backend = backend
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
                proposal_reason=subject.metadata.get("reason", f"网格候选 #{idx}"),
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
        forecast_head["ds"] = forecast_head["ds"].astype(str)
        return {
            "backend": "timesfm",
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
        return f"TimesFM forecast agent failed to find a valid configuration.{suffix}"

    def refit_and_forecast(self, task: ForecastTask, params: TimesFmHyperParams) -> dict[str, Any]:
        ctx = load_task_context(task)
        yhat = _forecast_once(
            ctx.series.values,
            horizon_len=ctx.horizon,
            params=params,
            backend=self.backend,
        )
        forecast_df = pd.DataFrame({"ds": ctx.future_index, "yhat": yhat})
        return {"series": ctx.series, "forecast_df": forecast_df, "ctx": ctx}

    def save_outputs(
        self,
        task: ForecastTask,
        best_step: ForecastStep,
        final_result: dict[str, Any],
    ) -> dict[str, str]:
        output_dir = Path(task.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        trace_path = output_dir / "timesfm_search_trace.csv"
        forecast_path = output_dir / "timesfm_forecast.csv"
        summary_path = output_dir / "timesfm_best_summary.json"
        plot_path = output_dir / "timesfm_forecast_plot.png"

        pd.DataFrame(self._trace_rows()).to_csv(trace_path, index=False, encoding="utf-8-sig")
        final_result["forecast_df"].to_csv(forecast_path, index=False, encoding="utf-8-sig")

        ctx: TaskContext = final_result["ctx"]
        result_df, y_true, y_pred = build_result_table(
            ctx,
            final_result["forecast_df"]["yhat"].to_numpy(),
        )

        test_xlsx = output_dir / "timesfm_test.xlsx"
        try:
            result_df.to_excel(test_xlsx, index=False, sheet_name="test")
            result_path_key = "test_xlsx"
            result_path = test_xlsx
        except Exception:
            test_csv = output_dir / "timesfm_test.csv"
            result_df.to_csv(test_csv, index=False, encoding="utf-8-sig")
            result_path_key = "test_csv"
            result_path = test_csv

        summary: dict[str, Any] = {
            "backend": "timesfm",
            "task": {
                **ctx.metadata,
                "horizon": ctx.horizon,
                "selection_metric": self.selection_metric,
            },
            "best_params": best_step.evolvable_subjects.params.to_dict(),
            "best_feedback": asdict(best_step.feedback) if best_step.feedback else {},
            "trace_size": len(self.trace),
            "test_metrics": forecast_metrics(y_true, y_pred),
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
                    "message": feedback.message if feedback else "",
                }
            )
        return rows

    @staticmethod
    def _save_plot(series: pd.Series, forecast_df: pd.DataFrame, plot_path: Path) -> None:
        plt.figure(figsize=(14, 6))
        plt.plot(series.index, series.values, label="history", color="tab:blue")
        plt.plot(pd.to_datetime(forecast_df["ds"]), forecast_df["yhat"], label="forecast", color="tab:orange")
        plt.title("Forecast Agent - Auto TimesFM")
        plt.xlabel("Date")
        plt.ylabel("y")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()


__all__ = [
    "TimesFmHyperParams",
    "TimesFmEvaluator",
    "GridTimesFmStrategy",
    "AutoTimesFmForecastAgent",
]
