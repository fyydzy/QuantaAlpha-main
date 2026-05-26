"""PyTorch LSTM 预测后端（与 Ridge/Lasso 同一套 Forecast Agent 流程；训练循环与 run_lstm_0 一致）。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from quantaalpha.forecast_agent.data import (
    MONTH_COL,
    TARGET_COL,
    MIN_HISTORY_PERIODS,
    TaskContext,
    build_result_table,
    compute_score_metrics,
    forecast_metrics,
    format_month_ds_for_display,
    filter_through_period,
    load_tabular_feature_frame,
    load_task_context,
    period_to_lstm_slot,
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

# ---------- 与 run_lstm_0 一致 ----------
SELECTED_FEATURES = ["Lag_36", "HDD", "is_heating_season"]
SEQ_LENGTH = 3
HIDDEN_SIZE = 16
NUM_LAYERS = 1
LEARNING_RATE = 0.005
EPOCHS = 300
BATCH_SIZE = 16
RANDOM_SEED = 42
EMBEDDING_DIM = 4


def _torch_modules():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("LSTM 后端需要安装 torch：pip install torch") from exc
    return torch, nn, DataLoader, TensorDataset


@dataclass(frozen=True)
class LstmHyperParams:
    context_len: int = 111

    def signature(self) -> tuple[int]:
        return (self.context_len,)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_lstm_module_class():
    torch, nn, _DataLoader, _TensorDataset = _torch_modules()

    class _GasForecastLSTM(nn.Module):
        def __init__(self, continuous_size: int, hidden_size: int, num_layers: int, emb_dim: int) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.month_emb = nn.Embedding(num_embeddings=12 * 3, embedding_dim=emb_dim)
            total_input_size = continuous_size + emb_dim
            self.lstm = nn.LSTM(
                input_size=total_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x_cont: Any, x_month: Any) -> Any:
            month_vecs = self.month_emb(x_month)
            x = torch.cat((x_cont, month_vecs), dim=-1)
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            out, _ = self.lstm(x, (h0, c0))
            return self.fc(out[:, -1, :])

    return _GasForecastLSTM


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


def _build_feedback_message(metrics: dict[str, float], params: LstmHyperParams) -> str:
    notes: list[str] = []
    if metrics["smape"] > 35:
        notes.append("测试集误差较高")
    if abs(metrics["bias"]) > max(metrics["mae"] * 0.2, 1.0):
        notes.append("预测存在系统性偏差")
    if params.context_len < MIN_HISTORY_PERIODS:
        notes.append("上下文长度偏短")
    return "；".join(notes) if notes else "LSTM 当前配置较稳定"


def _load_feature_frame(path: str) -> pd.DataFrame:
    df_features = load_tabular_feature_frame(path)
    df_features["period_slot"] = df_features[MONTH_COL].astype(str).map(period_to_lstm_slot).astype(np.int64)
    return df_features


def _apply_context_window(df: pd.DataFrame, context_len: int) -> pd.DataFrame:
    if context_len <= 0 or len(df) <= context_len:
        return df.copy()
    return df.iloc[-context_len:].copy()


def _prepare_lstm_frames(
    ctx: TaskContext,
    params: LstmHyperParams,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    path = str(ctx.metadata["excel_path"])
    feature_df = _load_feature_frame(path)

    missing_features = [f for f in SELECTED_FEATURES if f not in feature_df.columns]
    if missing_features:
        raise ValueError(f"特征工程后缺少 LSTM 所需特征: {missing_features}")

    as_of = str(ctx.metadata["as_of_month"])
    train_ctx = filter_through_period(feature_df, as_of)
    train_ctx = _apply_context_window(train_ctx, params.context_len)
    min_periods = max(MIN_HISTORY_PERIODS, SEQ_LENGTH + 2)
    if len(train_ctx) < min_periods:
        raise ValueError(
            f"context={params.context_len} 训练旬数 {len(train_ctx)}，"
            f"需至少 {min_periods} 旬",
        )

    max_fc = max(ctx.forecast_months)
    work_df = filter_through_period(feature_df, max_fc).reset_index(drop=True)
    if len(work_df) < SEQ_LENGTH + len(ctx.forecast_months):
        raise ValueError("work_df 过短，无法构造预测序列。")

    return feature_df, train_ctx, work_df, list(ctx.forecast_months)


def _create_3d_sequences(
    x_cont_arr: np.ndarray,
    x_month_arr: np.ndarray,
    y_arr: np.ndarray,
    month_arr: np.ndarray,
    seq_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs_cont: list[np.ndarray] = []
    xs_month: list[np.ndarray] = []
    ys: list[float] = []
    ms: list[str] = []

    for i in range(len(x_cont_arr) - seq_length):
        xs_cont.append(x_cont_arr[i : i + seq_length])
        xs_month.append(x_month_arr[i : i + seq_length])
        ys.append(float(y_arr[i + seq_length]))
        ms.append(str(month_arr[i + seq_length]))

    return np.asarray(xs_cont), np.asarray(xs_month), np.asarray(ys), np.asarray(ms, dtype=object)


def _state_dict_cpu(model: Any) -> dict[str, Any]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _train_lstm(
    train_ctx: pd.DataFrame,
    work_df: pd.DataFrame,
    forecast_months: list[str],
) -> tuple[Any, StandardScaler, StandardScaler, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """与随机森林一致：仅使用 as_of 之后再经 context_len 截窗的 train_ctx。
    StandardScaler 仅在 train_ctx 所含旬对应的行上 fit；LSTM 仅用「预测目标旬落在 train_ctx」的序列做训练。
    work_df 仍可包含更早旬，仅为构造 SEQ_LENGTH 输入与预测未来旬所需。"""
    torch, nn, DataLoader, TensorDataset = _torch_modules()
    LSTMCls = _build_lstm_module_class()

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    month_arr = work_df[MONTH_COL].astype(str).to_numpy()
    x_cont_all = work_df[SELECTED_FEATURES].to_numpy(dtype=float)
    x_month_all = work_df["period_slot"].to_numpy(dtype=np.int64)
    y_all = work_df[TARGET_COL].to_numpy(dtype=float)

    train_month_mask = work_df[MONTH_COL].astype(str).isin(train_ctx[MONTH_COL].astype(str)).to_numpy()
    if not np.any(train_month_mask):
        raise ValueError("work_df 中找不到 train_ctx 对应旬。")

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(x_cont_all[train_month_mask])
    scaler_y.fit(y_all[train_month_mask].reshape(-1, 1))

    x_cont_scaled = scaler_x.transform(x_cont_all)
    y_scaled = scaler_y.transform(y_all.reshape(-1, 1)).flatten()

    x_cont_3d, x_month_3d, y_1d, y_month = _create_3d_sequences(
        x_cont_scaled, x_month_all, y_scaled, month_arr, SEQ_LENGTH
    )
    if len(x_cont_3d) == 0:
        raise ValueError("样本量不足，无法构造 LSTM 序列。")

    train_months_arr = np.asarray(train_ctx[MONTH_COL].astype(str).tolist(), dtype=object)
    seq_train_mask = np.isin(y_month, train_months_arr)
    if not np.any(seq_train_mask):
        raise ValueError("无训练序列（请检查 context_len 与 as_of）。")

    x_cont_train = x_cont_3d[seq_train_mask]
    x_month_train = x_month_3d[seq_train_mask]
    y_train = y_1d[seq_train_mask]

    x_cont_train_tensor = torch.tensor(x_cont_train, dtype=torch.float32)
    x_month_train_tensor = torch.tensor(x_month_train, dtype=torch.long)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(x_cont_train_tensor, x_month_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMCls(
        continuous_size=len(SELECTED_FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        emb_dim=EMBEDDING_DIM,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.MSELoss()

    model.train()
    for _epoch in range(EPOCHS):
        for b_x_cont, b_x_month, b_y in train_loader:
            optimizer.zero_grad()
            outputs = model(b_x_cont, b_x_month)
            loss = criterion(outputs, b_y)
            loss.backward()
            optimizer.step()

    return model, scaler_x, scaler_y, x_cont_3d, x_month_3d, y_1d, y_month


def _predict_future(
    model: Any,
    scaler_x: StandardScaler,
    scaler_y: StandardScaler,
    work_df: pd.DataFrame,
    y_month: np.ndarray,
    x_cont_3d: np.ndarray,
    x_month_3d: np.ndarray,
    y_1d: np.ndarray,
    forecast_months: list[str],
) -> np.ndarray:
    torch, _nn, _DataLoader, _TensorDataset = _torch_modules()
    fc_arr = np.asarray(forecast_months, dtype=str)
    seq_eval_mask = np.isin(y_month, fc_arr)
    if not np.any(seq_eval_mask):
        raise ValueError("无预测序列可对应 forecast_months。")

    x_cont_test = x_cont_3d[seq_eval_mask]
    x_month_test = x_month_3d[seq_eval_mask]
    y_test_scaled = y_1d[seq_eval_mask]
    y_test_month = y_month[seq_eval_mask]

    x_cont_test_tensor = torch.tensor(x_cont_test, dtype=torch.float32)
    x_month_test_tensor = torch.tensor(x_month_test, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        test_preds_scaled = model(x_cont_test_tensor, x_month_test_tensor).cpu().numpy().flatten()

    test_preds = scaler_y.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()
    _ = scaler_y.inverse_transform(np.asarray(y_test_scaled, dtype=float).reshape(-1, 1)).flatten()

    month_to_pred: dict[str, float] = {}
    for m, p in zip(y_test_month.astype(str), test_preds):
        month_to_pred[str(m)] = float(p)

    ordered: list[float] = []
    for m in forecast_months:
        if m not in month_to_pred:
            raise ValueError(f"LSTM 预测缺少旬: {m}")
        ordered.append(month_to_pred[m])
    return np.asarray(ordered, dtype=float)


class LstmEvaluator(ForecastEvaluator):
    """evaluate 结束后缓存 CPU state_dict + scaler，refit 只做前向与拼预测，避免重复 300 epoch。"""

    def __init__(self) -> None:
        self._last_bundle: tuple[dict[str, Any], StandardScaler, StandardScaler] | None = None

    def evaluate(self, task: ForecastTask, subjects: ForecastSubjects) -> ForecastFeedback:
        ctx = load_task_context(task)
        params: LstmHyperParams = subjects.params  # type: ignore[assignment]
        self._last_bundle = None
        try:
            _feature_df, train_ctx, work_df, forecast_months = _prepare_lstm_frames(ctx, params)
            model, scaler_x, scaler_y, x3, xm3, y1, ym = _train_lstm(train_ctx, work_df, forecast_months)
            self._last_bundle = (_state_dict_cpu(model), scaler_x, scaler_y)

            yhat = _predict_future(model, scaler_x, scaler_y, work_df, ym, x3, xm3, y1, forecast_months)
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
                metrics={**metrics},
            )
        except ImportError as exc:
            return _failed_feedback(str(exc))
        except Exception as exc:  # noqa: BLE001
            self._last_bundle = None
            return _failed_feedback(f"LSTM 推理失败: {exc}")


class FixedLstmStrategy(ForecastStrategy):
    def __init__(self, context_len: int = 111) -> None:
        self.context_len = int(context_len)

    def seed_subjects(self, task: ForecastTask) -> list[ForecastSubjects]:
        params = LstmHyperParams(context_len=self.context_len)
        return [
            ForecastSubjects(
                params=params,
                metadata={"reason": f"固定 context_len={params.context_len}"},
            )
        ]


class AutoLstmForecastAgent(ForecastAgent):
    def __init__(
        self,
        context_len: int = 111,
        strategy: ForecastStrategy | None = None,
        evaluator: ForecastEvaluator | None = None,
        selection_metric: str = "mape",
    ) -> None:
        super().__init__(
            strategy=strategy or FixedLstmStrategy(context_len=context_len),
            evaluator=evaluator or LstmEvaluator(),
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
            "backend": "lstm",
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
        return f"LSTM forecast agent failed to find a valid configuration.{suffix}"

    def refit_and_forecast(self, task: ForecastTask, params: LstmHyperParams) -> dict[str, Any]:
        LSTMCls = _build_lstm_module_class()
        ctx = load_task_context(task)
        _feature_df, train_ctx, work_df, forecast_months = _prepare_lstm_frames(ctx, params)

        ev = self.evaluator
        bundle = getattr(ev, "_last_bundle", None) if isinstance(ev, LstmEvaluator) else None
        if bundle is not None:
            state_d, scaler_x, scaler_y = bundle
            model = LSTMCls(
                continuous_size=len(SELECTED_FEATURES),
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                emb_dim=EMBEDDING_DIM,
            )
            model.load_state_dict(state_d)
            month_arr = work_df[MONTH_COL].astype(str).to_numpy()
            x_cont_all = work_df[SELECTED_FEATURES].to_numpy(dtype=float)
            x_month_all = work_df["period_slot"].to_numpy(dtype=np.int64)
            y_all = work_df[TARGET_COL].to_numpy(dtype=float)
            x_cont_scaled = scaler_x.transform(x_cont_all)
            y_scaled = scaler_y.transform(y_all.reshape(-1, 1)).flatten()
            x3, xm3, y1, ym = _create_3d_sequences(
                x_cont_scaled, x_month_all, y_scaled, month_arr, SEQ_LENGTH
            )
            yhat = _predict_future(model, scaler_x, scaler_y, work_df, ym, x3, xm3, y1, forecast_months)
        else:
            model, scaler_x, scaler_y, x3, xm3, y1, ym = _train_lstm(train_ctx, work_df, forecast_months)
            yhat = _predict_future(model, scaler_x, scaler_y, work_df, ym, x3, xm3, y1, forecast_months)

        forecast_df = pd.DataFrame({"ds": ctx.future_index, "yhat": yhat})
        return {
            "series": ctx.series,
            "forecast_df": forecast_df,
            "ctx": ctx,
            "lstm_config": {
                "selected_features": list(SELECTED_FEATURES),
                "seq_length": SEQ_LENGTH,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "embedding_dim": EMBEDDING_DIM,
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

        trace_path = output_dir / "lstm_search_trace.csv"
        forecast_path = output_dir / "lstm_forecast.csv"
        summary_path = output_dir / "lstm_best_summary.json"
        plot_path = output_dir / "lstm_forecast_plot.png"

        pd.DataFrame(self._trace_rows()).to_csv(trace_path, index=False, encoding="utf-8-sig")
        final_result["forecast_df"].to_csv(forecast_path, index=False, encoding="utf-8-sig")

        ctx: TaskContext = final_result["ctx"]
        result_df, y_true, y_pred = build_result_table(
            ctx,
            final_result["forecast_df"]["yhat"].to_numpy(),
        )
        test_xlsx = output_dir / "lstm_test.xlsx"
        try:
            result_df.to_excel(test_xlsx, index=False, sheet_name="test")
            result_path_key = "test_xlsx"
            result_path = test_xlsx
        except Exception:
            test_csv = output_dir / "lstm_test.csv"
            result_df.to_csv(test_csv, index=False, encoding="utf-8-sig")
            result_path_key = "test_csv"
            result_path = test_csv

        summary: dict[str, Any] = {
            "backend": "lstm",
            "task": {
                **ctx.metadata,
                "horizon": ctx.horizon,
                "selection_metric": self.selection_metric,
            },
            "best_params": best_step.evolvable_subjects.params.to_dict(),
            "best_feedback": asdict(best_step.feedback) if best_step.feedback else {},
            "trace_size": len(self.trace),
            "test_metrics": forecast_metrics(y_true, y_pred),
            "lstm_config": final_result.get("lstm_config", {}),
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
        plt.title("Forecast Agent - Auto LSTM")
        plt.xlabel("Date")
        plt.ylabel("y")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()


__all__ = [
    "LstmHyperParams",
    "LstmEvaluator",
    "FixedLstmStrategy",
    "AutoLstmForecastAgent",
]
